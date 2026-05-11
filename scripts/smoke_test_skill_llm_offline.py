"""
smoke_test_skill_llm_offline.py

End-to-end smoke test for the Skill-LLM offline-batch integration.
Validates the local-side code paths without needing a Kaggle batch run
to have happened. Creates a tiny synthetic extractions JSONL, exercises
the loader + ModelManager wiring, and reports pass/fail per assertion.

Run:
    python scripts/smoke_test_skill_llm_offline.py

What it checks:
    1. SkillLLMOfflineExtractor loads a JSONL with mixed records correctly.
    2. Lookup by sentence_id (canonical Phase 1.1 path) returns the right item.
    3. Lookup by sentence_text fallback works when sentence_id is empty.
    4. Missing records return empty lists (not None, not exception).
    5. coverage() correctly reports present / missing counts.
    6. ModelManager.extract_with_skill_llm_offline returns the same shape
       as ModelManager.extract_with_bert -- dicts with "text" + "confidence"
       keys -- so the rest of the pipeline can consume it unchanged.
    7. Empty SKILL/KNOWLEDGE arrays in the source JSONL produce empty
       lists (not exceptions).
    8. Items with empty skill_span strings are filtered out.

Exits 0 on all-pass, 1 on any failure. Used as a unit-test surrogate
since the project doesn't have a pytest harness for pipeline.py yet.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path so imports work when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


SYNTHETIC_EXTRACTIONS = [
    # Happy path: sentence_id + non-empty SKILL + KNOWLEDGE
    {
        "sentence_id": "job_smoke_0000",
        "sentence_text": "Strong Python skills required for backend development.",
        "SKILL": [
            {"skill_span": "backend development", "context": "for backend development"}
        ],
        "KNOWLEDGE": [
            {"skill_span": "Python", "context": "Strong Python skills required"}
        ],
        "model": "skill_llm_8b_lora_v1",
        "extracted_at": "2026-05-12T03:00:00Z",
    },
    # Sentence with multiple skills + knowledge
    {
        "sentence_id": "job_smoke_0001",
        "sentence_text": "Experience designing REST APIs and implementing CI/CD pipelines is a plus.",
        "SKILL": [
            {"skill_span": "designing REST APIs", "context": "Experience designing REST APIs and"},
            {"skill_span": "implementing CI/CD pipelines", "context": "and implementing CI/CD pipelines is"},
        ],
        "KNOWLEDGE": [
            {"skill_span": "REST APIs", "context": "designing REST APIs and"},
            {"skill_span": "CI/CD", "context": "implementing CI/CD pipelines"},
        ],
        "model": "skill_llm_8b_lora_v1",
        "extracted_at": "2026-05-12T03:00:00Z",
    },
    # No extractions (e.g., boilerplate sentence that survived the relevance filter)
    {
        "sentence_id": "job_smoke_0002",
        "sentence_text": "We offer competitive compensation and a flexible work environment.",
        "SKILL": [],
        "KNOWLEDGE": [],
        "model": "skill_llm_8b_lora_v1",
        "extracted_at": "2026-05-12T03:00:00Z",
    },
    # No sentence_id but sentence_text set (legacy / fallback path)
    {
        "sentence_id": "",
        "sentence_text": "Familiarity with Docker containers is required.",
        "SKILL": [],
        "KNOWLEDGE": [
            {"skill_span": "Docker containers", "context": "Familiarity with Docker containers"}
        ],
        "model": "skill_llm_8b_lora_v1",
        "extracted_at": "2026-05-12T03:00:00Z",
    },
    # Edge: item with empty skill_span -- should be filtered out by loader
    {
        "sentence_id": "job_smoke_0003",
        "sentence_text": "Test sentence with a malformed extraction.",
        "SKILL": [
            {"skill_span": "", "context": "garbage"},
            {"skill_span": "valid skill", "context": "valid"},
        ],
        "KNOWLEDGE": [],
        "model": "skill_llm_8b_lora_v1",
        "extracted_at": "2026-05-12T03:00:00Z",
    },
]


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

_PASS = 0
_FAIL = 0
_FAIL_NAMES: list[str] = []


def check(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        print(f"  [PASS] {name}")
        _PASS += 1
    else:
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")
        _FAIL += 1
        _FAIL_NAMES.append(name)


def write_synthetic_jsonl(path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in SYNTHETIC_EXTRACTIONS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_loader_basic(extractor) -> None:
    print("\n[TEST] SkillLLMOfflineExtractor basic load + lookup")

    check(
        "loads all 5 records",
        len(extractor) >= 4,  # 4 with sentence_id; 5th is sentence_id="" so it's text-only
        f"got len={len(extractor)}, expected >= 4",
    )


def test_lookup_by_sentence_id(extractor) -> None:
    print("\n[TEST] lookup by sentence_id (canonical path)")

    skills, knowledge = extractor.extract(
        "Strong Python skills required for backend development.",
        "job_smoke_0000",
    )
    check("happy path returns 1 skill", len(skills) == 1, f"got {len(skills)} skills")
    check("happy path returns 1 knowledge", len(knowledge) == 1, f"got {len(knowledge)} knowledge")
    check(
        "skill text is correct",
        skills and skills[0].get("text") == "backend development",
        f"got text={skills[0].get('text') if skills else None!r}",
    )
    check(
        "knowledge text is correct",
        knowledge and knowledge[0].get("text") == "Python",
        f"got text={knowledge[0].get('text') if knowledge else None!r}",
    )
    check(
        "confidence is float in [0, 1]",
        skills and isinstance(skills[0].get("confidence"), float) and 0.0 <= skills[0]["confidence"] <= 1.0,
        f"got confidence={skills[0].get('confidence') if skills else None!r}",
    )


def test_multi_item_record(extractor) -> None:
    print("\n[TEST] multi-skill / multi-knowledge record")

    skills, knowledge = extractor.extract(
        "Experience designing REST APIs and implementing CI/CD pipelines is a plus.",
        "job_smoke_0001",
    )
    check("returns 2 skills", len(skills) == 2, f"got {len(skills)} skills")
    check("returns 2 knowledge", len(knowledge) == 2, f"got {len(knowledge)} knowledge")
    skill_texts = {s["text"] for s in skills}
    check(
        "includes 'designing REST APIs'",
        "designing REST APIs" in skill_texts,
        f"got {skill_texts}",
    )
    check(
        "includes 'implementing CI/CD pipelines'",
        "implementing CI/CD pipelines" in skill_texts,
        f"got {skill_texts}",
    )


def test_empty_arrays(extractor) -> None:
    print("\n[TEST] record with empty SKILL/KNOWLEDGE arrays")

    skills, knowledge = extractor.extract(
        "We offer competitive compensation and a flexible work environment.",
        "job_smoke_0002",
    )
    check("empty skills returns []", skills == [], f"got {skills!r}")
    check("empty knowledge returns []", knowledge == [], f"got {knowledge!r}")


def test_fallback_by_text(extractor) -> None:
    print("\n[TEST] sentence_text fallback (sentence_id missing)")

    skills, knowledge = extractor.extract(
        "Familiarity with Docker containers is required.",
        sentence_id="",  # forces text fallback
    )
    check(
        "fallback returns 1 knowledge",
        len(knowledge) == 1 and knowledge[0]["text"] == "Docker containers",
        f"got {knowledge!r}",
    )


def test_missing_record(extractor) -> None:
    print("\n[TEST] missing record returns empty lists (not exception)")

    skills, knowledge = extractor.extract(
        "Some sentence that was never extracted.",
        "job_nonexistent_9999",
    )
    check("missing returns empty skills", skills == [], f"got {skills!r}")
    check("missing returns empty knowledge", knowledge == [], f"got {knowledge!r}")


def test_empty_skill_span_filtered(extractor) -> None:
    print("\n[TEST] items with empty skill_span are filtered out")

    skills, knowledge = extractor.extract(
        "Test sentence with a malformed extraction.",
        "job_smoke_0003",
    )
    check(
        "returns only the valid item",
        len(skills) == 1 and skills[0]["text"] == "valid skill",
        f"got {skills!r}",
    )


def test_coverage(extractor) -> None:
    print("\n[TEST] coverage() diagnostic")

    expected_sids = ["job_smoke_0000", "job_smoke_0001", "job_smoke_9999", "job_smoke_0002"]
    rep = extractor.coverage(expected_sids)
    check("coverage expected=4", rep["expected"] == 4, f"got {rep['expected']}")
    check("coverage present=3", rep["present"] == 3, f"got {rep['present']}")
    check("coverage missing=1", rep["missing"] == 1, f"got {rep['missing']}")
    check("coverage ratio=0.75", abs(rep["coverage"] - 0.75) < 1e-6, f"got {rep['coverage']}")


def test_modelmanager_routing() -> None:
    print("\n[TEST] ModelManager.extract_with_skill_llm_offline")
    from pipeline import AdvancedPipelineConfig, ModelManager

    # Point pipeline config at our synthetic file (after the tempdir has it).
    # The temp file path was assigned in main(); fetch via the env-overridden config.
    mm = ModelManager()
    mm.load_skill_llm_offline_if_needed()
    check(
        "skill_llm_offline loader was attached",
        mm.skill_llm_offline is not None,
        f"got {mm.skill_llm_offline!r}",
    )
    if mm.skill_llm_offline is None:
        return

    skills, knowledge = mm.extract_with_skill_llm_offline(
        "Strong Python skills required for backend development.",
        "job_smoke_0000",
    )
    check(
        "modelmanager returns (list, list) tuple",
        isinstance(skills, list) and isinstance(knowledge, list),
        f"got types ({type(skills).__name__}, {type(knowledge).__name__})",
    )
    check(
        "modelmanager item shape matches extract_with_bert",
        skills and "text" in skills[0] and "confidence" in skills[0],
        f"got keys {list(skills[0].keys()) if skills else None}",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 72)
    print("Skill-LLM offline integration — smoke test")
    print("=" * 72)

    # Create the synthetic JSONL in a temp dir, point AdvancedPipelineConfig at it.
    with tempfile.TemporaryDirectory(prefix="skill_llm_smoke_") as tmp:
        tmp = Path(tmp)
        jsonl_path = tmp / "skill_llm_extractions.jsonl"
        write_synthetic_jsonl(jsonl_path)
        print(f"[INFO] wrote synthetic JSONL to {jsonl_path}")

        from pipeline import AdvancedPipelineConfig
        AdvancedPipelineConfig.SKILL_LLM_EXTRACTIONS_PATH = str(jsonl_path)
        AdvancedPipelineConfig.EXTRACTION_MODE = "skill_llm_offline"

        from extractors.skill_llm_offline import SkillLLMOfflineExtractor
        extractor = SkillLLMOfflineExtractor(jsonl_path)

        test_loader_basic(extractor)
        test_lookup_by_sentence_id(extractor)
        test_multi_item_record(extractor)
        test_empty_arrays(extractor)
        test_fallback_by_text(extractor)
        test_missing_record(extractor)
        test_empty_skill_span_filtered(extractor)
        test_coverage(extractor)
        test_modelmanager_routing()

    print()
    print("=" * 72)
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    if _FAIL_NAMES:
        print("Failed:")
        for n in _FAIL_NAMES:
            print(f"  - {n}")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
