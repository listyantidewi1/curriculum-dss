"""
smoke_test_competency_evaluator.py

Offline smoke test for the Phase 2.5 competency evaluator.

Builds three synthetic CompetencyV2 objects exercising the grounding gate:

    1. PERFECT — every related_skill appears verbatim in a source_sentence
                 → grounding_score == 1.0, passes.
    2. PARAPHRASE — skills don't appear verbatim, but are semantically very
                    close to source_sentences (SBERT cosine >= 0.65)
                    → most verified by SBERT; passes if mostly aligned.
    3. HALLUCINATED — half the related_skills don't appear in any
                      source_sentence (and aren't paraphrases either)
                      → grounding_score < 0.80, fails the gate.

Then runs both modes:
    a. Substring-only (SBERT disabled) — verifies the substring path.
    b. SBERT-enabled — verifies the paraphrase recovery path.

Exits 0 on all PASS, 1 on any FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


_PASS = 0
_FAIL = 0
_FAIL_NAMES: list = []


def check(name, condition, detail=""):
    global _PASS, _FAIL
    if condition:
        print(f"  [PASS] {name}")
        _PASS += 1
    else:
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")
        _FAIL += 1
        _FAIL_NAMES.append(name)


def make_perfect():
    from competency_v2_schema import CompetencyV2
    return CompetencyV2(
        id="comp_perfect_001",
        title="Implement secure authentication flows",
        description="Build OAuth, sessions, and input validation.",
        related_skills=[
            "OAuth 2.0",
            "session management",
            "input validation",
        ],
        rationale="x" * 400,
        batch_reasoning_id="br_test_001",
        contributing_item_ids=["item_a", "item_b"],
        source_job_ids=["job_1"],
        source_sentence_ids=["s1", "s2", "s3"],
        source_sentences=[
            "Strong understanding of OAuth 2.0 is required for this role.",
            "Experience with session management in distributed web apps.",
            "Implement input validation per OWASP guidelines.",
        ],
    )


def make_paraphrase():
    from competency_v2_schema import CompetencyV2
    return CompetencyV2(
        id="comp_paraphrase_002",
        title="Build modern web application UIs",
        description="React + TypeScript + responsive design.",
        related_skills=[
            "React component design",
            "type-safe JavaScript development",
            "modern CSS layout techniques",
        ],
        rationale="x" * 400,
        batch_reasoning_id="br_test_002",
        contributing_item_ids=["item_c", "item_d"],
        source_job_ids=["job_2"],
        source_sentence_ids=["s4", "s5", "s6"],
        source_sentences=[
            "Strong background in React and component-based UI architecture.",
            "TypeScript experience required for type-safe frontend code.",
            "Build responsive layouts with modern CSS grid and flexbox.",
        ],
    )


def make_hallucinated():
    from competency_v2_schema import CompetencyV2
    return CompetencyV2(
        id="comp_hallucinated_003",
        title="Deploy cloud infrastructure",
        description="AWS, Terraform, Kubernetes.",
        related_skills=[
            "AWS",                                      # IN source_sentences
            "quantum cryptography research",            # NOT in source — hallucinated
            "telepathic communication protocol",        # NOT in source — hallucinated
            "Terraform infrastructure-as-code",         # paraphrase of source
        ],
        rationale="x" * 400,
        batch_reasoning_id="br_test_003",
        contributing_item_ids=["item_e", "item_f"],
        source_job_ids=["job_3"],
        source_sentence_ids=["s7", "s8"],
        source_sentences=[
            "Deploy applications to AWS using best practices for security.",
            "Use Terraform to define infrastructure as code for our environments.",
        ],
    )


def test_substring_mode():
    print("\n[TEST] substring-only mode (SBERT disabled)")
    from competency_evaluator import EvaluatorConfig, evaluate_competencies

    config = EvaluatorConfig(disable_sbert=True)

    comps = [make_perfect(), make_paraphrase(), make_hallucinated()]
    passing, failing, report = evaluate_competencies(comps, config=config)

    # PERFECT: every skill substring-matches → score 1.0 → passes
    perfect = next(c for c in comps if c.id == "comp_perfect_001")
    check("PERFECT score == 1.0", perfect.grounding_score == 1.0,
          f"got {perfect.grounding_score}")
    check("PERFECT passes gate", perfect.grounding_passed is True)
    check("PERFECT method == substring", perfect.grounding_method == "substring",
          f"got {perfect.grounding_method!r}")

    # PARAPHRASE: 0 of 3 skills substring-match (since they're paraphrases)
    # → score 0.0 → fails. This is correct behavior for substring-only mode.
    para = next(c for c in comps if c.id == "comp_paraphrase_002")
    check("PARAPHRASE score < 0.80 in substring-only mode",
          para.grounding_score < 0.80,
          f"got {para.grounding_score}")
    check("PARAPHRASE fails gate (substring-only)", para.grounding_passed is False)

    # HALLUCINATED: 1 of 4 skills (AWS) substring-matches; 3 fail → 0.25 → fails
    hall = next(c for c in comps if c.id == "comp_hallucinated_003")
    check("HALLUCINATED score < 0.80",
          hall.grounding_score < 0.80,
          f"got {hall.grounding_score}")
    check("HALLUCINATED fails gate", hall.grounding_passed is False)
    check("HALLUCINATED method indicates failures",
          "failures" in hall.grounding_method or hall.grounding_method == "substring",
          f"got {hall.grounding_method!r}")

    check("report.n_evaluated == 3", report.n_evaluated == 3)
    check("report.n_passed >= 1 (PERFECT)", report.n_passed >= 1)
    check("report has runtime", report.runtime_seconds >= 0)


def test_sbert_mode():
    print("\n[TEST] SBERT-enabled mode")
    from competency_evaluator import EvaluatorConfig, evaluate_competencies

    config = EvaluatorConfig(disable_sbert=False, sbert_threshold=0.55)
    comps = [make_perfect(), make_paraphrase(), make_hallucinated()]
    passing, failing, report = evaluate_competencies(comps, config=config)

    # PERFECT still passes (substring is cheaper than SBERT)
    perfect = next(c for c in comps if c.id == "comp_perfect_001")
    check("PERFECT passes (SBERT mode)", perfect.grounding_passed is True)

    # PARAPHRASE should now pass via SBERT
    para = next(c for c in comps if c.id == "comp_paraphrase_002")
    print(f"        PARAPHRASE score: {para.grounding_score:.3f}, method: {para.grounding_method}")
    check("PARAPHRASE score >= 0.80 (SBERT recovery)",
          para.grounding_score >= 0.80,
          f"got {para.grounding_score:.3f} — SBERT failed to recover paraphrases")
    check("PARAPHRASE passes gate (SBERT mode)",
          para.grounding_passed is True,
          f"score={para.grounding_score}")
    check("PARAPHRASE method indicates SBERT used",
          "sbert" in para.grounding_method.lower(),
          f"got {para.grounding_method!r}")

    # HALLUCINATED: 1 substring (AWS) + 1 paraphrase (Terraform iac) + 2 unverifiable
    # = 2/4 = 0.5 → still fails
    hall = next(c for c in comps if c.id == "comp_hallucinated_003")
    print(f"        HALLUCINATED score: {hall.grounding_score:.3f}, method: {hall.grounding_method}")
    check("HALLUCINATED fails gate even with SBERT",
          hall.grounding_passed is False,
          f"got score={hall.grounding_score}")
    check("HALLUCINATED has unverified skills logged",
          "no substring" in hall.grounding_reasoning.lower() or "unverified" in hall.grounding_reasoning.lower(),
          "expected 'no substring' or 'unverified' in reasoning")


def test_empty_skills():
    print("\n[TEST] empty related_skills edge case")
    from competency_v2_schema import CompetencyV2
    from competency_evaluator import EvaluatorConfig, evaluate_competencies

    comp = CompetencyV2(
        id="comp_empty",
        title="x",
        description="y",
        related_skills=[],
        rationale="x" * 400,
        batch_reasoning_id="br",
        contributing_item_ids=["item_a", "item_b"],
        source_job_ids=["j"],
        source_sentences=["s"],
    )
    passing, failing, report = evaluate_competencies([comp], config=EvaluatorConfig(disable_sbert=True))
    check("empty-skills competency has score 0.0", comp.grounding_score == 0.0)
    check("empty-skills competency fails gate", comp.grounding_passed is False)
    check("empty-skills method tag", comp.grounding_method == "no_skills")


def test_schema_fields():
    print("\n[TEST] schema fields populated by evaluator")
    from competency_evaluator import EvaluatorConfig, evaluate_competencies

    comp = make_perfect()
    evaluate_competencies([comp], config=EvaluatorConfig(disable_sbert=True))
    check("grounding_score float", isinstance(comp.grounding_score, float))
    check("grounding_passed bool", isinstance(comp.grounding_passed, bool))
    check("grounding_method non-empty", bool(comp.grounding_method))
    check("grounding_reasoning has audit content",
          len(comp.grounding_reasoning) > 30,
          f"len={len(comp.grounding_reasoning)}")
    check("evaluator_version set",
          comp.evaluator_version.startswith("v2.5"),
          f"got {comp.evaluator_version!r}")

    # Serialization roundtrip
    d = comp.to_dict()
    for k in ("grounding_score", "grounding_passed", "grounding_method", "grounding_reasoning", "evaluator_version"):
        check(f"to_dict() includes {k}", k in d)


def main() -> int:
    print("=" * 72)
    print("Phase 2.5 competency evaluator — offline smoke test")
    print("=" * 72)

    test_substring_mode()
    test_sbert_mode()
    test_empty_skills()
    test_schema_fields()

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
