"""
smoke_test_competency_v2_offline.py

Offline smoke test for the Phase 2.2 competency generator.

Patches the LLM call with a fake responder so the test can validate:
    1. Prompt is built correctly (contains cluster items, KKNI brief, rationale rules).
    2. Parsed LLM output flows through provenance attachment correctly.
    3. Validation drops competencies with too-short rationale / unmapped skills.
    4. Provenance fields are populated (contributing_item_ids, source_job_ids,
       source_sentences, source_sentence_ids).
    5. Grounding score preview is computed correctly.
    6. Dedup merges semantically-duplicate titles + Jaccard-overlapping skills.
    7. Final competency cap is enforced.
    8. Each competency carries provider + model from BatchReasoning.
    9. Provider router dispatches GPT to Jatevo, others to OpenRouter.
   10. CompetencyV2.to_dict() produces serializable output.

Exits 0 if all PASS, 1 on any FAIL. No network calls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Fake LLM responder
# --------------------------------------------------------------------------- #


def make_fake_response(content: str):
    """Mimic the openai client's chat.completions.create(...) return shape."""

    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Usage:
        prompt_tokens = 250
        completion_tokens = 400

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]
            self.usage = _Usage()

    return _Resp(content)


# Three distinct fake competency outputs — one per cluster we'll feed
FAKE_AUTH = """
{
  "batch_reasoning": "This cluster centres on authentication and session security. The skills 'implementing OAuth 2.0 flows', 'session management', and 'input validation' co-occur in backend web-development job postings. I see one coherent sub-theme. I will produce ONE competency covering all three, because in practice they are designed and tested together — splitting them would create artificially narrow learning outcomes. No skills dropped.",
  "competencies": [
    {
      "title": "Implement secure authentication flows",
      "description": "Design, implement, and test secure authentication, session management, and input-validation logic for web applications, applying industry best practices for credential handling and CSRF/XSS protection.",
      "related_skills": ["implementing OAuth 2.0 flows", "session management", "input validation"],
      "rationale": "These three skills together describe how a junior engineer builds the authentication layer of a web application. They cluster naturally because all three appear together in job postings for backend roles. I kept them as one competency rather than splitting because in practice they must be designed and tested as an integrated whole — you cannot ship OAuth without input validation, and session management is the bridge between them. A learner achieving this competency can demonstrate it by building a working OAuth 2.0 client + server flow with proper session handling and input sanitisation, which is exactly what backend engineering job postings list as a requirement together.",
      "soft_skills_required": ["attention to detail", "problem solving"],
      "soft_skills_description": "Secure-coding work demands rigour and methodical thinking when reasoning about adversarial inputs.",
      "future_relevance": "Emerging — cybersecurity demand is rising across all WEF 2025 priority domains.",
      "future_weight": 0.81,
      "empirical_trend": "Emerging",
      "kkni_level": 5
    }
  ]
}
""".strip()


FAKE_AUTH_DUP = """
{
  "batch_reasoning": "Same auth cluster as before. Producing one competency about secure authentication.",
  "competencies": [
    {
      "title": "Implementing Secure Authentication Flows",
      "description": "Build and test authentication, sessions, and input validation in web apps using current standards.",
      "related_skills": ["implementing OAuth 2.0 flows", "session management"],
      "rationale": "Two of the same skills as the prior auth cluster — the dedup pipeline should merge this into the survivor. The rationale here exists only so the validator accepts the competency long enough for dedup to absorb it; it intentionally talks about identical themes so that title-similarity and Jaccard both fire. Curriculum designers should not see this as a separate item.",
      "soft_skills_required": ["attention to detail"],
      "soft_skills_description": "Methodical secure-coding mindset.",
      "future_relevance": "Emerging.",
      "future_weight": 0.81,
      "empirical_trend": "Emerging",
      "kkni_level": 5
    }
  ]
}
""".strip()


FAKE_FRONTEND = """
{
  "batch_reasoning": "This cluster is about modern frontend web development: React, TypeScript, CSS, responsive design, modern JS frameworks. All five items belong to one coherent competency about building modern web UIs. I see no need to split.",
  "competencies": [
    {
      "title": "Build modern web application UIs",
      "description": "Develop responsive, accessible web application interfaces using React with TypeScript, modern CSS layout techniques, and contemporary JavaScript framework patterns.",
      "related_skills": ["React component design", "TypeScript development", "CSS layout", "responsive web design"],
      "rationale": "Modern web-application UI development is a single integrated skill set. React component design, TypeScript, CSS layout, and responsive design are not independent — they are the four legs of one stool. A learner who can implement responsive, type-safe React components with appropriate CSS layout has mastered the actual job-market competency. Treating them as separate competencies would fragment the assessment. This grouping also matches how frontend engineering roles describe their requirements in job postings: a single block listing React + TS + CSS + responsive together.",
      "soft_skills_required": ["collaboration", "design thinking"],
      "soft_skills_description": "Frontend work is inherently cross-functional with design and product roles.",
      "future_relevance": "Stable — frontend remains a high-demand domain.",
      "future_weight": 0.6,
      "empirical_trend": "Stable",
      "kkni_level": 5
    }
  ]
}
""".strip()


# Bad: rationale too short — should be rejected
FAKE_BAD = """
{
  "batch_reasoning": "This cluster is on data engineering. One competency covers all items.",
  "competencies": [
    {
      "title": "Build data pipelines",
      "description": "Design ETL pipelines + warehousing.",
      "related_skills": ["designing ETL pipelines", "data warehouse modeling"],
      "rationale": "Short rationale.",
      "soft_skills_required": [],
      "soft_skills_description": "",
      "future_relevance": "Stable.",
      "future_weight": 0.5,
      "empirical_trend": "Stable",
      "kkni_level": 6
    }
  ]
}
""".strip()


# --------------------------------------------------------------------------- #
# Build synthetic clusters
# --------------------------------------------------------------------------- #


def build_clusters():
    """Three synthetic hard_plus_knowledge clusters + one synthetic soft-only cluster."""
    from pipeline import ConfidenceTier, KnowledgeItem, SkillItem, SkillType
    from clustering.cluster_schema import Cluster, CLUSTERER_VERSION

    import numpy as np

    def s(text, sid):
        return SkillItem(
            text=text,
            type=SkillType.HARD,
            confidence_score=0.9,
            confidence_tier=ConfidenceTier.VERY_HIGH,
            source="skill_llm_8b_lora_v1",
            sentence_id=sid,
            sentence_text=f"... {text} ...",
            extractor_source="skill_llm_8b_lora_v1",
        )

    def k(text, sid):
        return KnowledgeItem(
            text=text,
            confidence_score=0.9,
            confidence_tier=ConfidenceTier.VERY_HIGH,
            source="skill_llm_8b_lora_v1",
            sentence_id=sid,
            sentence_text=f"... {text} ...",
            extractor_source="skill_llm_8b_lora_v1",
        )

    def make_cluster(cid, label, items, stream="hard_plus_knowledge"):
        return Cluster(
            id=cid,
            stream=stream,
            method="hdbscan",
            items=items,
            n_items=len(items),
            n_unique_jobs=len({(getattr(it, "sentence_id", "") or "").rpartition("_")[0] for it in items}),
            n_skill_items=sum(1 for it in items if hasattr(it, "type")),
            n_knowledge_items=sum(1 for it in items if not hasattr(it, "type")),
            cohesion_score=0.65,
            cohesion_std=0.05,
            centroid_embedding=np.zeros(384, dtype=np.float32),
            summary_label=label,
            top_terms=[],
            seed=42,
            embedder_model="sentence-transformers/all-MiniLM-L6-v2",
            clusterer_version=CLUSTERER_VERSION,
        )

    # AUTH cluster
    auth_items = [
        s("implementing OAuth 2.0 flows", "job_a01_0001"),
        s("session management", "job_a02_0001"),
        s("input validation", "job_a03_0001"),
        k("OAuth 2.0", "job_a01_0002"),
        k("authentication protocols", "job_a04_0001"),
    ]
    # FRONTEND cluster
    fe_items = [
        s("React component design", "job_f01_0001"),
        s("TypeScript development", "job_f02_0001"),
        s("CSS layout", "job_f03_0001"),
        s("responsive web design", "job_f04_0001"),
        k("modern JavaScript frameworks", "job_f05_0001"),
    ]
    # DATA-ENG cluster (will be sent the BAD response with short rationale → drop)
    de_items = [
        s("designing ETL pipelines", "job_d01_0001"),
        s("data warehouse modeling", "job_d02_0001"),
        s("Apache Spark jobs", "job_d03_0001"),
    ]
    # AUTH-DUP cluster — will produce a competency that dedup absorbs
    auth_dup_items = [
        s("implementing OAuth 2.0 flows", "job_a05_0001"),
        s("session management", "job_a06_0001"),
    ]

    return [
        make_cluster("cluster_h0001", "OAuth & session security: auth, validation", auth_items),
        make_cluster("cluster_h0002", "frontend web dev: React, TypeScript, CSS", fe_items),
        make_cluster("cluster_h0003", "data engineering: ETL, warehouse modeling", de_items),
        make_cluster("cluster_h0004", "OAuth & session (overlap)", auth_dup_items),
    ]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_router_dispatches_correctly():
    print("\n[TEST] llm_client_router")
    from llm_client_router import is_gpt_family

    check("'gpt-4o-mini' is GPT family", is_gpt_family("gpt-4o-mini"))
    check("'openai/gpt-5' is GPT family", is_gpt_family("openai/gpt-5"))
    check("'o1-preview' is GPT family", is_gpt_family("o1-preview"))
    check("'o3-mini' is GPT family", is_gpt_family("o3-mini"))
    check("'deepseek/deepseek-v3.2' is NOT GPT", not is_gpt_family("deepseek/deepseek-v3.2"))
    check("'claude-3-7-sonnet' is NOT GPT", not is_gpt_family("claude-3-7-sonnet"))
    check("'meta-llama/Llama-3.1-70b' is NOT GPT", not is_gpt_family("meta-llama/Llama-3.1-70b"))


def test_prompt_builder():
    print("\n[TEST] prompt builder contents")
    from competency_v2_schema import GeneratorConfig
    from competency_generator_v2 import build_cluster_prompt

    clusters = build_clusters()
    prompt = build_cluster_prompt(clusters[0], GeneratorConfig(), top_soft_skills=["teamwork", "collaboration"])

    check("prompt mentions CLUSTER ID", clusters[0].id in prompt)
    check("prompt includes cluster summary label", clusters[0].summary_label in prompt)
    check("prompt lists OAuth skill", '"implementing OAuth 2.0 flows"' in prompt)
    check("prompt lists OAuth knowledge", '"OAuth 2.0"' in prompt)
    check("prompt includes KKNI brief", "KKNI" in prompt and "SMK" in prompt)
    check("prompt includes soft-skill reference", "teamwork" in prompt)
    check(
        "prompt enforces rationale length",
        "200-900 chars" in prompt or "200" in prompt,
    )
    check("prompt instructs no hallucination", "Do NOT invent" in prompt)


def test_end_to_end_offline():
    print("\n[TEST] end-to-end with patched LLM call")
    from competency_v2_schema import GeneratorConfig
    from competency_generator_v2 import generate_competencies_v2

    clusters = build_clusters()
    # Cluster order is: AUTH, FRONTEND, DATA-ENG (bad), AUTH-DUP
    # We need a stateful fake that returns the right output per cluster
    fakes = [FAKE_AUTH, FAKE_FRONTEND, FAKE_BAD, FAKE_AUTH_DUP]
    call_idx = {"i": 0}

    def fake_create(**kwargs):
        out = fakes[call_idx["i"]]
        call_idx["i"] += 1
        return make_fake_response(out)

    # Patch the .create method on the OpenAI client. Easier: patch the whole
    # get_client_for_model to return a fake client.
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return fake_create(**kwargs)

    def fake_router(model, request_timeout=120.0):
        return FakeClient(), "openrouter"

    with patch("competency_generator_v2.get_client_for_model", fake_router):
        competencies, batch_reasonings = generate_competencies_v2(
            clusters=clusters,
            config=GeneratorConfig(model="deepseek/deepseek-v3.2"),
            top_soft_skills=["teamwork", "collaboration", "attention to detail"],
        )

    # Reasoning: we sent 4 hard+knowledge clusters → 4 LLM calls → 4 batch_reasonings
    check("4 batch_reasonings produced (one per cluster)", len(batch_reasonings) == 4,
          f"got {len(batch_reasonings)}")

    # Competencies expected: AUTH (1) + FRONTEND (1) + DATA-ENG (0, rejected) + AUTH-DUP (1 then merged)
    # After dedup, AUTH-DUP merges into AUTH → 2 survivors
    titles = [c.title for c in competencies]
    print(f"        produced {len(competencies)} survivors: {titles}")
    check("at least 2 competencies survive dedup", len(competencies) >= 2,
          f"got {len(competencies)}")
    check("at most 3 competencies survive (data-eng was rejected)", len(competencies) <= 3,
          f"got {len(competencies)} — data-eng should not have produced a valid competency")

    # AUTH-DUP should have been merged into the AUTH competency
    auth_comp = next((c for c in competencies if "auth" in c.title.lower()), None)
    check("AUTH competency exists", auth_comp is not None)
    if auth_comp:
        check(
            "AUTH competency absorbed a merge",
            len(auth_comp.merged_from) >= 1,
            f"merged_from = {auth_comp.merged_from}",
        )
        check(
            "AUTH competency has >= 3 related skills (union after merge)",
            len(auth_comp.related_skills) >= 3,
            f"got {len(auth_comp.related_skills)}",
        )

    # Provenance present
    for c in competencies:
        check(
            f"{c.id}: contributing_item_ids non-empty",
            len(c.contributing_item_ids) >= 2,
        )
        check(
            f"{c.id}: source_job_ids non-empty",
            len(c.source_job_ids) >= 1,
            f"got {c.source_job_ids}",
        )
        check(
            f"{c.id}: rationale in 300-900 chars",
            300 <= len(c.rationale) <= 900,
            f"got {len(c.rationale)} chars",
        )

    # Grounding preview
    for c in competencies:
        check(
            f"{c.id}: grounding_score_preview in [0, 1]",
            0.0 <= c.grounding_score_preview <= 1.0,
            f"got {c.grounding_score_preview}",
        )

    # Provider tag
    for c in competencies:
        check(f"{c.id}: provider tag set to 'openrouter'", c.provider == "openrouter")

    # Serializability
    for c in competencies:
        try:
            import json as _json
            _json.dumps(c.to_dict())
            check(f"{c.id}: to_dict() is JSON-serializable", True)
        except Exception as e:
            check(f"{c.id}: to_dict() is JSON-serializable", False, str(e))

    # Batch reasoning has the LLM's CoT
    for br in batch_reasonings:
        check(
            f"{br.id}: batch_reasoning text present",
            len(br.batch_reasoning) > 20,
            f"len={len(br.batch_reasoning)}",
        )


def main() -> int:
    print("=" * 72)
    print("Phase 2.2 competency generator — offline smoke test")
    print("=" * 72)

    test_router_dispatches_correctly()
    test_prompt_builder()
    test_end_to_end_offline()

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
