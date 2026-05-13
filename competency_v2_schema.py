"""
competency_v2_schema.py

Dataclasses for the Phase 2.2 cluster-driven competency generator.

A `CompetencyV2` is one curriculum-ready learning outcome with mandatory
provenance back to the cluster items and source sentences that contributed
to it. A `BatchReasoning` is the LLM's chain-of-thought captured during the
LLM call that produced one or more competencies for a single cluster.

See docs/PHASE_2_2_DESIGN.md for the full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


GENERATOR_VERSION = "v2.0.0"


@dataclass
class BatchReasoning:
    """The LLM's top-level thinking for one cluster.

    One record per LLM call. Phase 2.2 produces 1-4 competencies per cluster;
    all of those competencies share the same `BatchReasoning` (linked via
    `CompetencyV2.batch_reasoning_id`).

    Public-surface visibility: rendered on the competency detail page behind
    a "How the LLM grouped these skills (batch reasoning) ▾" toggle.
    """

    id: str                         # FK target for CompetencyV2.batch_reasoning_id
    cluster_id: str                 # which Phase 2.1 cluster this reasoning is for
    timestamp: str                  # ISO-8601 UTC
    provider: str                   # "openrouter" | "jatevo"
    model: str                      # e.g. "deepseek/deepseek-v3.2" or "openai/gpt-5"
    prompt_sha256: str              # hash of the rendered prompt for audit
    batch_reasoning: str            # the LLM's chain-of-thought for this cluster
    n_skills_in: int                # how many items were in the cluster input
    n_competencies_out: int         # how many competencies the LLM proposed (pre-validation)
    raw_response: str = ""          # full raw LLM text response, for debug
    latency_seconds: float = 0.0    # wall-clock for the LLM call
    prompt_tokens: int = 0          # if the provider reports usage
    completion_tokens: int = 0
    # v9 sprint: optional Bahasa Indonesia translation of `batch_reasoning`,
    # populated by the runner's EN→ID translation pass when
    # --translate-output-to-id is on. Empty otherwise.
    batch_reasoning_id: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "cluster_id": self.cluster_id,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "batch_reasoning": self.batch_reasoning,
            "n_skills_in": self.n_skills_in,
            "n_competencies_out": self.n_competencies_out,
            "raw_response": self.raw_response,
            "latency_seconds": round(float(self.latency_seconds), 3),
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "batch_reasoning_id": self.batch_reasoning_id,
        }


@dataclass
class CompetencyV2:
    """One curriculum-ready competency with full provenance + reasoning."""

    # Identity
    id: str                                      # e.g. "comp_a1b2c3"
    title: str                                   # verb-led learning outcome
    description: str                             # 1-3 sentences, assessable

    # Content
    related_skills: List[str]                    # hard skills (drawn from cluster items)
    soft_skills_required: List[str] = field(default_factory=list)
    soft_skills_description: str = ""            # 1 sentence

    # Reasoning (NEW for v2 — explainability)
    rationale: str = ""                          # 400-800 chars, public-facing
    batch_reasoning_id: str = ""                 # FK to BatchReasoning

    # Provenance (Req 6)
    contributing_item_ids: List[str] = field(default_factory=list)
    source_job_ids: List[str] = field(default_factory=list)
    source_sentence_ids: List[str] = field(default_factory=list)
    source_sentences: List[str] = field(default_factory=list)

    # Future-awareness
    future_relevance: str = ""                   # 1 sentence; cites future_weight / trend source
    future_weight: float = 0.0                   # max future_weight across related_skills
    empirical_trend: str = "Stable"              # "Emerging" / "Declining" / "Stable" / "Mixed"

    # KKNI (informational; Phase 2.3 labeller may overwrite)
    kkni_level: Optional[int] = None             # 1-9
    kkni_level_source: str = "llm_suggested"     # "llm_suggested" | "sbert_labeller"
    kkni_match_similarity: Optional[float] = None

    # Education-level demand distribution (Phase 2.4)
    # Maps Indonesian education stage → fraction of source jobs requiring it.
    # e.g. {"SMK": 0.60, "D3": 0.25, "S1": 0.15}
    education_levels_demanded: dict = field(default_factory=dict)
    # KKNI-level distribution (1-9 → fraction)
    education_kkni_demanded: dict = field(default_factory=dict)
    # Audit: how many source jobs had any education info
    n_jobs_with_education: int = 0

    # Grounding preview — computed by Phase 2.2 generator (cheap text-key match)
    grounding_score_preview: float = 0.0         # |verified related_skills| / |total related_skills|
    grounding_flagged: bool = False              # True if preview < 0.80

    # Canonical grounding — computed by Phase 2.5 evaluator. After the 2026-05-12
    # honesty audit we split this into TWO independent measurements:
    #   skill_grounding_score    — "do the listed related_skills actually appear
    #                              in source_sentences?" (semantic SBERT match)
    #   sentence_relevance_score — "do the source_sentences actually describe
    #                              what this competency claims to teach?"
    # The canonical `grounding_score` is the MIN of the two: both must be high.
    # This catches the Azure-DevOps-like case where a narrow competency title
    # has broadly-matching but topically-irrelevant source sentences.
    grounding_score: float = 0.0                 # canonical = min(skill, sentence). The Phase 2.5 gate uses this.
    skill_grounding_score: float = 0.0           # fraction of related_skills verified against source_sentences
    sentence_relevance_score: float = 0.0        # fraction of source_sentences semantically supporting the competency
    grounding_passed: Optional[bool] = None      # True if grounding_score >= evaluator threshold; None if not yet evaluated
    grounding_method: str = ""                   # "sbert" — semantic only after the honesty audit (substring deprecated)
    grounding_reasoning: str = ""                # human-readable evaluator audit string
    evaluator_version: str = ""                  # populated when Phase 2.5 runs

    # Generator metadata
    generated_at: str = ""                       # ISO-8601 UTC
    generator_version: str = GENERATOR_VERSION
    cluster_id: str = ""                         # source cluster id (Phase 2.1)
    cluster_summary_label: str = ""              # human-readable cluster label
    provider: str = ""                           # "openrouter" | "jatevo"
    model: str = ""                              # e.g. "deepseek/deepseek-v3.2"

    # Dedup audit (when a competency is the survivor of a merge)
    merged_from: List[str] = field(default_factory=list)

    # Tier 2D (2026-05-13) — title evidence audit. Each entry is
    # {"term": "Azure DevOps", "literal_hits": 1, "n_sentences": 4, "min_required": 2,
    #  "severity": "warning"|"info"}. Surfaces to the dashboard as a "title
    # overreaches evidence" badge so reviewers can spot brand/proper-noun
    # tokens in titles that the source sentences don't actually back.
    title_evidence_concerns: List[dict] = field(default_factory=list)

    # v8 sprint Phase 4 — demand + priority signals. Computed as a post-process
    # after all competencies are generated and Phase 2.4 (education aggregation)
    # has run.
    #   demand_score   = n_unique_jobs / max_n_unique_jobs_across_competencies
    #                    in this run; normalised [0, 1]
    #   priority_score = 0.40 × demand_score + 0.30 × grounding_score
    #                    + 0.30 × future_weight  (per Req 9.6)
    demand_score: float = 0.0
    priority_score: float = 0.0

    # v9 sprint Phase 2.4b — SKKNI occupation mapping (skema okupasi
    # Indonesia). Top-K matches above cosine threshold against the enriched
    # SKKNI catalogue. Each entry:
    #   {"occupation_en": "Junior Programmer",
    #    "occupation_id": "Skema Okupasi Pemrogram Junior",
    #    "sector": "Teknologi Informasi dan Komunikasi",
    #    "cosine": 0.78,
    #    "description_en": "..."}
    # Empty list means no SKKNI match passed the threshold — useful flag for
    # competencies that don't fit any Indonesian occupational standard.
    occupation_matches: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "related_skills": list(self.related_skills),
            "soft_skills_required": list(self.soft_skills_required),
            "soft_skills_description": self.soft_skills_description,
            "rationale": self.rationale,
            "batch_reasoning_id": self.batch_reasoning_id,
            "contributing_item_ids": list(self.contributing_item_ids),
            "source_job_ids": list(self.source_job_ids),
            "source_sentence_ids": list(self.source_sentence_ids),
            "source_sentences": list(self.source_sentences),
            "future_relevance": self.future_relevance,
            "future_weight": round(float(self.future_weight), 4),
            "empirical_trend": self.empirical_trend,
            "kkni_level": self.kkni_level,
            "kkni_level_source": self.kkni_level_source,
            "kkni_match_similarity": (
                round(float(self.kkni_match_similarity), 4)
                if self.kkni_match_similarity is not None else None
            ),
            "education_levels_demanded": dict(self.education_levels_demanded),
            "education_kkni_demanded": dict(self.education_kkni_demanded),
            "n_jobs_with_education": int(self.n_jobs_with_education),
            "grounding_score_preview": round(float(self.grounding_score_preview), 4),
            "grounding_flagged": bool(self.grounding_flagged),
            "grounding_score": round(float(self.grounding_score), 4),
            "skill_grounding_score": round(float(self.skill_grounding_score), 4),
            "sentence_relevance_score": round(float(self.sentence_relevance_score), 4),
            "grounding_passed": self.grounding_passed,
            "grounding_method": self.grounding_method,
            "grounding_reasoning": self.grounding_reasoning,
            "evaluator_version": self.evaluator_version,
            "generated_at": self.generated_at,
            "generator_version": self.generator_version,
            "cluster_id": self.cluster_id,
            "cluster_summary_label": self.cluster_summary_label,
            "provider": self.provider,
            "model": self.model,
            "merged_from": list(self.merged_from),
            "title_evidence_concerns": list(self.title_evidence_concerns),
            "demand_score": round(float(self.demand_score), 4),
            "priority_score": round(float(self.priority_score), 4),
            "occupation_matches": list(self.occupation_matches),
        }

    # ---------------------------------------------------------------------
    # Provenance invariants (mirror docs/PHASE_2_2_DESIGN.md "Provenance
    # preservation" table). Used by the generator + by tests to short-circuit
    # on malformed output.
    # ---------------------------------------------------------------------

    def validate_provenance(
        self,
        min_rationale: int = 200,
        max_rationale: int = 900,
    ) -> List[str]:
        """Return a list of human-readable violations, empty if all pass.

        Bounds are passed in from GeneratorConfig so they're tunable without
        editing this method.
        """
        issues = []
        if len(self.related_skills) < 2:
            issues.append("related_skills must have >= 2 items")
        if len(self.contributing_item_ids) < 2:
            issues.append("contributing_item_ids must have >= 2 items")
        if len(self.source_job_ids) < 1:
            issues.append("source_job_ids must be non-empty")
        if len(self.source_sentences) < 1:
            issues.append("source_sentences must be non-empty")
        if not (min_rationale <= len(self.rationale) <= max_rationale):
            issues.append(
                f"rationale must be {min_rationale}-{max_rationale} chars (got {len(self.rationale)})"
            )
        if self.rationale.startswith("This competency"):
            issues.append("rationale starts with boilerplate prefix 'This competency'")
        if not self.batch_reasoning_id:
            issues.append("batch_reasoning_id must be set")
        if not self.title.strip():
            issues.append("title must be non-empty")
        if not self.description.strip():
            issues.append("description must be non-empty")
        return issues


@dataclass
class GeneratorConfig:
    """Knobs for the cluster-driven competency generator."""

    # Provider routing — durable rule per memory/project_llm_provider_routing.md:
    #   - GPT models → Jatevo (api_keys/jatevo.txt)
    #   - Everything else → OpenRouter (api_keys/OpenRouter.txt)
    #
    # Default chosen 2026-05-12 from a 6-model A/B (see
    # results/competency_v2_comparison/comparison.md). gpt-5.4-mini matched
    # the most expensive models (gpt-5.5, claude-sonnet-4.5) on every quality
    # metric while being 4× faster and ~30× cheaper.
    #   Fallbacks (only if Jatevo is unavailable):
    #     - "deepseek/deepseek-chat" (cheap, OpenRouter, no bug)
    #     - "anthropic/claude-sonnet-4.5" (premium quality, OpenRouter)
    #   AVOID: "deepseek/deepseek-v3.2" — has a token-corruption bug that
    #          replaces " and " with "religious" throughout output.
    model: str = "gpt-5.4-mini"

    # Generation
    temperature: float = 0.3                # higher than legacy 0.2 — small creativity for rationale
    max_tokens: int = 2000                  # legacy 1500; we add rationale + batch_reasoning fields
    max_competencies_per_cluster: int = 4   # hard cap per LLM call; legacy was 12 per batch

    # Retry / robustness
    max_retries: int = 2
    retry_backoff_base: float = 2.0

    # Validation — relaxed 2026-05-12 after first live run on real clusters
    # showed 73% drop rate at 300-char floor (LLMs often produce useful 200-300
    # char rationales; rejecting them throws away good signal).
    min_related_skills_per_competency: int = 2
    min_rationale_chars: int = 200
    max_rationale_chars: int = 900

    # Dedup tuning history:
    #   2026-05-12  Raised Jaccard 0.40 -> 0.55 after over-merge on results.old2.
    #   2026-05-13  Raised Jaccard 0.55 -> 0.65, but n1k still saw 143 -> 12
    #               (91% merge). Diagnosed: Stage 3 (SBERT title) at 0.85
    #               was the dominant merger, treating semantically-similar
    #               but pedagogically-distinct titles as duplicates
    #               (e.g., "Apply Kubernetes" ≈ "Use Docker"). Raised
    #               SBERT title threshold 0.85 -> 0.92 so only near-exact
    #               title duplicates merge. Pedagogically-distinct topics
    #               survive.
    dedup_semantic_title_threshold: float = 0.92  # SBERT cosine for title-merge
    dedup_jaccard_skills_threshold: float = 0.65  # skill-overlap for merge

    # Hard cap per pipeline run (after dedup across clusters).
    # Raised 2026-05-13 from 12 to 50 after diagnosing that the n1k corpus
    # (67 distinct cluster topics) was always truncating to 12 regardless of
    # dedup threshold changes — losing 80% of curriculum signal. 50 still
    # protects against runaway counts on noisy corpora but lets a real
    # 50-topic curriculum surface.
    final_competency_cap: int = 50

    # Reproducibility / audit
    seed: int = 42
    request_timeout_seconds: int = 120
