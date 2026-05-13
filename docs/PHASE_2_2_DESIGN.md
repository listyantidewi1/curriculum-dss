# Phase 2.2 — Cluster-driven competency generator rewrite (design)

**Status:** design draft, 2026-05-12. Pre-implementation. Source code to land in `competency_generator_v2.py` (alongside legacy `generate_competencies.py` during migration). Replaces legacy domain-batched generator per `.kiro/specs/pipeline-redesign-v2/requirements.md` Req 5–7.

**Sprint slot:** Week 2 of 4-week sprint (2026-05-18 → 2026-05-25), ~4 days estimated.

---

## Context — why we rewrite

The legacy `generate_competencies.py` (1,456 lines, `build_prompt` + `call_llm_for_competencies` + dedup pipeline) was designed for the pre-v2 architecture:

- **Batching** is **domain-based**: skills are grouped by `best_future_domain` from `future_weight_mapping.py`, then chunked. "Uncertain" / "Unmapped" sub-cluster via SBERT agglomerative inside the batch. Coherent but ad-hoc — clusters are formed *inside* the generator, not as a first-class pipeline stage.
- **Output schema** lacks **provenance fields** (`contributing_item_ids`, `source_job_ids`, `source_sentences`). Each competency has `related_skills` (a list of skill strings) but no link back to the specific sentences / extracted items those skills came from. Req 6 mandates provenance throughout.
- **No explicit reasoning trace.** The LLM produces a competency JSON; the *why* is opaque. Users see "title + description" but can't audit the LLM's grouping logic. Misaligned with the v2 explainability mission ("no black boxes between '10,000 job postings' and 'recommend this competency'").
- **No grounding-readiness.** The Phase 2.5 evaluator computes `grounding_score = |verified_related_skills| / |total_related_skills|` and gates at ≥ 0.80. The legacy output schema doesn't preserve the linkage needed to verify each `related_skills` entry against the contributing items.

Phase 2.2 fixes all of this. The user (2026-05-12) also requested a fifth axis:

- **LLM reasoning logged per competency.** When a curriculum designer opens a competency's detail page, they should see not just provenance ("here are the source sentences") but also **why this LLM thought this grouping made sense** — a curated rationale string the LLM produced *while* it was generating the competency. This is now a first-class output field, not a debug afterthought.

---

## Target architecture

```
Phase 2.1 output:                       ┌─────────────────────────────────────┐
List[Cluster] where                     │ Cluster:                            │
  Cluster.items = [SkillItem,           │   id, items, cohesion_score,        │
                   KnowledgeItem]       │   centroid_embedding,                │
  each item carries provenance:         │   summary_label (heuristic, str)     │
   (sentence_id, sentence_text,         └─────────────────────────────────────┘
    extractor_source)                                  │
                                                       ▼
                                ┌────────────────────────────────────────┐
                                │  competency_generator_v2.py            │
                                │                                        │
                                │  for cluster in clusters:              │
                                │    1. Build prompt with:               │
                                │       - cluster's items + their        │
                                │         provenance                     │
                                │       - future_weight annotations      │
                                │       - few-shot examples (curated)    │
                                │       - KKNI reference                 │
                                │       - request reasoning + rationale  │
                                │    2. Call LLM (DeepSeek-V3 / OpenRouter)│
                                │    3. Parse output:                    │
                                │       - batch_reasoning (CoT)          │
                                │       - competencies[]:                │
                                │         · id, title, description       │
                                │         · related_skills               │
                                │         · rationale (NEW)              │
                                │         · contributing_item_ids (NEW)  │
                                │         · source_sentences (NEW)       │
                                │         · source_job_ids (NEW)         │
                                │         · soft_skills_required         │
                                │         · soft_skills_description      │
                                │         · kkni_level (LLM-suggested;   │
                                │           SBERT labeller may overwrite)│
                                │         · future_relevance             │
                                │    4. Validate: every related_skill in │
                                │       cluster.items (drop hallucinated)│
                                │    5. Compute & attach grounding_score │
                                │       (preview; Phase 2.5 owns the gate)│
                                └────────────────────────────────────────┘
                                                       │
                                                       ▼
                                ┌────────────────────────────────────────┐
                                │  Aggregator: merge across clusters     │
                                │  - semantic-title dedup (SBERT >= 0.85)│
                                │  - jaccard-skill merge (>=0.40)        │
                                │  - absorb provenance, reasoning,       │
                                │    rationales of merged sources        │
                                │  - hard cap 12 per pipeline run (not   │
                                │    per cluster — clusters may be small)│
                                └────────────────────────────────────────┘
                                                       │
                                                       ▼
                                Phase 2.5 evaluator (grounding gate)
                                drops any competency with grounding < 0.80
```

---

## Output schema (target)

```python
@dataclass
class CompetencyV2:
    # Identity
    id: str                                      # e.g. "comp_a1b2c3"
    title: str                                   # verb-led, curriculum-style learning outcome
    description: str                             # 1-3 sentences, assessable

    # Content
    related_skills: List[str]                    # hard skills (drawn from cluster items)
    soft_skills_required: List[str]              # 3-6, from top_soft_skills + LLM judgment
    soft_skills_description: str                 # 1 sentence, how the soft skills support

    # Reasoning (NEW for v2 — explainability)
    rationale: str                               # Why these skills cluster into THIS competency.
                                                 # Curated by LLM during generation; 2-4 sentences.
                                                 # User-facing; will be rendered on the public detail page.

    # Provenance (NEW for v2 — Req 6)
    contributing_item_ids: List[str]             # item.id of every cluster item that contributed
    source_job_ids: List[str]                    # unique job_ids across contributing items
    source_sentences: List[str]                  # unique sentence_texts; capped at 20 for UI
    source_sentence_ids: List[str]               # matching sentence_ids (for canonical lookup)

    # Future-awareness
    future_relevance: str                        # 1 sentence; cites future_weight or empirical trend
    future_weight: float                         # max future_weight across related_skills
    empirical_trend: str                         # "Emerging" / "Declining" / "Stable" / "Mixed"

    # KKNI (informational, Phase 2.3 labeller may overwrite)
    kkni_level: int                              # 1-9
    kkni_level_source: str                       # "llm_suggested" or "sbert_labeller"
    kkni_match_similarity: Optional[float]       # SBERT cosine if relabelled, else None

    # Evaluator preview (Phase 2.5 computes the canonical value)
    grounding_score_preview: float               # 0-1; |verified| / |total related_skills|
    grounding_flagged: bool                      # True if preview < 0.80

    # Generator metadata (for reproducibility audit)
    generated_at: str                            # ISO timestamp
    generator_version: str                       # "v2.0"
    cluster_id: str                              # which Phase 2.1 cluster produced this
    batch_reasoning_id: str                      # foreign key into batch_reasonings table
```

A separate `BatchReasoning` table stores **one record per LLM call**:

```python
@dataclass
class BatchReasoning:
    id: str                                      # FK target for competency.batch_reasoning_id
    cluster_id: str
    timestamp: str
    model: str                                   # e.g. "deepseek/deepseek-v3.2"
    prompt_sha256: str                           # for prompt versioning audit
    batch_reasoning: str                         # the LLM's chain-of-thought for THIS batch
                                                 # (which skills cluster, why, what overlaps were
                                                 #  resolved, what was excluded and why)
    n_skills_in: int
    n_competencies_out: int
    raw_response: str                            # full LLM response, for debugging
```

`batch_reasoning` is **separate from per-competency `rationale`**:
- `batch_reasoning` = the LLM's top-level thinking when looking at the cluster as a whole. 1 record per LLM call. Useful for paper-level audit ("why did the LLM produce these 4 competencies from this cluster of 12 skills?").
- `rationale` = per-competency justification. 1 record per competency. Useful for user-facing dashboard ("why is THIS specific competency a real, distinct thing?").

---

## Prompt design (chain-of-thought + structured output)

The legacy prompt asks for JSON only. The v2 prompt is two-part:

### Part 1 — System role (unchanged structure, minor edits)

```
You are an expert in competency-based education and vocational curriculum design.
You are given a CLUSTER of verified hard skills extracted from real job postings.
Your job is to synthesize them into curriculum-ready competency statements that a
school or institution could include in a software-engineering curriculum.
```

### Part 2 — User prompt (rewritten for cluster input + reasoning request)

```
The skills below are grouped into a single semantic CLUSTER (mean intra-cluster
SBERT cosine = 0.71). Each line shows the skill text + optional future-weight /
empirical-trend annotations + the count of source sentences.

CLUSTER ID: cluster_42
CLUSTER SUMMARY (heuristic): "web security and authentication"

SKILLS IN THIS CLUSTER:
- implementing OAuth 2.0 flows (future_weight=0.81, domain=Networks & Cybersecurity;
  empirical_trend=Emerging; 12 source sentences across 8 jobs)
- session management (future_weight=0.81, domain=Networks & Cybersecurity; 9 source
  sentences across 7 jobs)
- input validation (future_weight=0.81, domain=Networks & Cybersecurity; 18 sentences,
  15 jobs)
- ...

YOUR TASK:

First, in a `reasoning` field, explain your thinking step-by-step:
  1. What sub-themes do you see in this cluster?
  2. Should they be ONE competency or MULTIPLE? Justify the split.
  3. Are there skills that don't fit any competency? List them and say why
     (we'd rather drop weak fits than force-include them).
  4. Which skills did you keep, which did you drop, and why?

Then, in a `competencies` array, produce 1-4 well-anchored competency statements.
For each competency, write a `rationale` (2-4 sentences) explaining specifically:
  - Why this title / description matches the included skills
  - What learner could demonstrate if they achieve it
  - Why this is distinct from any other competency in this batch

[KKNI table, future-weighting rules, anti-hallucination rule, soft-skill rule, etc.
 — copied from legacy prompt]

RULES:
- HARD CAP: at most 4 competencies per cluster (legacy was 12 per batch; clusters
  are smaller and more coherent so the per-cluster cap is tighter).
- related_skills MUST come ONLY from the cluster's skills. Do NOT invent.
- Each competency must include AT LEAST 2 related_skills (singletons are usually
  too narrow to be a curriculum competency; drop them or merge with another).
- Be concrete and assessable. Avoid HR jargon.

OUTPUT (JSON only, no markdown fences):
{
  "reasoning": "<your batch-level thinking, 3-6 sentences>",
  "competencies": [
    {
      "title": "...",
      "description": "...",
      "related_skills": ["...", "..."],
      "rationale": "<why this competency is well-anchored and distinct>",
      "soft_skills_required": ["...", "..."],
      "soft_skills_description": "...",
      "future_relevance": "...",
      "kkni_level": 5
    }
  ]
}
```

**Key changes from legacy prompt:**
- Cluster context explicit (cluster_id, summary, cohesion)
- Per-skill provenance hint (source-sentence count) — helps LLM weight high-evidence skills
- Explicit `reasoning` field requested with structured questions
- Per-competency `rationale` requested
- Tighter cap (4 per cluster vs 12 per batch) because clusters are smaller
- Validation rule: `related_skills` MUST be from input

---

## Cluster → competency mapping

Phase 2.1's `Cluster` object provides:
- `items`: list of `SkillItem | KnowledgeItem` (each with sentence_id, sentence_text)
- `cohesion_score`: mean pairwise SBERT cosine inside the cluster
- `centroid_embedding`: for downstream KKNI labeller similarity
- `summary_label`: heuristic short label (e.g., "web security and auth") from a top-k frequency analysis of item texts. NOT used as input to the LLM prompt itself, but logged for human audit.

Generator iterates clusters. For each cluster:

1. **Skip clusters too small** (`len(items) < 2`) — singletons are too narrow to be a competency. Log to `dropped_clusters.csv` for audit.

2. **Build prompt** as above.

3. **Call LLM** with retry/backoff (same `_call_with_retry` pattern as `sentence_relevance_filter.py`).

4. **Parse output**:
   - Strict JSON parse with `_parse_llm_json` (existing helper)
   - Validate every competency has `related_skills`, `title`, `description`, `rationale`
   - Drop competencies with `related_skills` not in cluster items (anti-hallucination)
   - Drop competencies with `len(related_skills) < 2` (singleton-skill competencies)

5. **Attach provenance**:
   - For each competency, walk `related_skills` → find matching cluster items
   - Aggregate `contributing_item_ids`, `source_job_ids`, `source_sentence_ids`, `source_sentences`
   - Compute `grounding_score_preview = matched / len(related_skills)`

6. **Persist** to `BatchReasoning` table + add to in-memory competency list.

After all clusters processed, run the legacy 3-stage dedup pipeline (normalize-key match → semantic-title SBERT → Jaccard merge) but **with provenance merging**: when two competencies merge, the survivor's `contributing_item_ids` / `source_*` lists become the union of both inputs; rationales are concatenated with a `[merged from comp_xxx + comp_yyy]` marker.

Finally apply the **hard cap 12 per pipeline run** (legacy was per-batch; cluster-driven natural count is usually around 30-50 pre-dedup, 15-25 post-dedup, so 12 is still a meaningful filter).

---

## Provenance preservation — invariants

After Phase 2.2, every competency MUST satisfy:

| Invariant | Check |
|---|---|
| `contributing_item_ids` non-empty | `len(c.contributing_item_ids) >= 2` |
| `source_job_ids` non-empty | `len(c.source_job_ids) >= 1` |
| `source_sentences` non-empty | `len(c.source_sentences) >= 1` |
| Every `related_skill` traces to ≥ 1 contributing item | `grounding_score_preview >= 1.0 / len(related_skills)` |
| `rationale` is long-form and not boilerplate | `300 <= len(c.rationale) <= 900 chars AND not c.rationale.startswith("This competency")` |
| `batch_reasoning_id` resolvable | `BatchReasoning.lookup(c.batch_reasoning_id) is not None` |

Phase 2.5 evaluator's stricter gate (`grounding_score ≥ 0.80`) applies on top.

---

## Reasoning logging — the user-facing story

When a curriculum designer (or anyone on the public dashboard) opens `/competencies/<id>`, they see:

```
┌─────────────────────────────────────────────────────────────┐
│ Competency: Implement secure authentication flows           │
│                                                             │
│ KKNI Level 5 · Education level: D3/D4 · Software domain    │
│                                                             │
│ Description: <2-3 sentences>                                │
│                                                             │
│ ─── Related skills ────────────────────────────────────────│
│ • OAuth 2.0 flows  (12 jobs)                               │
│ • session management  (9 jobs)                              │
│ • input validation  (18 jobs)                               │
│                                                             │
│ ─── Why this competency? (LLM rationale) ─────────────────│
│ "These three skills together describe how a junior engineer │
│  builds the authentication layer of a web application. They │
│  cluster naturally because all three appear together in     │
│  job postings for backend roles. I kept them as one         │
│  competency rather than splitting because in practice they  │
│  must be designed and tested as an integrated whole — you   │
│  can't ship OAuth without input validation."                │
│                                                             │
│ ─── Source job postings (provenance) ──────────────────────│
│ 23 unique sentences across 18 job postings                  │
│ [View source sentences ▾]                                   │
│   "Strong understanding of OAuth 2.0 and session management │
│    required..." — job in-abc123                              │
│   "Must implement input validation per OWASP guidelines..." │
│   ... [21 more]                                             │
│                                                             │
│ ─── Future-of-work alignment ──────────────────────────────│
│ Future weight: 0.81 (Networks & Cybersecurity, WEF 2025)   │
│ Empirical trend: Emerging                                   │
└─────────────────────────────────────────────────────────────┘
```

The **"Why this competency?"** block is the per-competency `rationale`. By default the UI shows the first ~200 chars with a **"Read more ▾"** toggle to expand the full long-form rationale (400–800 chars). Below that is a second collapsible labeled **"How the LLM grouped these skills (batch reasoning) ▾"** — public, collapsed by default — that exposes the full `batch_reasoning` string for the cluster.

---

## Cost / latency / model choice

- **Model**: DeepSeek-V3.2 via OpenRouter (current pipeline default for competency gen). Strong on Indonesian context, JSON-following, and reasoning. Cost ~$0.0002 per cluster call.
- **Cost ballpark**: 50 clusters per pipeline run × $0.0002 = ~$0.01/run for the generator. Negligible against the $1000 budget.
- **Token budget**: max_tokens=1500 per call (legacy was 2000; tighter cap because cluster size is smaller). Reasoning + 4 competencies × ~150 tokens each fits comfortably.
- **Reasoning models (e.g., Claude 3.7 Sonnet thinking, GPT-5 reasoning)** are **deferred** for v2 sprint. The current Chain-of-Thought-in-prompt approach is sufficient. Switching to a thinking model is a 1-config-line swap if needed.

---

## Migration path

1. **Land `competency_generator_v2.py`** alongside legacy `generate_competencies.py`. Toggle via `AdvancedPipelineConfig.COMPETENCY_GENERATOR_VERSION = "v1" | "v2"`. Default "v1" until Phase 2.5 evaluator is also ready (so we don't ship un-vetted v2 competencies to users).

2. **Phase 2.5 evaluator lands.** Grounding gate enabled.

3. **Flip default to "v2"** in pipeline-redesign-v2 Phase 2.6 (UI deployment for user testing).

4. **Delete legacy** after user testing concludes (post-deadline). Until then, keep legacy reachable for ablation / paper baseline comparison.

The toggle prevents the situation where v2 has a bug we don't catch and users see broken competencies during user testing.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM `reasoning` field is generic ("These skills are all about X.") | Validate reasoning length ≥ 100 chars + ≥ 2 sentences. Re-prompt once if too short. |
| LLM hallucinates skills not in cluster | Validation step drops competencies with extra-cluster `related_skills`. Counted in `dropped_hallucinated_count` audit metric. |
| Per-competency `rationale` leaks PII from job postings | Rationales are LLM-summarized, not verbatim quotes. PII risk low. Manual spot-check of first 50 generated competencies during smoke test. |
| Token budget overrun for large clusters (> 20 items) | Cap input at 15 highest-future-weight items per call; remaining items go to a sibling cluster sub-call if total cluster > 30. |
| Generator output unparseable JSON | `_parse_llm_json` already handles markdown-fence stripping + truncated-trailing-brace recovery. Retry once on parse failure. |
| Reasoning + rationale add ~25% to output token cost | Acceptable: ~$0.0001 → ~$0.00013 per call. Within budget. |

---

## Resolved decisions (locked 2026-05-12)

1. **Rationale length: long (400–800 chars, ~4–6 sentences).** UI renders the first ~200 chars by default and exposes a **"Read more ▾"** toggle to expand the full rationale. Keeps detail page clean for browsing; preserves the full justification for users who want to audit it.

2. **`batch_reasoning` is public via "View reasoning ▾" toggle.** On the competency detail page, a collapsible block titled "How the LLM grouped these skills (batch reasoning)" exposes the full `batch_reasoning` string. Collapsed by default. Same UI pattern as the rationale toggle. Aligns with the "no black boxes" mission — full audit trail visible to any user, not gated behind admin login.

3. **Add `reasoning_quality` to expert-review rubric.** Section 3 of `docs/EXPERT_REVIEW_RUBRIC.md` gets a 5th Likert column: *"This competency's rationale clearly explains why these skills belong together. (1 = Strongly Disagree, 5 = Strongly Agree)"*. Adds ~30 seconds per competency × 75 competencies per reviewer ≈ ~40 min total reviewer-time overhead. Becomes a paper finding (RQ5 augmentation): correlation between `reasoning_quality` rating and `grounding_score`.

4. **A/B test two LLM model families in parallel.** Run identical clusters through:
   - **DeepSeek-V3.2 via OpenRouter** (current default for v2 competency gen) — cheap, good JSON.
   - **GPT-5 (or GPT-4o if GPT-5 not yet on Jatevo) via Jatevo** — reasoning-grade output, separate $1000 budget pool, no extra cost from OpenRouter pool.
   - Selected by `AdvancedPipelineConfig.COMPETENCY_LLM_PROVIDER = "openrouter_deepseek" | "jatevo_gpt"`. Per-batch override possible.
   - Comparison: side-by-side on 5–10 clusters, measure (a) `rationale` informativeness (manual rate 1–5), (b) hallucination rate (skills not in cluster), (c) `batch_reasoning` quality. Decide default after data, not vibes.
   - **Provider-routing rule (durable):** Jatevo serves **GPT models only** (key: `api_keys/jatevo.txt`). OpenRouter serves everything else (key: `api_keys/OpenRouter.txt`). Implement `get_competency_llm_client(model_name)` to dispatch by model family — prevents the past 401 incident from `2aced75`.

5. **Keep v1 and v2 in parallel until Phase 2.5 evaluator lands, then flip default to v2.** `AdvancedPipelineConfig.COMPETENCY_GENERATOR_VERSION = "v1" | "v2"`, default "v1" through Week 2. Pipeline runs that explicitly request "v2" use the new path; users on default surface stay on v1. After Phase 2.5 evaluator + grounding gate land (Week 2 end), flip default to "v2". v1 stays callable for ablation / paper baseline; deleted only after user testing concludes. Safety net intact.

---

## File deliverables (when implemented)

| Path | Purpose | LOC estimate |
|---|---|---|
| `competency_generator_v2.py` | New cluster-driven generator | ~600 |
| `competency_v2_schema.py` | Dataclasses for `CompetencyV2`, `BatchReasoning` | ~80 |
| `tests/test_competency_v2.py` | Unit tests (golden cluster → expected competencies smoke test) | ~200 |
| `docs/PHASE_2_2_DESIGN.md` (this doc) | Design reference | — |
| `pipeline.py` update | Wire v2 toggle | ~30 lines diff |
| `dashboard/templates/competency_detail.html` update | Show rationale + provenance | ~50 |

Total: ~1,000 new lines + ~80 diff. 4 days estimated.

---

## Implementation order (when sprint Week 2 kicks off)

1. **Schema dataclasses** (~1h) — `CompetencyV2`, `BatchReasoning`. No LLM dependency.
2. **Prompt builder** + few-shot examples (~3h) — adapt legacy `build_prompt` for cluster input.
3. **Generator loop** with retry/backoff (~4h) — bulk of the work.
4. **Provenance attachment** + validation (~3h) — the explainability-critical glue.
5. **Dedup with provenance merging** (~3h) — adapt legacy `_deduplicate_competencies`.
6. **Unit tests with golden clusters** (~3h) — synthetic cluster → expected schema.
7. **Pipeline.py integration toggle** (~1h).
8. **Dashboard template update** (~2h) — show rationale + provenance UI.
9. **Smoke test on real cluster output from Phase 2.1** (~2h) — end-to-end.

Total: ~22 person-hours. ≈ 3 working days. 0.5 day buffer for dedup edge cases.

---

## Verification — what "Phase 2.2 done" looks like

1. ✅ `competency_generator_v2.py` exists and is invoked when `COMPETENCY_GENERATOR_VERSION = "v2"`.
2. ✅ Every emitted competency has non-empty `contributing_item_ids`, `source_job_ids`, `source_sentences`.
3. ✅ Every competency has a `rationale` of 300–900 chars (long-form).
4. ✅ A `BatchReasoning` record exists per LLM call, FK'd from competencies.
5. ✅ `grounding_score_preview` computed and attached to every competency.
6. ✅ Unit tests pass: synthetic 3-skill cluster → expected 1-2 competencies with valid rationale.
7. ✅ Smoke test on real Phase 2.1 cluster output (when 2.1 lands): top-20 competencies have human-readable rationales that don't look templated.
8. ✅ Dashboard detail page renders rationale + provenance side-by-side.

After 1-8, Phase 2.5 evaluator lands and gates ≥ 0.80 grounding score on the actual emitted competencies.
