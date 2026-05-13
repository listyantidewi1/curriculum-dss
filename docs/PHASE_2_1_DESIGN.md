# Phase 2.1 — Skill clustering (design)

**Status:** design draft, 2026-05-12. Pre-implementation. Source code to land in `clustering/` package. Implements `.kiro/specs/pipeline-redesign-v2/requirements.md` Req 5. Produces `Cluster` objects consumed by Phase 2.2.

**Sprint slot:** Week 1 tail / Week 2 head (2026-05-15 → 2026-05-18), ~2 days estimated.

---

## Context — why we need this phase

Phase 2.2's cluster-driven competency generator (see `docs/PHASE_2_2_DESIGN.md`) needs `Cluster` objects as input. Today there's no such object: the legacy `generate_competencies.py` does sub-clustering inside its own prompt builder, mixing concerns and making provenance audit difficult.

Phase 2.1 promotes clustering to a first-class pipeline stage so that:

1. **Cluster identity is durable.** Every cluster gets a stable `id` that Phase 2.2 records into `competency.cluster_id`. Auditors can trace a competency back to its source cluster, and the cluster back to its source items + sentences + jobs.
2. **Clustering decisions are reproducible.** Deterministic with seed control — required for the stability experiment (Jaccard top-20 vs N) and for paper reproducibility.
3. **Cohesion is measurable and gated.** Every cluster carries `cohesion_score` (mean intra-cluster SBERT cosine). Phase 2.2 skips clusters below threshold; the paper reports cohesion distribution as a clustering-quality metric.
4. **Layer-2 LLM items and Skill-LLM items co-cluster.** Provenance `extractor_source` is preserved on each item, so the cluster knows which extractor each member came from — useful for Layer 1 / Layer 2 agreement analysis (Req 9.2).

---

## Target architecture

```
Phase 1 output (per pipeline.py):                  ┌──────────────────────────────────┐
  Lists of SkillItem + KnowledgeItem               │ SkillItem / KnowledgeItem:       │
  Each item carries:                               │   text                           │
    - text                                         │   type (SkillType.HARD/SOFT)     │
    - type (SkillItem only)                        │   confidence_score               │
    - confidence_score                             │   source / extractor_source      │
    - sentence_id, sentence_text                   │   sentence_id, sentence_text     │
    - extractor_source                             │   (KnowledgeItem has no `type`)  │
                                                   └──────────────────────────────────┘
                                                                  │
                                                                  ▼
                                ┌────────────────────────────────────────┐
                                │  clustering/skill_clusterer.py         │
                                │                                        │
                                │  1. Dedup & canonicalize items by text │
                                │     (collapse "Python" + "python" +    │
                                │      "PYTHON" → one canonical item     │
                                │      with merged provenance)           │
                                │                                        │
                                │  2. Split into 3 streams:              │
                                │     a. HARD skills + KNOWLEDGE (mix)   │
                                │     b. SOFT skills (separate)          │
                                │     c. dropped: < min_global_frequency │
                                │                                        │
                                │  3. For each stream:                   │
                                │     a. SBERT-embed item texts          │
                                │     b. HDBSCAN(min_cluster_size=3)     │
                                │     c. Compute cohesion per cluster    │
                                │     d. Drop clusters with cohesion     │
                                │        < cohesion_threshold (0.50)     │
                                │     e. Agglomerative recovery on       │
                                │        HDBSCAN noise points            │
                                │     f. Split oversized clusters (>25)  │
                                │     g. Name clusters (TF-IDF+bigrams)  │
                                │                                        │
                                │  4. Emit Cluster objects               │
                                └────────────────────────────────────────┘
                                                  │
                                                  ▼
                                Phase 2.2 competency generator
                                (consumes hard+knowledge clusters;
                                 soft skills passed as a global
                                 top-K list, NOT clustered)
```

---

## Output schema

```python
@dataclass
class Cluster:
    # Identity
    id: str                                 # e.g. "cluster_h0042" (h=hard+knowledge), "cluster_s0007" (s=soft)
    stream: str                             # "hard_plus_knowledge" | "soft_skill"
    method: str                             # "hdbscan" | "agglomerative_recovery" | "agglomerative_split"

    # Members — note: items is heterogeneous for the hard+knowledge stream
    items: List[Union[SkillItem, KnowledgeItem]]
    n_items: int                            # len(items) — convenience
    n_unique_jobs: int                      # unique job_ids across items
    n_skill_items: int                      # how many of items are SkillItem
    n_knowledge_items: int                  # how many are KnowledgeItem

    # Quality metrics
    cohesion_score: float                   # mean pairwise SBERT cosine inside the cluster (Req 5)
    cohesion_std: float                     # std of pairwise cosines — flag heterogeneous clusters
    centroid_embedding: np.ndarray          # mean of item embeddings; used by KKNI labeller (Phase 2.3)

    # Naming
    summary_label: str                      # TF-IDF + bigram heuristic, ~3-6 words
    top_terms: List[str]                    # top-5 TF-IDF terms for the cluster

    # Generator metadata (for reproducibility audit)
    seed: int                               # numpy/random seed used
    embedder_model: str                     # e.g. "all-MiniLM-L6-v2"
    clusterer_version: str                  # "v2.1"
```

`SkillItem.type` distinguishes Hard vs Soft inside the stream split. `KnowledgeItem` has no `type` field — items of this class are routed to the hard+knowledge stream unconditionally.

---

## Why co-cluster hard skills + knowledge?

KKNI defines a competency as `knowledge + skill + attitude`. A competency like *"Implement secure authentication flows"* combines:

- skills: "implementing OAuth 2.0", "session management"
- knowledge: "OAuth 2.0 protocol", "HTTP security headers"

If we cluster skills and knowledge in separate spaces, Phase 2.2 has to re-join them — adding complexity and a join-quality failure mode. Co-clustering means the same SBERT embedding space surfaces both, and Phase 2.2 receives one coherent cluster per competency to reason over.

**Item-type preservation:** every cluster item still has its class (`SkillItem` or `KnowledgeItem`). Phase 2.2's prompt can render them differently if needed ("SKILLS: ..." vs "KNOWLEDGE: ..."). Phase 2.3 (KKNI labeller) and Phase 2.5 (grounding evaluator) consume item-type-aware fields.

**Why NOT cluster soft skills with hard skills:** soft skills ("teamwork", "communication", "adaptability") live in a different semantic space. Embedding "communication" near "implementing OAuth" produces a cohesion-violating cluster, and the LLM in Phase 2.2 already handles soft skills via a separate top-K reference list (legacy behavior, retained).

---

## Clustering pipeline — step by step

### Step 1 — Canonicalize & deduplicate items

Items with semantically identical text but different casing / whitespace / punctuation collapse to one canonical item with merged provenance:

```
"Python", "python", "PYTHON", "  Python  " → canonical "Python"
canonical.sentence_ids = union of all source sentence_ids
canonical.extractor_source = "fused" if mixed extractors, else the single source
```

Canonicalization uses lowercase + strip + collapse whitespace. ESCO-style normalization is explicitly **not** applied (preserves verbs per the extractor decision).

**Frequency filter:** drop canonical items appearing in only 1 job posting. Rationale: a single-job mention is almost always noise or extreme niche; the stability experiment (Jaccard top-20) shows these contribute to run-to-run instability. Configurable: `MIN_GLOBAL_FREQUENCY = 2` (default 2; set to 1 for ablation).

### Step 2 — Stream split

```
HARD + KNOWLEDGE stream = [SkillItem where type == HARD] ∪ [all KnowledgeItem]
SOFT stream = [SkillItem where type == SOFT]
```

Run the rest of the pipeline independently on each stream. The two streams' outputs are tagged with `stream` for downstream routing.

### Step 3 — Embed

SBERT model: **`sentence-transformers/all-mpnet-base-v2`** (768d). Stronger than MiniLM on short technical phrases; the ~3× CPU cost is negligible at our scale (≤10K items). Alternative: `all-MiniLM-L6-v2` (384d, faster) for the stability experiment's large-N runs.

Each item text is wrapped in a light template before embedding: `"skill or knowledge area: <text>"` — improves cohesion for very short items ("Docker" → "skill or knowledge area: Docker" is more anchored than the bare token).

Cache embeddings to `cache/embeddings/<sha256(text)>.npy` so reruns and the stability experiment don't repay the embedding cost.

### Step 4 — HDBSCAN primary clustering

```python
HDBSCAN(
    min_cluster_size=3,         # singletons + pairs are too narrow to seed a competency
    min_samples=2,              # noise threshold; lower = more clusters, more noise
    metric="euclidean",         # SBERT embeddings are normalized → equiv to cosine
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.15,  # merge tightly-similar microclusters
)
```

Why HDBSCAN over k-means / agglomerative-only:
- No need to choose `k` (we don't know the cluster count a priori — it varies with corpus size)
- Variable density tolerance: tech-heavy clusters are dense; long-tail clusters are sparser
- Native noise label `-1` — exposes items that don't fit anywhere, which we then recover separately

### Step 5 — Compute cohesion + cohesion gate

For each cluster:

```
cohesion_score = mean of pairwise SBERT cosines for all (i,j) pairs in cluster, i<j
cohesion_std   = stddev of same
```

Drop clusters where `cohesion_score < COHESION_THRESHOLD` (default **0.55**; Req 5 floor 0.50). Dropped clusters' items go back to the noise pool for agglomerative recovery (Step 6). Log to `dropped_clusters.csv` with reason `"low_cohesion"` for audit.

Cohesion threshold is the single most impactful knob. **Default 0.55** was chosen via real-data sweep on `results.old2/` Phase 1 output (3,196 items → 145 after frequency filter; sweep at 0.40/0.45/0.50/0.55/0.60/0.65). See `results/clustering_sweep/sweep_metrics.png`. Findings:

| Threshold | Effect on this corpus |
|---|---|
| 0.40 / 0.45 / 0.50 | Identical results — HDBSCAN's natural floor was 0.527, so no extra weak clusters appeared even at 0.40. **15 clusters, 2 incoherent.** |
| **0.55** (chosen default) | Drops the two cross-domain noise clusters ("agile + CNC + gaming" coh=0.531; "APIs + AutoCAD + PowerShell" coh=0.527). **13 clean clusters survive, min cohesion 0.601.** |
| 0.60 | Identical to 0.55 on this corpus — no further dropouts. Reserved as the strict-ablation point. |
| 0.65 | Too strict — kills AWS/cloud, AI/ML, DevOps, all valuable clusters. **6 clusters.** |

Paper ablation sweep: report **0.50 / 0.55 / 0.60** (the meaningful range on this data).

### Step 6 — Agglomerative recovery on noise

HDBSCAN noise points (label `-1`) get a second pass: agglomerative clustering with Ward linkage (Padovano's choice), stopping when no new cluster would meet `COHESION_THRESHOLD`. Recovered clusters are tagged `method="agglomerative_recovery"` so the paper can quantify how much each method contributes.

Items still uncategorized after recovery go to `ungrouped_items.csv` for the audit log. They do **not** participate in Phase 2.2 — better to drop than to force-fit.

### Step 7 — Split oversized clusters

Clusters with `n_items > 25` exceed Phase 2.2's prompt token budget (legacy `max_tokens=1500` accommodates ~15 items × 100 tokens each + reasoning + 4 competencies). Split by re-running Ward agglomerative on the cluster with `n_clusters = ceil(n_items / 15)`. Children tagged `method="agglomerative_split"`. Each child inherits the parent's `summary_label` with a suffix (`"<parent>: sub-cluster 1"`).

### Step 8 — Cluster naming (heuristic)

For each cluster:

1. Concatenate item texts. Compute TF-IDF over this corpus (cluster-level) vs the full pipeline corpus (background frequencies).
2. Extract the top-5 unigrams + top-3 bigrams + top-2 trigrams by TF-IDF score.
3. Combine into a short label: `<most-distinctive-trigram or bigram>: <top-unigram>, <2nd unigram>`.
4. Fallback: if TF-IDF is degenerate (all uniform), use the first item's text as the label.

Example output for an OAuth/auth cluster:
- `summary_label = "Authentication & session security: OAuth, session, validation"`
- `top_terms = ["OAuth", "session management", "input validation", "authentication", "JWT"]`

Naming is **heuristic** — Phase 2.2 LLM produces the canonical competency title. The `summary_label` is for human-readable audit logs and dashboard cluster-browser.

---

## Configuration parameters

```python
@dataclass
class ClusteringConfig:
    embedder_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_cache_dir: str = "cache/embeddings/"

    min_global_frequency: int = 2          # drop items appearing in < N jobs
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.15

    cohesion_threshold: float = 0.50       # Req 5
    max_cluster_size: int = 25             # split threshold

    seed: int = 42                          # reproducibility
    enable_agglomerative_recovery: bool = True
    enable_oversize_split: bool = True
```

All knobs surface in `AdvancedPipelineConfig` so the stability experiment can sweep them.

---

## Provenance — invariants

After Phase 2.1, every emitted `Cluster` MUST satisfy:

| Invariant | Check |
|---|---|
| `n_items >= 3` | (after splitting / recovery) |
| `cohesion_score >= 0.50` | (Req 5 threshold) |
| Every item has `sentence_id` OR `sentence_text` non-null | enables Phase 2.2 provenance attachment |
| `centroid_embedding` shape matches embedder dim | (e.g., 768 for mpnet) |
| `seed` and `embedder_model` recorded | enables stability-experiment replay |
| Items' `extractor_source` preserved | enables Layer 1 / Layer 2 agreement analysis |

---

## Edge cases & how we handle them

| Case | Handling |
|---|---|
| Pipeline run with < 50 items | Skip HDBSCAN (won't find structure); fall back to agglomerative-only with `n_clusters = max(3, n_items // 5)`. |
| All items go to noise (HDBSCAN finds no dense cluster) | Pure agglomerative pass on full set. Log a warning; the paper reports % of runs that hit this fallback. |
| 80%+ of items end up in one giant cluster | Trigger oversize split (Step 7) recursively. Cap recursion depth at 3. |
| Two streams produce wildly different cluster counts (e.g., 50 hard+knowledge clusters, 2 soft) | Expected. Soft-skill space is small (~50 unique items in practice); 2–5 soft clusters is typical. Not an error. |
| Items with identical text but different `extractor_source` ("BERT", "LLM", "BERT+LLM") | Canonicalized to one item; `extractor_source` becomes `"fused"`. Original sources preserved in a `source_history` field for audit. |
| Empty embedder cache + 10K items + cold start | First run takes ~3 min on CPU for embedding; subsequent runs hit cache → < 5s. Document in `INTEGRATION.md`. |

---

## Cost / latency

- **Local CPU** — no API calls. SBERT embedding + HDBSCAN + agglomerative all run on the user's laptop or in CI.
- **First run on 10K items:** ~3 min embedding + ~30s clustering + ~10s naming ≈ **~4 min total**.
- **Cached runs:** ~30s clustering + ~10s naming ≈ **~40s total**.
- **Memory:** 10K × 768d float32 = ~30MB embeddings; HDBSCAN scales to 50K points comfortably.

No GPU required. No new infrastructure.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| SBERT embeddings degrade on very short tokens ("AWS", "K8s") | Template wrap: `"skill or knowledge area: <text>"` provides context. Validated on a sample of 50 short tokens during smoke test. |
| HDBSCAN finds many tiny clusters (< 5 items each) | `min_cluster_size=3` already filters singletons + pairs. If problem persists, raise to 5 (config). |
| Cohesion threshold too strict → too many ungrouped items | Default 0.50 is empirically reasonable; sweep 0.40 / 0.50 / 0.60 in the paper ablation. |
| Cluster-naming heuristic produces low-quality labels | Labels are for audit only; Phase 2.2 LLM produces the canonical competency title. Fallback (first-item-as-label) catches the worst cases. |
| Stability across runs: same N, different seeds → different clusterings | HDBSCAN is deterministic for fixed input + parameters. The seed knob is for the random-sample step in the stability experiment, NOT for clustering itself. Document this clearly. |
| Co-clustered skills + knowledge confuse Phase 2.2 LLM | Prompt distinguishes items by class tag (`SKILL:` vs `KNOWLEDGE:`). Manual spot-check first 10 clusters during smoke test. |
| Oversize-split children have weird summary labels | Inherit parent label with `": sub-cluster N"` suffix. Curation effort low. |
| Caching bug: stale cache after embedder model change | Cache filename includes `sha256(model_name + text)`. Model swap invalidates cache cleanly. |

---

## Open questions (need user input before implementation)

None at this point. The five Phase 2.2 questions are now locked; Phase 2.1's defaults follow the v2 spec (Req 5) directly. If anything surprises us during implementation we'll surface it then.

---

## File deliverables (when implemented)

| Path | Purpose | LOC estimate |
|---|---|---|
| `clustering/__init__.py` | Package init | ~10 |
| `clustering/skill_clusterer.py` | Main `cluster_skills(items, config) → List[Cluster]` entry point + pipeline | ~400 |
| `clustering/cluster_naming.py` | TF-IDF + bigram heuristic labeler | ~120 |
| `clustering/cluster_schema.py` | `Cluster`, `ClusteringConfig` dataclasses | ~80 |
| `cache/embeddings/` (dir) | gitignored SBERT embedding cache | — |
| `tests/test_skill_clusterer.py` | Unit tests + golden-input smoke test | ~250 |
| `docs/PHASE_2_1_DESIGN.md` (this doc) | Design reference | — |
| `pipeline.py` update | Wire clustering between Phase 1 fusion and Phase 2.2 generator | ~40 lines diff |

Total: ~860 new lines + ~40 diff. ~2 working days.

---

## Implementation order (when sprint kicks off)

1. **Schema dataclasses** (~1h) — `Cluster`, `ClusteringConfig`. No external dependencies beyond `numpy`.
2. **Item canonicalization** (~2h) — lowercase + dedup + provenance merge. Easy unit-testable.
3. **Embedder wrapper** with disk cache (~2h) — wraps `sentence-transformers` with sha256-keyed cache.
4. **HDBSCAN primary pass** (~2h) — straightforward; uses `hdbscan` library (already in `requirements.txt`? — verify; if not, `pip install hdbscan`).
5. **Cohesion computation** (~1h) — pairwise cosine; numpy-only.
6. **Agglomerative recovery** (~2h) — scipy hierarchical clustering on noise points.
7. **Oversize split** (~1h) — same agglomerative on > 25-item clusters.
8. **Cluster naming heuristic** (~2h) — scikit-learn TfidfVectorizer + bigram extraction.
9. **Pipeline.py integration** (~1h) — call between fusion and Phase 2.2.
10. **Unit tests with synthetic items** (~3h) — golden in/out fixtures.
11. **Smoke test on real Phase 1 output** (~2h) — end-to-end, verify cohesion distribution.

Total: ~19 person-hours ≈ 2.5 working days. Allow 0.5 day buffer.

---

## Verification — what "Phase 2.1 done" looks like

1. ✅ `clustering/skill_clusterer.py` exists with `cluster_skills(items, config)` entry point.
2. ✅ Returned `Cluster` objects all have `cohesion_score >= 0.50` (Req 5).
3. ✅ Items are split into two streams (`hard_plus_knowledge` and `soft_skill`); soft stream isn't fed into Phase 2.2.
4. ✅ `summary_label` populated and human-readable on top-20 clusters from real data.
5. ✅ Embedding cache works: second run on same input is > 50× faster than first.
6. ✅ Unit tests pass: synthetic 30-item input → expected ≥ 3 clusters with cohesion ≥ 0.50.
7. ✅ Smoke test on real Phase 1 output: cohesion distribution histogram shows median ≥ 0.55, tail ≥ 0.50.
8. ✅ `dropped_clusters.csv` and `ungrouped_items.csv` written for audit.
9. ✅ `pipeline.py` calls clustering between Phase 1 fusion and Phase 2.2 entry; `Cluster` objects flow correctly.
10. ✅ Stability: re-running with same seed + same input produces byte-identical cluster IDs and memberships.

After 1–10, Phase 2.2 generator consumes the clusters and produces competencies with full provenance.

---

## What we explicitly do NOT do in Phase 2.1

- **Do not cluster soft skills.** Pass them as a global top-K list to Phase 2.2 instead.
- **Do not LLM-name clusters.** Heuristic naming is sufficient for audit; LLM produces competency titles in Phase 2.2.
- **Do not normalize to ESCO / O*NET.** Preserves verb-led phrasing per the extractor decision.
- **Do not skip items below `min_global_frequency`** without logging. They go to `dropped_items.csv` so the paper can quantify what's left out.
- **Do not feed clustering results back into Phase 1.** No re-extraction loops in v2.
- **Do not depend on UMAP for dimensionality reduction.** SBERT 768d + HDBSCAN handles 10K points fine; UMAP adds a stochastic step that complicates reproducibility.
