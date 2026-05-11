# Calculations, Ranking, Voting, and Decision Logic

This document provides a **scientifically rigorous** description of all formulas, weighting schemes, voting rules, and ranking decisions used in the Future-Aware Hybrid Skill Extraction pipeline and dashboard.

**For detailed scientific methods with worked examples**, see **[SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md)** (binomial test, effect sizes, Cohen's/Fleiss' Kappa, power analysis, FDR, multi-comparison correction, etc.).

---

## 1. Pipeline Priority Score (recommendations.py)

The **priority_score** for curriculum recommendations is a weighted linear combination of normalized signals. **Coverage is not used by default** because the system is designed to help schools design better curriculum; existing curriculum may be outdated and is used only for insights, not for prioritization.

```
priority_score = w_demand × demand_norm
               + w_trend × trend_score_norm
               + w_future × future_norm
               + (w_coverage × coverage_gap  if use_coverage else 0)
```

### Default Weights (use_coverage=False)

| Signal | Weight | Rationale |
|--------|--------|-----------|
| demand | 0.40 | Job-market demand (frequency) is primary |
| trend | 0.30 | Empirical emerging/declining from time series |
| future | 0.30 | Forecast-aligned future relevance |
| coverage | 0.0 | Not used by default; curriculum is for insights only |

### Signal Definitions

**demand_norm** (∈ [0, 1]):
```
demand_norm = demand_freq / max(demand_freq)
```
- `demand_freq`: count of job postings mentioning the skill
- Normalized so the most frequent skill has value 1.0

**trend_score_norm** (∈ [-1, 1]):
- **Emerging**: `min(1.0, |slope| / 0.5)` — positive, capped at 1
- **Declining**: `max(-1.0, -|slope| / 0.5)` — negative, capped at -1
- **Stable**: 0
- Slope from FDR-controlled time-series regression (Benjamini–Hochberg q < 0.05)

**future_norm** (∈ [0, 1] for positive weights):
```
future_norm = future_weight / max(|future_weight|)
```
- `future_weight` from `future_weight_mapping.py` (see §2)

**coverage_gap** (∈ [0, 1]):
```
coverage_gap = 1.0 - mean(coverage_percentage) / 100
```
- 1.0 = skill not covered by curriculum; 0.0 = fully covered
- **Note:** Coverage is informational only. It is not used in the default priority score because schools use the system to design better curriculum; existing curriculum may be outdated.

### Ablation

Each signal can be disabled for ablation study:
- `use_trend`, `use_future`, `use_coverage`, `use_validity`
- When disabled, that term is 0 in the formula
- The `with_coverage` variant enables coverage for comparison

---

## 2. Future Weight (future_weight_mapping.py)

The **future_weight** maps skills/knowledge to forecast domains (WEF, O*NET, McKinsey) via SBERT embeddings:

```
future_weight = similarity(item, best_domain) × trend_score
```

### Components

**similarity**:
- Cosine similarity between item embedding and domain example terms
- Best domain = argmax over all domains in `future_domains.csv`

**trend_score**:
- From domain metadata: Strong_Growth ≈ 1.0, Moderate_Growth ≈ 0.5–0.9, Decline ≈ -0.3 to -0.6
- Encodes forecast direction for that domain

**mapping_margin** (uncertainty):
```
mapping_margin = top1_similarity - top2_similarity
```
- Large margin = confident mapping; small margin = ambiguous

---

## 2.1 Normalization and Grouping (Skills, Knowledge, Competencies)

To reduce redundancy from case and punctuation variants (e.g., "AI agents" vs "AI Agents", "AI/ML" vs "AI ML"), the pipeline applies **deterministic normalization** before aggregation.

### Normalization Rule (reproducible)

```
normalize(text) = lowercase(strip(text))
                ; collapse [\s_/|,;:.()\[\]{}]+ → single space
                ; collapse repeated spaces
```

Applied in: `future_weight_mapping.py`, `recommendations.py`, `generate_competencies.py` (via `domain_batching.normalize_for_grouping` for skill lookup in future_skill_weights), dashboard `_normalize_knowledge_key`.

### Grouping Logic

| Item type | Group key | Canonical form | Aggregation |
|-----------|-----------|----------------|-------------|
| Skills | normalize(skill) | Most frequent raw form | demand_freq = sum; type = mode (bloom column removed in pipeline-redesign-v2 Phase 1.3) |
| Knowledge | normalize(knowledge) | Most frequent raw form | freq = sum; mean_confidence = mean |
| Competencies | normalize(title) | First in group | occurrence_count = group size |

### Scientific Rationale

- **Deterministic**: Same input always yields same key.
- **Occurrence preservation**: Grouping sums occurrence counts; high occurrence indicates higher demand.
- **Canonical form**: Most frequent raw string used for display and embedding to avoid information loss.

---

## 3. Majority Vote (Multi-Reviewer Feedback)

When multiple reviewers assess the same item, the pipeline uses **majority vote** to merge feedback.

### Algorithm (import_feedback.py, dashboard/app.py)

```python
def _majority(votes: list) -> str:
    # Strip and filter non-empty
    votes = [str(v).strip() for v in votes if v and str(v).strip()]
    if not votes:
        return ""
    # Most common value; ties broken by first occurrence
    return Counter(votes).most_common(1)[0][0]
```

### Applied To

| Field | Valid Values | Use |
|-------|--------------|-----|
| human_valid | valid, invalid | Skill/knowledge validity |
| ~~human_bloom~~ | — | **Removed in pipeline-redesign-v2 Phase 1.3.** Bloom corrections no longer collected from reviewers. |
| human_type | Hard, Soft, Both | Skill type correction |
| human_quality | 1–5, poor, fair, good, excellent | Competency quality |
| human_relevant | yes, no, partial | Competency relevance |

### Inter-Rater Reliability

When ≥2 reviewers assess the same overlap items:

- **2 raters:** Cohen's Kappa: `κ = (P_o - P_e) / (1 - P_e)`
- **3+ raters:** Fleiss' Kappa (see SCIENTIFIC_METHODOLOGY.md §5)

Overlap items are selected via stratified random sampling (not first-N) at
gold set export time to avoid order bias.

Stored in `feedback_store/inter_rater_report.json` per feedback source.

---

## 3.0 Effect Sizes (Extraction Evaluation)

For proportions (e.g., precision vs chance):

**Odds ratio** (vs p0 = 0.5):
```
odds_ratio = (p / (1 - p)) / (p0 / (1 - p0)) = p / (1 - p)  when p0 = 0.5
```

**Risk difference**:
```
risk_difference = p - p0
```

**Cohen's h** (for comparing two proportions p1, p2):
```
h = 2 × (arcsin(√p1) - arcsin(√p2))
```
- Small: |h| ≈ 0.2; Medium: |h| ≈ 0.5; Large: |h| ≈ 0.8

---

## 3.1 Recall Limitation (Gold-Set Design)

In the gold-set evaluation paradigm:
- **Precision** = tp / n (correct extractions among those labeled) — **reported**.
- **Recall** cannot be computed: the gold set is a stratified sample of pipeline *outputs*, not an exhaustive annotation of all true items per posting.  False negatives (fn) are unknown.
- **F1** is therefore not reported (it would equal precision by construction).

See [SCIENTIFIC_METHODOLOGY.md §10](SCIENTIFIC_METHODOLOGY.md#10-recall-limitation-gold-set-design) for full discussion.

---

## 4. Dashboard Ranking (Skills)

### Model-Only Mode

Sort key: `(-priority_score, rank, -demand_freq)`
- Uses pipeline `priority_score` as-is
- No human feedback influence

### Human-Adjusted Mode

```
human_adjusted_score = priority_score + status_boost
```

**status_boost**:
- `verified`: +0.05
- `invalid`: -0.15
- `not_verified`: 0

Sort key: `(-human_adjusted_score, rank, -demand_freq)`

**Verification status** is derived from `feedback_store/skill_feedback.csv` merged with `expert_review_skills.csv` on `review_id`, using majority vote on `human_valid` per skill.

---

## 5. Dashboard Ranking (Knowledge)

### Knowledge Aggregation

Duplicate knowledge items (same normalized text across jobs) are **aggregated**:

```
normalize(text) = lower(strip(text)), collapse punctuation/spaces
```

Per normalized key:
- `occurrence_count`: total mentions across jobs
- `confidence_score`: mean of per-row confidence
- `job_count`: distinct job_ids

**Rationale**: High occurrence indicates high demand; aggregation avoids inflating rankings with duplicates.

### Model-Only Score

```
model_score = 0.45 × conf_norm + 0.25 × fw_norm + 0.30 × occ_norm
```

- `conf_norm`: confidence_score (already 0–1)
- `fw_norm`: future_weight / max(future_weight)
- `occ_norm`: occurrence_count / max(occurrence_count)

### Human-Adjusted Score

```
human_score = model_score + status_adj
```

**status_adj**:
- `verified`: +0.12
- `invalid`: -0.18
- `not_verified`: 0

---

## 6. Dashboard Ranking (Competencies)

### Base Score (from Related Skills)

```
c_score = max(priority of related_skills)
```
- Uses skill `_human_adjusted_score` or `_model_score` depending on ranking mode
- Competency inherits the highest-priority related skill

### Human-Adjusted Score

```
human_score = 0.7 × c_score + 0.3 × q_score + rel_adj + verify_adj
```

**q_score** (quality mapping):
| human_quality | q_score |
|---------------|---------|
| 1, poor | 0.2 |
| 2, fair | 0.4 |
| 3, good | 0.7 |
| 4, 5, excellent | 0.8–1.0 |

**rel_adj** (relevance):
| human_relevant | rel_adj |
|----------------|---------|
| yes | +0.20 |
| partial | +0.08 |
| no | -0.20 |

**verify_adj**:
- `verified`: +0.12
- else: 0

---

## 7. FDR-Controlled Trend Detection (skill_time_trend_analysis.py)

**Benjamini–Hochberg** procedure controls false discovery rate:

1. Compute p-value per skill (regression slope vs time)
2. Sort p-values ascending
3. Find largest k such that p_(k) ≤ (k/m) × α, with α = 0.05
4. Reject hypotheses 1..k → q-value = p × m / rank

**trend_label**:
- Emerging: slope > 0, q < 0.05
- Declining: slope < 0, q < 0.05
- Stable: else

---

## 8. Calibrated Verification (validate_parameters.py)

Validates pipeline confidence scores against human labels:

- **AUC-ROC**: discrimination ability
- **Brier score**: calibration quality (MSE of predicted vs observed)
- **Calibration curve**: 10 bins, predicted prob vs observed frequency

**Threshold selection:** The optimal confidence threshold is selected via
**5-fold cross-validation** (CV) to avoid circularity (fitting and evaluating
on the same data).  Each fold fits the Youden-optimal threshold on training
data and evaluates on held-out data.  The exported threshold
(`calibrated_threshold.json`) is the **mean of per-fold thresholds**, not the
full-data optimum.  If CV fails (insufficient data), falls back to the
full-data Youden threshold with `threshold_selection: "full_data_fallback"`.

---

## 9. Recommendation Evaluation (recommendations.py)

When expert labels `expert_priority` (yes/no/partial) exist for top-20:

**Precision@N**:
```
P@N = (yes + 0.5×partial) / N
```

**NDCG@N**:
```
DCG = Σ (rel_i / log2(i+2))
NDCG = DCG / IDCG
```
- rel: yes=1, partial=0.5, no=0
- IDCG = DCG of ideal ranking

---

## 9.5 Competency Grounding Score (Phase 2.5 — hallucination gate)

The grounding score measures whether each generated competency is empirically
grounded in extracted skills/knowledge from real job postings. It is the
**anti-hallucination defence** for the recommendation system.

### Formula

For each competency `c` with declared `related_skills = [s_1, s_2, …, s_k]`
and a provenance set `evidence_pool = {extracted items contributing to c}`:

```
grounding(c) = | {s_i in related_skills : exists e in evidence_pool with
                   SBERT_cos(s_i, e) ≥ tau_grounding} | / k
```

Where:
- `tau_grounding = 0.70` (configurable; SBERT cosine similarity threshold)
- `k = |related_skills|`
- A `related_skill` is "grounded" if at least one extracted item in the
  competency's provenance is semantically close to it (cosine ≥ tau)

### Gate

```
if grounding(c) < 0.80:
    flag c as "potentially_hallucinated"
    exclude c from the dashboard recommendation surface
```

The 0.80 threshold means at least 4 out of every 5 `related_skills` must
trace back to actual evidence. A competency that fails this gate is **never
shown to users** — hallucination is a technical impossibility at the
recommendation surface, not a quality problem reviewers must spot-check.

### Reporting

Per `metrics_competency.txt`:
- `grounding_mean`, `grounding_median`, `grounding_p10` across all generated competencies
- `flagged_hallucinated_count` (competencies excluded by the gate)
- `coherence_mean` (Req 7.2: mean pairwise SBERT cosine within `related_skills`)
- `coverage` (Req 7.3: fraction of extracted items represented in at least one surviving competency)

### Worked Example

Competency: "Implement secure REST APIs with authentication"
- `related_skills` = ["implementing REST APIs", "designing authentication flows", "applying OAuth 2.0", "input validation"] (k=4)
- `evidence_pool` includes extracted items: "build RESTful endpoints", "OAuth 2.0 integration", "JWT token validation", "session management"
- Cosine similarities (SBERT):
  - "implementing REST APIs" vs "build RESTful endpoints": 0.84 ≥ 0.70 ✓
  - "designing authentication flows" vs "session management": 0.72 ≥ 0.70 ✓
  - "applying OAuth 2.0" vs "OAuth 2.0 integration": 0.91 ≥ 0.70 ✓
  - "input validation" vs all items: max 0.62 < 0.70 ✗
- grounding = 3/4 = 0.75
- **0.75 < 0.80 → FLAGGED as potentially hallucinated, excluded from recommendations**

The "input validation" skill couldn't be traced to a specific evidence item;
the competency over-claimed. Excluding it prevents misleading users.

---

## 9.6 Top-20 Stability Metric (production-volume experiment)

For declaring stability of competency recommendations across sample sizes
(Phase 2.5b stability experiment, see [SCIENTIFIC_METHODOLOGY.md §16c](SCIENTIFIC_METHODOLOGY.md)):

### Within-N stability (across seeds)

For 3 runs at the same N with different random seeds, producing top-20
competency lists A₁, A₂, A₃:

```
within_N_jaccard(N) = mean( Jaccard(A_i, A_j) for i < j )
                    = (J(A_1,A_2) + J(A_1,A_3) + J(A_2,A_3)) / 3

Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Competencies compared by **normalized title** (`normalize_for_grouping`) so
cosmetic rephrasing doesn't break matching.

### Across-N stability

For consecutive sample sizes N and N_prev:

```
across_N_jaccard(N, N_prev) = mean over seeds of
                              Jaccard(top20_at_N, top20_at_N_prev)
```

### Stability declaration

N is **declared stable** when:
```
within_N_jaccard(N) ≥ 0.80   AND   across_N_jaccard(N, N_prev) ≥ 0.85
```

The smallest such N is the **stability point N\***. Production runs use
N ≥ N* to guarantee stable recommendations.

---

## 10. Competency–Knowledge Enrichment (Dashboard)

Lexical overlap (tokens ≥4 chars) between competency (title, description, related_skills) and knowledge items:

```
overlap = |skill_tokens ∩ knowledge_tokens|
```

Top 8 knowledge items by `(overlap DESC, confidence DESC, text)` attached as `related_knowledge`.

---

## Summary Table

| Component | Formula / Rule |
|-----------|----------------|
| Pipeline priority | 0.40×demand + 0.30×trend + 0.30×future (coverage for insights only) |
| Future weight | similarity × trend_score |
| Majority vote | Most frequent non-empty value |
| Skill human boost | +0.05 (valid), -0.15 (invalid) |
| Knowledge model | 0.45×conf + 0.25×fw + 0.30×occ |
| Knowledge human adj | +0.12 (valid), -0.18 (invalid) |
| Competency human | 0.7×skill + 0.3×quality + rel_adj + verify_adj |
| FDR threshold | q < 0.05 |
| P@N | (yes + 0.5×partial) / N |
| Grounding score (Phase 2.5 hallucination gate) | grounded_count / |related_skills|; gate at ≥ 0.80 |
| Within-N Jaccard (stability) | mean pairwise Jaccard of top-20 across 3 seeds at same N; ≥ 0.80 declares stable |
| Across-N Jaccard (stability) | Jaccard(top20_N, top20_N-1); ≥ 0.85 declares diminishing returns |

---

## 11. Reproducibility: Random Seeds

All sampling operations use `config.RANDOM_SEED` (default: 42) as the single
source of truth.  Scripts that accept a `--seed` CLI argument override this
value for the current run.

| Script | Seed Source | Usage |
|--------|-------------|-------|
| `pipeline.py` | `config.RANDOM_SEED` / `--seed` | Job sampling |
| `export_gold_set.py` | `--seed` (default 42) | Gold set stratified sampling, IRR overlap selection |
| `export_for_review.py` | `config.RANDOM_SEED` | Skill/knowledge stratified sampling for review |
| `export_competencies_for_review.py` | `config.RANDOM_SEED` | Competency batch sampling |
| `validate_parameters.py` | `config.RANDOM_SEED` | CV fold shuffling |
| `skill_time_trend_analysis.py` | `config.RANDOM_SEED` | Stability analysis base seed |
| `plot_generator.py` | `config.RANDOM_SEED` | KMeans clustering |

*Last updated: pipeline-redesign-v2 (May 2026) — Bloom validation removed; type validation retained; sentence-level provenance plus Skill-LLM extractor in Phase 1.5.*
