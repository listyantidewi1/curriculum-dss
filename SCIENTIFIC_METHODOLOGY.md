# Scientific Methodology

This document provides a **detailed, scientifically rigorous** description of all statistical methods, formulas, and decision logic used in the Future-Aware Hybrid Skill Extraction pipeline. It includes worked examples for every major procedure.

**Related documents:**
- [CALCULATIONS.md](CALCULATIONS.md) — Formulas and summary tables
- [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) — Research questions, gold set design, limitations
- [PIPELINE.md](PIPELINE.md) — Pipeline flow, data dependencies

---

## Table of Contents

1. [Binomial Test for Extraction Precision](#1-binomial-test-for-extraction-precision)
2. [Effect Sizes for Proportions](#2-effect-sizes-for-proportions)
3. [Multi-Comparison Correction](#3-multi-comparison-correction)
4. [Power Analysis for Gold Set Size](#4-power-analysis-for-gold-set-size)
5. [Inter-Rater Reliability (Cohen's Kappa and Fleiss' Kappa)](#5-inter-rater-reliability-cohens-kappa-and-fleiss-kappa)
6. [Normalization and Grouping](#6-normalization-and-grouping)
7. [Priority Score and Future Weight](#7-priority-score-and-future-weight)
8. [FDR-Controlled Trend Detection](#8-fdr-controlled-trend-detection)
9. [Weight Sensitivity Analysis](#9-weight-sensitivity-analysis)
10. [Recall in the Gold-Set Paradigm](#10-recall-in-the-gold-set-paradigm)
11. [Calibrated Verification Metrics](#11-calibrated-verification-metrics)
12. [Future-Domain Mapping Evaluation](#12-future-domain-mapping-evaluation)
13. [Domain-Based Batching for Competency Generation](#13-domain-based-batching-for-competency-generation)
14. [Recommendation Evaluation (P@N, NDCG)](#14-recommendation-evaluation-pn-ndcg)

---

## 1. Binomial Test for Extraction Precision

### Purpose

To test whether an extractor's precision is better than chance. We use a one-sided binomial test against the null hypothesis that precision = 0.5 (random guessing).

### Hypothesis

- **H₀**: precision = 0.5
- **H₁**: precision > 0.5

### Formula

Let X ~ Binomial(n, p₀) where n = number of labeled items, p₀ = 0.5. We reject H₀ if the observed count of correct extractions is in the critical region:

```
P(X ≥ k | p₀ = 0.5) ≤ α
```

We choose the smallest k such that P(X ≥ k | p₀) ≤ α = 0.05. The p-value is:

```
p-value = P(X ≥ observed_correct | p₀ = 0.5)
```

### Implementation

Uses `scipy.stats.binomtest(n_correct, n, p=0.5, alternative="greater")`.

### Worked Example

**Scenario:** Hybrid extractor has 120 correct out of 150 labeled skills.

- n = 150, n_correct = 120
- precision = 120/150 = 0.80
- H₀: p = 0.5
- p-value = P(X ≥ 120 | n=150, p=0.5)

For large n, the binomial distribution is approximately normal. In practice:

- Expected under H₀: 75 correct
- Observed: 120 correct
- The p-value is extremely small (≈ 0), so we **reject H₀** and conclude precision is significantly better than chance.

**Result:** `p_value_vs_chance ≈ 0`, `significant_at_005 = True`

### Wilson Score Confidence Interval

For a proportion p with n observations, the Wilson score interval gives better coverage than the normal approximation, especially for small n or extreme p:

```
centre = p + z²/(2n)
margin = z × √(p(1-p)/n + z²/(4n²))
denom = 1 + z²/n
lower = (centre - margin) / denom
upper = (centre + margin) / denom
```

With z = 1.96 (95% CI).

**Example:** precision p = 0.72, n = 100  
centre = 0.72 + 3.84/200 = 0.739, margin ≈ 0.088, denom ≈ 1.038  
CI ≈ **[0.628, 0.798]**

---

## 2. Effect Sizes for Proportions

### Odds Ratio (vs Chance)

When comparing precision p against chance p₀ = 0.5:

```
odds(p) = p / (1 - p)
odds(p₀) = p₀ / (1 - p₀) = 1   (when p₀ = 0.5)

odds_ratio = odds(p) / odds(p₀) = p / (1 - p)
```

**Interpretation:**
- OR = 1: no effect vs chance
- OR > 1: better than chance (e.g., OR = 4 → 4× more likely to be correct than incorrect)
- OR < 1: worse than chance

### Risk Difference

```
risk_difference = p - p₀
```

**Interpretation:** Absolute increase in correct proportion over chance. For p₀ = 0.5, risk_difference = p - 0.5.

### Cohen's h (Comparing Two Proportions)

For comparing two proportions p₁ and p₂:

```
h = 2 × (arcsin(√p₁) - arcsin(√p₂))
```

**Effect size interpretation:**
- |h| ≈ 0.2: small
- |h| ≈ 0.5: medium
- |h| ≈ 0.8: large

### Worked Examples

**Example 1 — Odds ratio and risk difference**

- precision p = 0.80, p₀ = 0.5
- odds_ratio = 0.80 / 0.20 = **4.0**
- risk_difference = 0.80 - 0.5 = **0.30**

**Example 2 — Cohen's h (BERT vs LLM)**

- BERT precision p₁ = 0.65, LLM precision p₂ = 0.50
- h = 2 × (arcsin(√0.65) - arcsin(√0.50))
- arcsin(√0.65) ≈ 0.947, arcsin(√0.50) ≈ 0.785
- h ≈ 2 × 0.162 ≈ **0.32** (small-to-medium effect)

---

## 3. Multi-Comparison Correction

When testing multiple sources (BERT, LLM, Hybrid) or performing pairwise comparisons, we correct for multiple testing to control the family-wise error rate (FWER) or false discovery rate (FDR).

### Bonferroni Correction (Per-Source Tests)

For k independent tests, multiply each p-value by k, or equivalently use α/k instead of α:

```
p_adjusted = min(1, p_raw × k)
```

**Example:** 3 sources, raw p-values [0.02, 0.08, 0.001]. Bonferroni:
- 0.02 × 3 = 0.06 (not significant at 0.05)
- 0.08 × 3 = 0.24
- 0.001 × 3 = 0.003 (still significant)

### Benjamini-Hochberg (Pairwise Comparisons)

For pairwise two-proportion z-tests between sources, we apply Benjamini-Hochberg FDR:

1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ … ≤ p_(m)
2. Find largest k such that p_(k) ≤ (k/m) × α
3. Reject hypotheses 1..k; q-value for each: q_i = p_i × m / rank_i

**Example:** 3 pairwise comparisons, p-values [0.03, 0.06, 0.12], α = 0.05

| Rank | p-value | BH threshold (k/3 × 0.05) | Reject? |
|------|---------|---------------------------|---------|
| 1 | 0.03 | 0.0167 | No (0.03 > 0.0167) |
| 2 | 0.06 | 0.0333 | No |
| 3 | 0.12 | 0.05 | No |

No rejections after BH. If p-values were [0.01, 0.02, 0.10]:
- 0.01 ≤ 0.0167 → reject first
- 0.02 ≤ 0.0333 → reject second
- 0.10 > 0.05 → do not reject third

### Two-Proportion Z-Test

For comparing BERT (n₁, x₁ correct) vs LLM (n₂, x₂ correct):

```
p₁ = x₁/n₁,  p₂ = x₂/n₂
p_pool = (x₁ + x₂) / (n₁ + n₂)
SE = √[p_pool(1 - p_pool) × (1/n₁ + 1/n₂)]
z = (p₁ - p₂) / SE
p-value = 2 × (1 - Φ(|z|))   [two-tailed]
```

---

## 4. Power Analysis for Gold Set Size

### Purpose

Determine the minimum number of labeled items (n) needed to detect a meaningful precision improvement over chance with adequate statistical power.

### Assumptions

| Parameter | Default | Meaning |
|-----------|---------|---------|
| p₀ | 0.5 | Null (chance) |
| p₁ | 0.7 | Alternative (RQ1 target) |
| α | 0.05 | Type I error |
| Power | 0.80 | 1 - β (Type II error) |

### Method

1. Find critical value k: smallest integer such that P(X ≥ k | p₀, n) ≤ α
2. Power = P(X ≥ k | p₁, n)

We search for smallest n such that power ≥ 0.80.

### Worked Example

- p₀ = 0.5, p₁ = 0.7, α = 0.05, target power = 0.80
- For n = 37: critical k ≈ 24; P(X ≥ 24 | p₁=0.7, n=37) ≈ 0.80
- **Result:** n ≥ 37 required for power 0.80

**Current gold set:**
- Skills: n = 150 → power ≈ 1.00
- Knowledge: n = 100 → power ≈ 0.99

### Running the Script

```bash
python scripts/power_analysis_gold_set.py --p0 0.5 --p1 0.7
```

---

## 5. Inter-Rater Reliability (Cohen's Kappa and Fleiss' Kappa)

### When to Use Which

| Raters | Metric |
|--------|--------|
| 2 | Cohen's Kappa |
| 3+ | Fleiss' Kappa |

### Cohen's Kappa (2 Raters)

```
κ = (P_o - P_e) / (1 - P_e)
```

- **P_o** = observed agreement = fraction of items where both raters agree
- **P_e** = expected agreement by chance = Σ (freq₁(j) × freq₂(j)) over categories j

**Interpretation:**
- κ < 0: worse than chance
- κ = 0: chance agreement
- κ ∈ (0, 0.20]: slight
- κ ∈ (0.20, 0.40]: fair
- κ ∈ (0.40, 0.60]: moderate
- κ ∈ (0.60, 0.80]: substantial
- κ > 0.80: almost perfect

**Example:**

| Item | Rater A | Rater B |
|------|---------|---------|
| 1 | valid | valid |
| 2 | valid | invalid |
| 3 | invalid | invalid |
| 4 | valid | valid |
| 5 | invalid | invalid |

- P_o = 4/5 = 0.80
- P_e: freq_A(valid)=3/5, freq_A(invalid)=2/5; freq_B(valid)=2/5, freq_B(invalid)=3/5
- P_e = (3/5)(2/5) + (2/5)(3/5) = 12/25 + 6/25 = 0.72
- κ = (0.80 - 0.72) / (1 - 0.72) = 0.08 / 0.28 ≈ **0.29** (fair agreement)

### Fleiss' Kappa (3+ Raters)

For each subject i and category j, let n_ij = number of raters who assigned subject i to category j. N = number of raters, n = number of subjects.

```
P_i = (1 / (N×(N-1))) × Σ_j [n_ij × (n_ij - 1)]
P̄ = (1/n) × Σ_i P_i

p_j = proportion of all assignments to category j
P_e = Σ_j p_j²

κ = (P̄ - P_e) / (1 - P_e)
```

**Example:** 3 raters, 5 items, 2 categories (valid, invalid)

| Item | valid | invalid |
|------|-------|---------|
| 1 | 3 | 0 |
| 2 | 2 | 1 |
| 3 | 0 | 3 |
| 4 | 3 | 0 |
| 5 | 1 | 2 |

- For item 1: P₁ = (1/6) × [3×2 + 0] = 1
- For item 2: P₂ = (1/6) × [2×1 + 1×0] = 0.33
- … similarly for others
- P̄ = mean of P_i
- p_valid, p_invalid from column sums
- P_e = p_valid² + p_invalid²
- κ = (P̄ - P_e) / (1 - P_e)

---

## 6. Normalization and Grouping

### Normalization Rule

To merge equivalent items (e.g., "AI agents" vs "AI Agents", "AI/ML" vs "AI ML"):

```
normalize(text) = lowercase(strip(text))
                ; replace [\s_/|,;:.()\[\]{}]+ with single space
                ; collapse repeated spaces; strip
```

**Implementation (regex):**
```python
t = str(text).strip().lower()
t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
t = re.sub(r"\s+", " ", t).strip()
```

### Examples

| Raw Input | Normalized Key |
|-----------|----------------|
| "AI agents" | "ai agents" |
| "AI Agents" | "ai agents" |
| "AI/ML" | "ai ml" |
| "AI ML" | "ai ml" |
| "Python Programming" | "python programming" |
| "  Data Analysis  " | "data analysis" |

### Grouping Logic

| Item Type | Group Key | Canonical Form | Aggregation |
|-----------|-----------|----------------|-------------|
| Skills | normalize(skill) | Most frequent raw | demand_freq = sum; type = mode (bloom column removed in pipeline-redesign-v2 Phase 1.3) |
| Knowledge | normalize(knowledge) | Most frequent raw | freq = sum; mean_confidence = mean |
| Competencies | normalize(title) | First in group | occurrence_count = size of group |

### Worked Example

**Input (skills from multiple jobs):**
- "Python", "python", "Python " → all map to key "python"
- Raw counts: Python=45, python=30, "Python "=5
- **Canonical form:** "Python" (most frequent)
- **Aggregated demand_freq:** 45 + 30 + 5 = 80

---

## 7. Priority Score and Future Weight

### Priority Score (recommendations.py)

```
priority_score = 0.40 × demand_norm + 0.30 × trend_norm + 0.30 × future_norm
```

**Signal definitions:**

1. **demand_norm** = demand_freq / max(demand_freq) ∈ [0, 1]
2. **trend_norm** ∈ [-1, 1]:
   - Emerging: min(1.0, |slope| / 0.5)
   - Declining: max(-1.0, -|slope| / 0.5)
   - Stable: 0
3. **future_norm** = future_weight / max(|future_weight|) ∈ [0, 1]

**Coverage** is not used by default (curriculum is for insights only).

### Worked Example

Skill "Machine Learning":
- demand_freq = 80, max_demand = 100 → demand_norm = 0.80
- slope = 0.15 (emerging), |slope|/0.5 = 0.30 → trend_norm = 0.30
- future_weight = 0.7, max_fw = 0.9 → future_norm = 0.78

```
priority_score = 0.40×0.80 + 0.30×0.30 + 0.30×0.78
               = 0.32 + 0.09 + 0.234 = 0.644
```

### Future Weight (future_weight_mapping.py)

```
future_weight = similarity(item, best_domain) × trend_score
```

- **similarity:** cosine similarity between SBERT embedding of item and domain (name + example terms)
- **trend_score:** expert-assigned score from domain metadata (see below)

**Mapping margin (uncertainty):**
```
mapping_margin = top1_similarity - top2_similarity
```

- Large margin → confident mapping
- Small margin → ambiguous (top 2 domains very close)

**Example:**
- Skill "Deep Learning" vs domains: AI & Machine Learning (sim=0.85), Cloud (sim=0.72)
- best_domain = AI & Machine Learning, trend_score = 1.0
- future_weight = 0.85 × 1.0 = **0.85**
- mapping_margin = 0.85 - 0.72 = **0.13**

### Trend Score Derivation (future_domains.csv)

The `trend_score` assigned to each future domain is **not derived from the
job-posting data**.  It is an expert-assigned value based on external labour
market reports.  The table below documents each value and its source:

| Domain ID | Domain | trend_score | Source Report / Section |
|-----------|--------|-------------|------------------------|
| WEF01 | AI & Machine Learning | 1.0 | WEF Future of Jobs 2025 — top growing skill cluster |
| WEF02 | Big Data & Analytics | 0.9 | WEF Future of Jobs 2025 — 2nd growing cluster |
| WEF03 | Cybersecurity | 0.9 | WEF Future of Jobs 2025 — critical infrastructure need |
| WEF04 | Cloud Computing | 0.85 | WEF Future of Jobs 2025 — infrastructure demand |
| WEF05 | Digital Platforms | 0.7 | WEF Future of Jobs 2025 — moderate growth |
| WEF06 | Sustainability Tech | 0.6 | WEF Future of Jobs 2025 — emerging but niche |
| WEF07 | Human-AI Collaboration | 0.95 | WEF Future of Jobs 2025 — co-pilot paradigm |
| WEF08 | Digital Fluency | 0.85 | WEF Future of Jobs 2025 — foundational literacy |
| MCK04 | Critical Thinking | 0.8 | McKinsey Future of Work 2025 — human-centric skills |
| MCK05 | Resilience | 0.75 | WEF Future of Jobs 2025 — adaptability premium |
| ONET01 | Software Engineering | 0.65 | O*NET Bright Outlook 2024 — stable growth |
| ONET04 | Game Dev / XR | 0.6 | O*NET Bright Outlook 2024 — XR expansion |
| MCK03 | Leadership | 0.55 | McKinsey Future of Work 2025 |
| ONET03 | UX/UI Design | 0.55 | O*NET Bright Outlook 2024 |
| ONET05 | QA & Testing | 0.55 | O*NET Bright Outlook 2024 |
| ESCO01 | Project Management | 0.55 | ESCO v1.2 2024 |
| ONET02 | Database Admin | 0.5 | O*NET Bright Outlook 2024 |
| ONET06 | Blockchain | 0.5 | O*NET Bright Outlook 2024 — uncertain trajectory |
| MCK01 | Process Automation | 0.5 | McKinsey Future of Work 2025 |
| ESCO02 | Business Analysis | 0.5 | ESCO v1.2 2024 |
| MCK02 | Communication | 0.4 | McKinsey Future of Work 2025 — stable |
| DEC03 | Routine Maintenance | -0.3 | McKinsey Future of Work 2023 — automation risk |
| DEC02 | Legacy Web Dev | -0.4 | Expert consensus — declining relevance |
| DEC01 | Routine Data Entry | -0.6 | WEF Future of Jobs 2023 — high automation risk |

**Limitations:**
- Scores are ordinal judgements, not interval-scale measurements.
- Scores may not generalize to all geographies or industries.
- The specific values (e.g., 0.85 vs 0.90) are not precisely calibrated;
  the ranking order is more meaningful than the exact magnitude.

### Trend Score Sensitivity Analysis

To validate robustness, the pipeline supports a sensitivity analysis that
perturbs all trend_scores by ±0.2 and measures the impact on the top-20
recommendation ranking via Jaccard overlap (see `scripts/trend_score_sensitivity.py`).

```bash
python scripts/trend_score_sensitivity.py
```

A Jaccard overlap > 0.7 across perturbations indicates that the
recommendation ranking is robust to the exact trend_score values.

---

## 8. FDR-Controlled Trend Detection

### Purpose

Identify skills that are significantly emerging or declining over time while controlling the false discovery rate across many tests.

### Method

1. For each skill: linear regression of monthly frequency vs time
2. Obtain p-value for slope (H₀: slope = 0)
3. Apply **Benjamini-Hochberg** to get q-values
4. Label: Emerging if slope > 0 and q < 0.05; Declining if slope < 0 and q < 0.05; else Stable

### Benjamini-Hochberg (Skill Trends)

```
Sort p-values: p_(1) ≤ p_(2) ≤ … ≤ p_(m)
For i = m-1 down to 1:
  rank = i
  q_(i) = min(p_(i) × m / rank, q_(i+1))
```

### Worked Example

5 skills, p-values = [0.001, 0.02, 0.03, 0.15, 0.40], slopes = [+0.2, -0.15, +0.1, +0.05, -0.02]

| Rank | p | q = p×5/rank |
|------|---|--------------|
| 1 | 0.001 | 0.005 |
| 2 | 0.02 | 0.05 |
| 3 | 0.03 | 0.05 |
| 4 | 0.15 | 0.1875 |
| 5 | 0.40 | 0.40 |

Skills 1, 2, 3 have q ≤ 0.05:
- Skill 1: slope > 0 → **Emerging**
- Skill 2: slope < 0 → **Declining**
- Skill 3: slope > 0 → **Emerging**
- Skills 4, 5: **Stable**

---

## 9. Weight Sensitivity Analysis

### Purpose

Assess how robust the top-20 recommendations are to changes in the priority score weights.

### Method

1. **Baseline:** (w_demand=0.4, w_trend=0.3, w_future=0.3)
2. **Sweep** alternative weight configurations
3. For each config: compute top-20, then **Jaccard similarity** vs baseline

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### Configurations Tested

| Config | w_demand | w_trend | w_future |
|--------|----------|---------|----------|
| d0.3_t0.35_f0.35 | 0.30 | 0.35 | 0.35 |
| d0.4_t0.3_f0.3 | 0.40 | 0.30 | 0.30 (baseline) |
| d0.5_t0.25_f0.25 | 0.50 | 0.25 | 0.25 |
| d0.3_t0.2_f0.5 | 0.30 | 0.20 | 0.50 |
| d0.4_t0.2_f0.4 | 0.40 | 0.20 | 0.40 |
| d0.5_t0.2_f0.3 | 0.50 | 0.20 | 0.30 |

### Worked Example

- Baseline top-20: {A, B, C, …, T}
- Config d0.5_t0.25_f0.25 top-20: {A, B, D, E, …, U}
- Intersection: 16 skills
- Union: 24 skills
- **Jaccard = 16/24 ≈ 0.67**

High Jaccard → robust to weight changes.

### Running

```bash
python recommendations.py --sensitivity
```

Output: `weight_sensitivity_report.json`

---

## 10. Recall Limitation (Gold-Set Design)

### Why Recall Is Not Estimable

True **recall** = tp / (tp + fn) requires knowing the *complete* set of
correct items in the population.  Our gold set is a **stratified sample of
pipeline outputs**, not an exhaustive annotation of all skills present in
each job posting.  Consequently:

| Metric | Computable? | Reason |
|--------|-------------|--------|
| **Precision** | Yes | = (correct items) / (total labeled items) |
| **Recall** | **No** | We do not enumerate *all* true skills per posting; fn is unknown |
| **F1** | **No** | Requires both precision and recall |

### Design Consequence

Earlier versions of `evaluate_extraction.py` reported a `recall_estimate`
field numerically identical to precision, and an F1 derived from it.  These
have been **removed** because they are misleading: when recall = precision
by construction, F1 = precision, adding no new information.

The report now contains **precision only**.

### When Recall Would Be Meaningful

To estimate recall one would need a **corpus-annotation gold set**: a random
sample of job postings where *every* true skill is exhaustively enumerated
by domain experts.  This is substantially more expensive than the current
output-sampling design and is left for future work.

---

## 11. Calibrated Verification Metrics

### AUC-ROC

Area under the Receiver Operating Characteristic curve. Measures discrimination: ability to rank correct extractions higher than incorrect.

- AUC = 0.5: random
- AUC = 1.0: perfect

### Brier Score

Mean squared error of predicted probability vs observed outcome:

```
Brier = (1/n) × Σ (predicted_i - observed_i)²
```

- 0 = perfect calibration
- 0.25 = random (for binary)

### Calibration Curve

10 bins of predicted probability. For each bin: mean predicted prob vs observed frequency. Ideal: points on diagonal.

### Youden Index

Maximize (sensitivity + specificity - 1) to choose optimal threshold.

---

## 12. Future-Domain Mapping Evaluation

### Top-1 Accuracy

For each item with expert-labeled `true_domain_id`, compare to pipeline `best_domain_id`:

```
top1_accuracy = count(true_domain_id == best_domain_id) / n_evaluable
```

Items with `true_domain_id` in {none, unclear} are excluded from evaluation.

### Mapping Margin

`mapping_margin = top1_similarity - top2_similarity` indicates mapping confidence. Higher margin → more confident assignment.

---

## 13. Domain-Based Batching for Competency Generation

### Purpose

To group skills by thematic domain before sending them to the LLM, so each competency-generation batch contains related skills (e.g., AI/ML together, cloud together). This reduces forced groupings of unrelated skills and improves scientific validity.

### Confidence Thresholds

Skills with domain assignments from `future_skill_weights.csv` are classified by confidence:

- **High confidence:** similarity ≥ 0.45 and mapping_margin ≥ 0.05 → use `best_future_domain`
- **Low confidence:** below either threshold → assign to "Uncertain" batch

Low-confidence skills are not forced into a specific domain to avoid false thematic grouping.

### Normalized-Key Lookup

To reduce coverage gap from case/punctuation variants (e.g., "AI/ML" vs "AI ML"), `load_skill_future_weights` builds a dual index:

1. Exact key: `str(skill).strip()`
2. Normalized key: `normalize_for_grouping(skill)` (lowercase, collapse punctuation)

Lookup order: exact first, then normalized.

### On-the-Fly Fallback

Skills not in `future_skill_weights.csv` get domain assignment via embedding similarity to `future_domains.csv` (when available). If no future_domains or embedding fails, they go to "Unmapped".

### Batch Merge (Strongly Similar Domains)

When two domain batches have domains with cosine similarity ≥ 0.7 (of their embeddings: domain_text + example_terms), they are merged. This avoids tiny batches (e.g., 5 skills) while preserving thematic coherence.

---

## 14. Recommendation Evaluation (P@N, NDCG)

Used when experts label top-N curriculum recommendations in `recommendations.csv` (expert_priority column).

### Precision@N

```
P@N = (count(yes) + 0.5 × count(partial)) / N
```

Expert labels: yes / partial / no for top-N recommendations.

**Example:** N=20, 12 yes, 4 partial, 4 no  
P@20 = (12 + 2) / 20 = **0.70**

### NDCG@N

```
rel_i = 1 (yes), 0.5 (partial), 0 (no)
DCG = Σ (rel_i / log₂(i+2))
IDCG = DCG of ideal ranking (all yes first)
NDCG = DCG / IDCG
```

**Example (N=5):** rel = [1, 0.5, 1, 0, 1]  
DCG = 1/log₂(2) + 0.5/log₂(3) + 1/log₂(4) + 0 + 1/log₂(6) ≈ 1 + 0.315 + 0.5 + 0 + 0.387 ≈ 2.20  
IDCG (ideal [1,1,1,0.5,0]) ≈ 2.89  
NDCG ≈ 2.20/2.89 ≈ **0.76**

---

## 15. Future-Domain Trend Score Derivation

### Source and Assignment Process

The `trend_score` values in `future_domains.csv` are **expert-assigned ordinal
scores** derived from published labour-market reports.  They are NOT
data-derived; this is a known limitation.

| Source Tag | Report | Trend Score Range |
|------------|--------|-------------------|
| WEF_FutureOfJobs_2025 | World Economic Forum, *Future of Jobs Report 2025* | 0.60–1.00 (growth) |
| WEF_FutureOfJobs_2023 | WEF, *Future of Jobs Report 2023* (for decline domains) | -0.60 (decline) |
| ONET_BrightOutlook_2024 | O*NET Bright Outlook Occupations, 2024 update | 0.50–0.65 |
| ESCO_v1.2_2024 | ESCO Classification v1.2, skills pillar | 0.50–0.55 |
| McKinsey_FutureOfWork_2025 | McKinsey Global Institute, *The Future of Work* | 0.40–0.80 |
| Expert_consensus | Pipeline authors' consensus (legacy web dev decline) | -0.40 |

### Mapping from Qualitative to Quantitative

| Qualitative Label | Score Range | Rationale |
|-------------------|-------------|-----------|
| Strong_Growth | 0.75–1.00 | Domains explicitly listed as "fastest growing" or "most in demand" in WEF 2025 top-10 |
| Moderate_Growth | 0.40–0.70 | Domains present in Bright Outlook or mentioned as growing but not top priority |
| Stable | 0.00–0.30 | Not explicitly mentioned as growing or declining |
| Decline | -0.60 to -0.30 | Domains flagged as "declining" or "at risk of automation" in WEF/McKinsey |

### Limitation

These scores encode **ordinal expert judgment**, not calibrated probabilities.
The exact numeric values (e.g., 0.85 vs 0.90) carry less meaning than their
ranking.  The sensitivity analysis below validates that recommendation
rankings are robust to ±0.2 perturbations of these scores.

### Sensitivity Analysis

The script `scripts/trend_score_sensitivity.py` perturbs each domain's
`trend_score` by ±0.2 (capped at [-1, 1]) and recomputes the top-20
recommendation rankings.  Jaccard overlap vs the unperturbed baseline
measures robustness.

Run:

```bash
python scripts/trend_score_sensitivity.py
```

Output: `results/trend_score_sensitivity_report.json`

---

## 16. Extraction Architecture and Weight Justification

### 16a. Extraction Design (Direction A)

| Aspect | Design | Rationale |
|--------|--------|-----------|
| **BERT scope** | Per sentence | JobBERT has 128-token limit; short segments suit NER. Preprocess splits on bullet boundaries and paragraph breaks; max_len 600 chars. |
| **LLM scope** | Full job description | Preserves document-level context; does not miss skills/knowledge that span sentences. |
| **BERT knowledge → LLM** | Passed as "Context (Tools detected by JobBERT)" | Anti-hallucination: tools BERT detected anchor the LLM so it does not fabricate unrelated skills or knowledge. Per-job scoping only. |
| **Knowledge output** | LLM-only (default) | BERT knowledge is not fused into final output. Evaluation showed BERT adds noise to knowledge (LLM 0.89 vs BERT+LLM 0.93). |
| **Skills output** | BERT+LLM fusion | Unchanged. Semantic agreement and fusion for skills. |

Config: `LLM_ONLY_KNOWLEDGE = True` (default). Use `--bert-knowledge` for hybrid knowledge fusion.

### 16b. Extraction Weight Justification (Skills)

The pipeline uses several heuristic weights for confidence scoring, fusion,
and recommendation ranking.  These weights are **not learned from data** but
are chosen based on domain reasoning and validated via sensitivity analysis.
This section documents the rationale for each weight group and reports
robustness results.

### A. BERT Confidence Weights

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `BERT_RAW_CONFIDENCE_WEIGHT` | 0.7 | CRF emission probability is the primary signal; dominant weight reflects that NER models are trained end-to-end. |
| `BERT_TYPE_CONFIDENCE_WEIGHT` | 0.3 | Hard/soft type classification provides secondary signal; lower weight because type errors should not dominate. |
| ~~`BERT_BLOOM_FACTOR_BASE`~~ | — | **Removed in pipeline-redesign-v2 Phase 1.3** along with the Bloom classifier. |
| `BERT_DENSITY_FACTOR_BASE` | 0.8 | Semantic density is a secondary quality signal. High floor (0.8) ensures simple-but-valid skills are not penalised excessively. |
| `BERT_STANDALONE_PENALTY` | 0.8 | BERT-only extractions (no LLM confirmation) receive a 20% discount; the penalty is moderate because BERT is a strong baseline. |

**Design principle:** BERT confidence = `(0.7 × raw_conf + 0.3 × type_conf) × density_factor`, where factors have high floors so secondary signals modulate but do not dominate. (The `bloom_factor` term was removed in pipeline-redesign-v2 Phase 1.3 along with the BloomClassifier.)

### B. LLM Confidence Weights

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LLM_BASE_CONFIDENCE` | 0.8 | LLM outputs are generally reliable but not calibrated; 0.8 is a conservative starting point. |
| `LLM_TYPE_FACTOR_BASE` | 0.6 | LLM type classification is less reliable than BERT's; lower floor than BERT. |
| ~~`LLM_BLOOM_FACTOR_BASE`~~ | — | **Removed in pipeline-redesign-v2 Phase 1.3** along with the Bloom classifier. |
| `LLM_DENSITY_FACTOR_BASE` | 0.8 | Same rationale as BERT: density should not dominate. |

### C. Fusion Weights (Skills Only; Knowledge is LLM-Only)

*Note: Knowledge output is LLM-only (Direction A); fusion applies to skills only.*

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `FUSION_MATCH_BONUS_WEIGHT` | 0.2 | When BERT and LLM agree, confidence receives a 20% boost relative to cosine similarity. Conservative to avoid over-inflating. |
| `FUSION_TYPE_DISAGREEMENT` | 0.8 | When BERT and LLM disagree on type (hard vs soft), apply a 20% penalty. Not severe because type disagreement is common for dual-nature skills. |
| ~~`FUSION_BLOOM_CONFIDENCE_SCALE`~~ | — | **Removed in pipeline-redesign-v2 Phase 1.3** along with Bloom-level fusion logic. |
| `AGREEMENT_EXACT_THRESHOLD` | 0.85 | Cosine similarity above 0.85 is treated as exact semantic match. Based on SBERT literature where 0.8+ indicates paraphrase-level similarity. |
| `AGREEMENT_PARTIAL_THRESHOLD` | 0.5 | Below 0.5, skills are considered unrelated. This is the standard SBERT "unrelated" boundary. |

### D. Recommendation Weights

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `w_demand` | 0.40 | Current employer demand is the strongest signal for curriculum relevance. |
| `w_trend` | 0.30 | Growth/decline trend indicates future relevance. Equal weight with future alignment reflects uncertainty in trend extrapolation. |
| `w_future` | 0.30 | Future-domain alignment from WEF/ONET/McKinsey taxonomy. Equal with trend to balance data-driven and expert-driven signals. |

### E. Deduplication Boost

| Formula | Rationale |
|---------|-----------|
| `boosted = max_conf + (1 - max_conf) × log₂(count) / 10` | Logarithmic dampening ensures diminishing returns: 2 mentions → +10% of remaining headroom; 4 mentions → +20%; 8 → +30%. The `/10` divisor was chosen to keep boosts conservative (max ~50% of headroom at count=1024). |

### F. Sensitivity Analysis

The script `scripts/weight_sensitivity_extraction.py` varies each extraction
weight group by ±20% and measures the change in precision on the gold set.
This validates that results are robust to reasonable weight perturbations.

**Interpretation:** If precision changes by <2 percentage points across all
perturbations, the weights are considered robust.  If any single weight
change causes >5pp precision swing, that weight requires empirical
calibration.

Run:

```bash
python scripts/weight_sensitivity_extraction.py
```

For recommendation weight sensitivity, see §9 and:

```bash
python recommendations.py --sensitivity
```

---

## 17. Spektrum Keahlian Integration

**Source:** Keputusan Menteri Pendidikan, Kebudayaan, Riset, dan Teknologi RI Nomor 244/M/2024 tentang Spektrum Keahlian dan Konversi Spektrum Keahlian SMK/MAK pada Kurikulum Merdeka.

### Purpose

The Spektrum Keahlian taxonomy provides an official Indonesian vocational classification (10 Bidang → Program → Konsentrasi). When a department is associated with a Spektrum code (e.g., 4.1.1 Rekayasa Perangkat Lunak), the pipeline filters `future_domains.csv` to only those domains deemed relevant for that vocational field.

### Mapping Logic (future_weight_mapping.py)

1. When `--spektrum-code` is set, load `spektrum_mapping.csv` (default: `data/spektrum_keahlian/` or `DATA/spektrum_keahlian/`).
2. Look up the code in `primary_domain_ids` (semicolon-separated domain_ids).
3. Filter `future_domains.csv` to only those `domain_id`s.
4. If the code is not in the mapping, try `*` (fallback) row.
5. If mapping file is missing or code has no match, use all domains (with a warning).

### Reproducibility

- `log_run_metadata.py` records `spektrum_code`, `future_domains_file`, and `spektrum_mapping_file` in `run_metadata.json`.
- `evaluate_extraction.py` and `evaluate_future_mapping.py` accept `--spektrum-code` and record it in their reports for per-Bidang stratification and provenance.

### Limitations

- Spektrum mapping is manually curated; coverage of non-IT Bidang (e.g., Agribisnis, Pariwisata) may be incomplete.
- `future_domains.csv` is IT/software-biased; applicability to other Bidang without domain expansion is limited.
- Cross-Bidang validity of extraction and mapping metrics is untested when gold sets span multiple Spektrum codes.

---

## Summary

| Component | Key Formula / Method |
|-----------|----------------------|
| Binomial test | H₀: p=0.5, one-sided, scipy.binomtest |
| Odds ratio | p/(1-p) when p₀=0.5 |
| Multi-comparison | Bonferroni (per-source), BH (pairwise) |
| Power analysis | n≥37 for p₁=0.7, α=0.05, power=0.80 |
| Cohen's Kappa | (P_o - P_e)/(1 - P_e), 2 raters |
| Fleiss' Kappa | (P̄ - P_e)/(1 - P_e), 3+ raters |
| Normalization | lowercase, collapse punctuation/spaces |
| Priority score | 0.4×demand + 0.3×trend + 0.3×future |
| Future weight | similarity × trend_score |
| FDR trends | Benjamini-Hochberg, q<0.05 |
| Weight sensitivity | Jaccard vs baseline top-20 |
| Recall | Not estimable (gold set samples outputs, not exhaustive annotations) |
| P@N | (yes + 0.5×partial)/N |
| NDCG | DCG/IDCG with rel ∈ {0, 0.5, 1} |
| Top-1 domain | count(agree)/n_evaluable |
| Domain batching | similarity≥0.45, margin≥0.05 for high conf; merge when cos_sim≥0.7 |
| Extraction weights | Heuristic; documented rationale + ±20% sensitivity (§15) |

---

---

## Scientific Visualization (plot_scientific_analysis.py)

The pipeline generates **scientific plots** that visualize the quantitative analysis:

| Plot | Description |
|------|-------------|
| `scientific_extraction_precision_skills/knowledge.png` | Precision by source with Wilson 95% CI; chance line at 0.5 |
| `scientific_odds_ratio_skills/knowledge.png` | Odds ratio vs chance (1.0) by source |
| `scientific_binomial_test_skills/knowledge.png` | Binomial distribution under H₀ with observed count |
| `scientific_trend_volcano.png` | FDR volcano: slope vs -log₁₀(q); Emerging/Declining/Stable |
| `scientific_trend_regression_samples.png` | Sample regression lines for top emerging/declining skills |
| `scientific_calibration_curve.png` | Reliability diagram (predicted vs observed) |
| `scientific_power_curve.png` | Power vs n for one-sided binomial test |
| `scientific_future_mapping.png` | Top-1 accuracy and mapping margin distribution |
| `scientific_weight_sensitivity.png` | Jaccard vs baseline for weight configurations |

Run: `python plot_scientific_analysis.py` (invoked by run.bat step 17).

**Interpretations:** See [docs/FIGURE_INTERPRETATIONS.md](docs/FIGURE_INTERPRETATIONS.md) for detailed explanations of each figure.

---

*Last updated: Domain-based batching (2025), full scientific methodology with worked examples.*
