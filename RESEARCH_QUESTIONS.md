# Research Questions and Evaluation Framework

This document defines the research questions, success metrics, and evaluation
protocol for the Future-Aware Hybrid Skill Extraction pipeline as a
**curriculum recommendation system** for vocational high schools.

**Related**: [CALCULATIONS.md](CALCULATIONS.md) — formulas; [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) — full scientific methods with worked examples.

---

## Design Intent

The pipeline is a **curriculum gap / reform tool**, not a compliance tool. It surfaces what the job market demands regardless of existing curriculum. This design is intentional: in many contexts (e.g. Indonesia), vocational curricula lag behind labour-market requirements, so prioritising alignment with existing standards would perpetuate outdated curricula. Recommendations prioritise **demand**, **empirical trend**, and **future-domain alignment**; curriculum coverage is used for insights only, not for ranking. The system helps schools identify skills their curriculum lacks and design reforms accordingly.

---

## Research Questions

### RQ1 — Extraction Quality
**Does hybrid extraction (JobBERT + LLM) outperform each component alone on
skill and knowledge correctness?**

*Note: Skills use BERT+LLM fusion; knowledge output is LLM-only (Direction A). BERT knowledge is passed to LLM as anti-hallucination context but not fused into the final list.*

| Metric | Definition | Target |
|--------|-----------|--------|
| Precision | correct extractions / all extractions | > 0.70 |

*Note: Recall and F1 are not estimable with this gold-set design (stratified sample of outputs, not exhaustive corpus annotations). See [SCIENTIFIC_METHODOLOGY.md §10](SCIENTIFIC_METHODOLOGY.md).*

Evaluation: compare **BERT-only**, **LLM-only**, and **Hybrid** on the gold set
(`DATA/labels/gold_skills.csv`, `DATA/labels/gold_knowledge.csv`).

### RQ2 — Scoring Calibration
**Do pipeline scoring signals (confidence, agreement, density) predict human
validity judgments?**

| Metric | Definition | Target |
|--------|-----------|--------|
| AUC-ROC | area under ROC curve for human_valid prediction | > 0.70 |
| Brier Score | mean squared error of calibrated probabilities | < 0.20 |
| Calibration Error | max abs(predicted prob - observed freq) in 10 bins | < 0.15 |

Evaluation: logistic regression on reviewed items; 5-fold cross-validated.

### RQ3 — Trend Detection
**Can we identify statistically robust emerging and declining skills from job
posting time series?**

| Metric | Definition | Target |
|--------|-----------|--------|
| FDR-controlled discoveries | skills with q < 0.05 | report count |
| Stability (Jaccard) | overlap of top-20 emerging across 3+ runs | > 0.60 |
| Sensitivity | consistent labels across min_jobs settings | report |

Evaluation: Benjamini-Hochberg FDR; stability across seeds and min_jobs.

### RQ4 — Future-Domain Mapping
**Does embedding-based domain mapping align with expert judgments?**

| Metric | Definition | Target |
|--------|-----------|--------|
| Top-1 Accuracy | expert agrees with best_future_domain | > 0.60 |
| Top-3 Accuracy | expert domain in top 3 domains | > 0.80 |
| Mapping Margin | mean(top1_sim - top2_sim) | report |

Evaluation: compare against `DATA/labels/gold_future_domain.csv`.

### RQ5 — Recommendation Quality
**Do ranked curriculum recommendations match expert priorities?**

*Expert = curriculum reformers or labour-market-informed experts who judge relevance for **future curriculum design**, not alignment with current standards.*

| Metric | Definition | Target |
|--------|-----------|--------|
| Precision@20 | fraction of top-20 recs rated priority by expert | > 0.60 |
| NDCG@20 | normalized discounted cumulative gain at 20 | > 0.60 |
| Ablation delta | change in P@20 when removing each signal | report |

Evaluation: expert labels top-N=20 recommendations; ablation removes one
signal at a time (demand, trend, future_weight). Coverage is optional (`with_coverage` variant); by default it is not used for prioritization.

---

## Gold Set Design

### Size
- Skills: 150 items (stratified by source, confidence tier, type)
- Knowledge: 100 items (stratified by confidence tier)
- Future-domain mapping: 100 items (skills + knowledge)
- Overlap for IRR: 30 items (configurable via `--overlap_n`), labeled by 2+ reviewers

### Stratification
- Extraction source: BERT / LLM / Hybrid (proportional)
- Confidence tier: Very High / High / Medium / Low (proportional)
- Skill type: Hard / Soft / Both (proportional)

### Power Analysis

- Assumptions: H0 precision = 0.5 (chance), H1 precision = 0.7 (RQ1 target), alpha = 0.05, target power = 0.80
- Required n: 37 items to achieve power >= 0.80 for the one-proportion binomial test
- Current gold set: skills n = 150 yields power ≈ 1.00; knowledge n = 100 yields power ≈ 0.99
- Interpretation: Gold set sizes are well above the minimum required for adequate statistical power

### Labeling Protocol
Each item is labeled with:
- `is_correct`: yes/no (was this correctly extracted from the text?)
- `type_label`: Hard/Soft/Both/Unknown (for skills)
- `bloom_label`: Remember/.../Create/Unknown (for skills)
- `true_domain_id`: best future domain (for mapping validation)

---

## Ablation Study Design

### Extraction ablation
| Variant | Description |
|---------|-------------|
| BERT-only | Items with source=BERT only |
| LLM-only | Items with source=LLM only |
| Hybrid | Items with source=BERT+LLM or Hybrid |

### Recommendation ablation
| Variant | Signals Used |
|---------|-------------|
| Full | demand + trend + future_weight (default; coverage for insights only) |
| No trend | demand + future_weight |
| No future | demand + trend |
| With coverage | demand + trend + future_weight + coverage_gap (optional) |
| Demand only | demand only |

### Stability
- 3-5 runs with different random seeds
- Report mean and std of all metrics
- Jaccard similarity for top-N lists across runs

---

## Evaluation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N (top-N) | 20 | Manageable for expert evaluation |
| FDR threshold | 0.05 | Standard in multiple testing |
| Calibration bins | 10 | Standard for reliability diagrams |
| Cross-validation folds | 5 | Balance between bias and variance |
| Stability runs | 3-5 | Minimum for variance estimation |
| min_jobs sensitivity | [5, 10, 15, 20] | Range around default |

---

## Limitations

- **Scope of applicability**: Domains: IT / Software / Game Dev (current focus); other sectors require domain-specific validation. Geography/language: English job postings; results may not generalize to non-English markets. Use case: curriculum reform and gap identification; not validation against outdated national standards.
- **BERT-only extraction**: BERT-only extraction performs below chance in current evaluation; Hybrid (BERT+GPT) and GPT-only are recommended for production. BERT is retained in the fusion pipeline for potential complementary signal.
- **Recall**: Only estimable from the labeled sample; true recall is unknown without full population labeling. Gold sets are stratified samples of extractions, not exhaustive enumerations of all true items in job postings.
- **Temporal bias**: Job posting dates may cluster in the scrape window; trend analyses reflect the available time range.
- **Domain coverage**: `future_domains.csv` may not cover all vocational fields; mapping accuracy is limited to included domains.
- **Generalizability**: Pipeline tuned for IT/Software/Game Dev context; results may not transfer to other sectors.
- **LLM variability**: Competency generation is non-deterministic when temperature > 0; use `--temperature 0` for reproducibility.
- **JobBERT domain**: JobBERT is trained primarily on (Western) job descriptions; generalization to Indonesian/SMK domains is untested.
- **Spektrum coverage**: The `spektrum_mapping.csv` to `future_domains.csv` mapping is manually curated; non-IT Bidang (e.g., Agribisnis, Pariwisata) may have incomplete or fallback mappings.
- **Cross-Bidang validity**: Gold-set representativeness and metric validity when evaluating across multiple Spektrum codes (Bidang) are not yet established.
