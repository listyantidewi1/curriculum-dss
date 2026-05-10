# Research Questions and Evaluation Framework

This document defines the research questions, success metrics, and evaluation
protocol for the Future-Aware Competency Recommendation System for vocational
high schools. After the 2026 reframe the system's deliverable is **hard-skill
competency statements with attached soft-skill requirements**, not raw skill
lists. Skill/knowledge extraction is the infrastructure that feeds the
competency layer.

**Related**: [CALCULATIONS.md](CALCULATIONS.md) — formulas; [SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md) — full scientific methods with worked examples.

---

## Design Intent

The pipeline is a **curriculum gap / reform tool**, not a compliance tool. It surfaces what the job market demands regardless of existing curriculum. This design is intentional: in many contexts (e.g. Indonesia), vocational curricula lag behind labour-market requirements, so prioritising alignment with existing standards would perpetuate outdated curricula. Recommendations prioritise **demand**, **empirical trend**, and **future-domain alignment**; curriculum coverage is used for insights only, not for ranking. The system helps schools identify skills their curriculum lacks and design reforms accordingly.

---

## Research Questions

### RQ1 — Extraction Quality (LLM-only primary; hybrid as ablation)
**Does LLM-only extraction produce sufficient quality for downstream competency
recommendation, and does adding BERT (hybrid) materially improve precision?**

After the 2026 reframe, **LLM-only is the primary extraction mode**; hybrid is
the ablation. The question becomes whether BERT contributes a measurable
precision lift over LLM-only — if not, the simpler LLM-only path stands.

| Metric | Definition | Target |
|--------|-----------|--------|
| Precision (LLM-only) | correct extractions / all extractions | > 0.70 |
| Hybrid lift | precision(hybrid) − precision(LLM-only) | reported (effect size + Wilson CI) |

*Note: Recall and F1 are not estimable with this gold-set design (stratified sample of outputs, not exhaustive corpus annotations). See [SCIENTIFIC_METHODOLOGY.md §10](SCIENTIFIC_METHODOLOGY.md).*

Evaluation: compare **LLM-only** (primary) and **Hybrid** (ablation) on the
gold set (`DATA/labels/gold_skills.csv`, `DATA/labels/gold_knowledge.csv`).
The hybrid run is performed in `results/hybrid/` (`run.bat` step 2);
`evaluate_extraction.py --llmonly-labels-dir results/hybrid/DATA/labels`
performs the comparison.

### RQ1b — Competency Generation Quality
**Do generated competencies meet curriculum-design quality bars, given the
hard-skills-only input and the soft-skill enrichment?**

| Metric | Definition | Target |
|--------|-----------|--------|
| Total competency count | post-dedup count for the corpus | report (target: materially lower than legacy 8–20-per-batch baseline) |
| Per-batch count | mean / max competencies per batch | mean ≤ 8, max ≤ 12 (hard cap) |
| Cohesion | mean within-batch SBERT pairwise similarity of `related_skills` | report (higher is better; baseline is the legacy "Uncertain"-includes-everything run) |
| Soft-skill grounding | fraction of `soft_skills_required` items found in extracted Soft-typed skills | report |
| Expert quality (1–5) | mean human_quality from review UI | ≥ 3.5 |
| Expert relevance | fraction marked human_relevant=yes | ≥ 0.80 |

Evaluation: `evaluate_competency_generation.py` on `competency_assessments.json`
plus a cohesion script over the per-batch SBERT embeddings of `related_skills`.

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
| Direction accuracy (holdout) | % correct Emerging/Declining/Stable in held-out months | > 0.60 |
| Slope correlation (holdout) | Pearson r between train-period slope and test-period slope | > 0.50 |

Evaluation: Benjamini-Hochberg FDR; stability across seeds and min_jobs; **longitudinal holdout** (`skill_trend_holdout_validation.py`) splits data into train/test by month (last N months held out), trains FDR model on train portion, measures predictive accuracy on test months.

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
- ~~`bloom_label`~~: removed in pipeline-redesign-v2 Phase 1.3 (Req 1); Bloom-level decisions are returned to curriculum stakeholders
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
| Demand only | demand only (demand-only floor baseline) |

### Coverage ablation
Tests the sensitivity of the top-20 recommendation list to non-zero coverage weights:

| w_coverage | Weight rebalancing |
|------------|-------------------|
| 0.0 | demand 0.40, trend 0.30, future 0.30 (default) |
| 0.10 | demand, trend, future each reduced proportionally |
| 0.20 | demand, trend, future each reduced proportionally |
| 0.30 | demand, trend, future each reduced proportionally |

Metric: Jaccard overlap vs no-coverage baseline. If Jaccard > 0.80 for all w_coverage values, coverage weight design is empirically justified. Results saved to `coverage_ablation_report.json`.

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
- **Skill normalization threshold**: SBERT cosine similarity threshold (0.82) for canonical skill grouping is not empirically validated; too aggressive may merge distinct skills.
- **Holdout validity**: Longitudinal holdout assumes non-seasonal trends; seasonal job-posting patterns could inflate or deflate direction accuracy.
