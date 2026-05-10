# 🎓 Future-Aware Competency Recommendation System

### **Curriculum design support for vocational schools (IT / Software / Game Development)**

This repository is a **competency recommendation system**: given a corpus of real
job postings, it produces a ranked list of **hard-skill competency statements**
(with soft-skill requirements attached per competency) that a school can use to
update its curriculum. Skill and knowledge extraction is the infrastructure;
**competency recommendations are the product**.

* Extract **hard skills** and **knowledge** from job postings (LLM-first; BERT retained for ablation)
* Map skills to **future-of-work domains** (WEF / O\*NET / McKinsey, SBERT cosine)
* Detect **emerging / declining** trends with FDR control
* Generate **hard-skill competencies** with **per-competency soft-skill requirements** (list + description)
* Rank **curriculum-gap recommendations** by demand × empirical trend × future relevance
* Support **expert review** via a multi-reviewer web UI
* **Normalize and deduplicate** extracted skills with SBERT semantic clustering
* **Longitudinal holdout validation** for trend detection (RQ3)
* **Coverage ablation testing** to empirically validate recommendation weights
* **Printable PDF-quality reports** with explainability scores from the dashboard

The system is modular, reproducible, and supports multi-run **experimental aggregation**.

---

# 📍 Project Status — pipeline-redesign-v2

The repository is mid-redesign. A mentor-driven architectural overhaul
(`pipeline-redesign-v2`) is reshaping the extraction stack to remove Bloom
classification from the pipeline, add full provenance traceability, replace
domain-based batching with principled clustering, and switch the BERT extractor
for an LLM-based one. Authoritative spec:
[`.kiro/specs/pipeline-redesign-v2/requirements.md`](.kiro/specs/pipeline-redesign-v2/requirements.md).

### Phase 1 — Data layer (current)

| Sub-phase | Work | Status |
|---|---|---|
| **1.1** | Sentence-level provenance throughout (every extracted item carries `sentence_id`, `sentence_text`, `extractor_source`) | ✅ Done |
| **1.2** | Zero-shot LLM sentence relevance filter (drops boilerplate / benefits / logistics) with SHA-256 cache | ✅ Done |
| **1.3** | Bloom taxonomy removed from the pipeline (decisions returned to curriculum stakeholders) | ✅ Done |
| **1.4** | JobBERT replicate (`baseline_versions/jjzha_replicate/`) — confirmed published checkpoint scores comparable to vanilla BERT; targets revised | ✅ Done |
| **1.5** | Skill-LLM LoRA fine-tune of LLaMA 3.1 8B (`baseline_versions/skill_llm/`) — new BERT-replacement, paper-spec recipe | 🔄 In progress (training on Kaggle) |

### Phase 2 — Pipeline reflow (queued)

After 1.5 lands, Phase 2 will replace domain-based batching with HDBSCAN +
agglomerative clustering, rewrite the competency generator to be cluster-driven
with full provenance, add a post-hoc SBERT-based KKNI labeler (Perpres 8/2012
levels 1–9), wire in education-level aggregation per competency, and add a
competency evaluator (grounding / coherence / coverage). Public UI gets a
"Why this competency?" provenance chain at the same time.

### Onboarding for contributors

- **[docs/RINGKASAN_KONTRIBUTOR.md](docs/RINGKASAN_KONTRIBUTOR.md)** —
  Indonesian onboarding doc, 8 sections, written for a developer joining the
  project. Read this first if you're new.
- **[docs/PIPELINE_DIAGRAM.md](docs/PIPELINE_DIAGRAM.md)** —
  the planned v2 architecture as Mermaid (renders inline on GitHub) plus an
  AI-image-generator prompt for slides / paper figures.

---

# 📖 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** (this file) | Overview, quick start, high-level workflow |
| **[PIPELINE.md](PIPELINE.md)** | Detailed pipeline documentation: phases, data flow, file dependencies, troubleshooting |
| **[docs/PIPELINE_DIAGRAM.md](docs/PIPELINE_DIAGRAM.md)** | Planned v2 architecture diagram (Mermaid + AI-image prompt) |
| **[docs/RINGKASAN_KONTRIBUTOR.md](docs/RINGKASAN_KONTRIBUTOR.md)** | Indonesian contributor onboarding (project goal, architecture, entry points, conventions) |
| **[docs/PENJELASAN_UMUM.md](docs/PENJELASAN_UMUM.md)** | Indonesian general-public explainer (non-technical, for educators / policymakers) |
| **[docs/KAJIAN_AKADEMIK.md](docs/KAJIAN_AKADEMIK.md)** | Indonesian academic reflection for promotor discussion |
| **[docs/PUBLIC_UI.md](docs/PUBLIC_UI.md)** | Public-surface architecture (audiences, URL map, publish flow, caching) |
| **[CALCULATIONS.md](CALCULATIONS.md)** | Scientific formulas: ranking, voting, weighting, priority scores, FDR, evaluation metrics |
| **[SCIENTIFIC_METHODOLOGY.md](SCIENTIFIC_METHODOLOGY.md)** | **Full scientific documentation**: all statistical methods, formulas, and worked examples (binomial test, effect sizes, Kappa, power analysis, FDR, normalization, etc.) |
| **[RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md)** | Research questions (RQ1–RQ5), evaluation metrics, gold set design, ablation study |
| **[baseline_versions/jjzha_replicate/REPLICATION_REPORT.md](baseline_versions/jjzha_replicate/REPLICATION_REPORT.md)** | Phase 1.4 close-out: F1 matrix, why we pivot to Skill-LLM |
| **[baseline_versions/skill_llm/README.md](baseline_versions/skill_llm/README.md)** | Phase 1.5: LoRA fine-tune setup, run book, expectations |
| **[docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** | Central index of all documentation |

---

# 🔍 Repository Structure

```
skill-extraction/
│
├── pipeline.py                      # Main hybrid extraction pipeline (sentence-level provenance per Phase 1.1)
├── config.py                        # Global configuration
├── plot_generator.py                # Visual analytics (Bloom plots removed in v2; stubs preserved)
├── verify_skills.py                 # Skill verification (calibrated or percentile)
├── sentence_relevance_filter.py     # Phase 1.2: zero-shot LLM sentence relevance filter (SHA-256 cached)
├── generate_competencies.py         # Future-aware competency generator (LLM, domain-based batching today; cluster-driven in Phase 2.1)
├── domain_batching.py               # Domain-based batching (legacy; Phase 2.1 replaces with HDBSCAN+agglomerative)
├── kkni.py                          # KKNI level descriptors (Perpres 8/2012); Phase 2.3 SBERT labeler will live next to it
├── recommendations.py               # Ranked curriculum recommendations + ablation
├── enrich_with_dates.py             # Attach job_date → extraction outputs
├── skill_time_trend_analysis.py     # FDR-controlled time-series trends + stability
├── future_weight_mapping.py         # Map skills/knowledge → future domains (with margin)
├── ingest_future_domains.py         # Normalize WEF/O*NET/McKinsey → future_domains.csv
│
├── skill_normalizer.py              # SBERT skill normalization (canonical forms via greedy clustering)
├── skill_trend_holdout_validation.py # Longitudinal holdout validation for RQ3
│
├── export_for_review.py             # Human-in-the-loop review tables
├── export_competencies_for_review.py # Competency review export
├── export_gold_set.py               # Stratified gold set for labeling
├── export_recommendations_for_review.py # Top-20 recs for expert priority labeling (RQ5 IRR)
├── import_feedback.py               # Merge feedback_store → feedback artifacts
├── apply_feedback.py                # Apply type corrections (Bloom corrections removed in v2)
│
├── evaluate_extraction.py           # Precision per extractor (BERT/LLM/Hybrid)
├── validate_parameters.py           # AUC, Brier, calibration, cross-validated AUC
├── evaluate_competency_generation.py # Competency quality metrics
├── evaluate_future_mapping.py       # Domain mapping accuracy vs gold labels
├── log_run_metadata.py              # Record run metadata + LLM prompt versioning (SHA-256)
├── aggregate_results.py             # Aggregate runs + cross-run summary
├── preprocess_jobs_pipeline.py      # Raw jobs → jobs_sentences.csv, jobs_metadata.csv (with dedup)
├── run_with_job_scraping.py        # One-step: preprocess + pipeline using job_scraping data
├── pipeline_orchestrator.py         # Dashboard: department-scoped pipeline with checkpoint/resume
│
├── review_ui/                       # Web UI for internal/development review
│   ├── app.py                       # FastAPI backend
│   ├── static/app.js                # Frontend logic
│   └── templates/index.html
│
├── gold_labeling_ui/                # Web UI for gold-set labeling (multi-reviewer)
│   ├── app.py                       # FastAPI backend
│   ├── static/app.js, style.css     # Frontend
│   └── templates/index.html         # Skills, Knowledge, Domain tabs
│
├── merge_gold_labels.py             # Majority-vote merge of multi-reviewer labels
│
├── dashboard/                       # Admin + school dashboard (production)
│   ├── app.py                       # FastAPI app, school review, results
│   ├── db.py                        # SQLite (schools, departments, users, runs)
│   ├── templates/                   # Jinja2 (admin, school)
│   └── static/style.css
│
├── feedback_store/                  # Per-reviewer feedback (default run)
│   ├── skill_feedback.csv
│   ├── knowledge_feedback.csv
│   └── competency_feedback.csv
│
├── DATA/
│   ├── labels/                      # Gold set for evaluation
│   │   ├── gold_skills.csv
│   │   ├── gold_knowledge.csv
│   │   ├── gold_future_domain.csv
│   │   └── gold_labels/             # Multi-reviewer labels (from gold_labeling_ui)
│   ├── samples/                     # Sample CSVs for dashboard simulation
│   │   ├── jobs_sample.csv
│   │   ├── curriculum_sample.csv
│   │   └── README.md
│   └── preprocessing/data_prepared/
│       ├── jobs_sentences.csv       # Pipeline input (from preprocess on job_scraping/output/english_jobs.csv)
│       └── jobs_metadata.csv        # job_id → job_date
│
├── job_scraping/                    # Job scrapers (default data source)
│   ├── scrape_english_jobs.py      # → output/english_jobs.csv
│   ├── scrape_indonesian_jobs.py   # → output/indonesian_jobs.csv (optional)
│   └── output/
│       ├── english_jobs.csv         # 12 months of job postings (English)
│       └── indonesian_jobs.csv      # Optional: Indonesian-language job postings
│
├── scripts/
│   ├── create_sample_csvs.py        # Generate DATA/samples/*.csv for dashboard
│   └── weight_sensitivity_extraction.py # Extraction weight sensitivity analysis
│
├── results/                         # Output of a single run
├── results_aggregated/              # Aggregated results across runs
│
├── baseline_versions/               # Research-track BERT/LLM extractor variants (mostly gitignored — weights too large)
│   ├── jobbert_crf/                 # Original multitask + CRF baseline (legacy)
│   ├── v3_stl/                     # Single-task + CRF baseline (legacy)
│   ├── jjzha_replicate/             # Phase 1.4: replication of jjzha/jobbert_skill_extraction (audit + report tracked)
│   └── skill_llm/                   # Phase 1.5: LoRA fine-tune of LLaMA 3.1 8B (paper-spec) + Kaggle training script
│
├── RESEARCH_QUESTIONS.md            # RQs, metrics, ablation design
├── CALCULATIONS.md                  # Ranking, voting, weighting formulas
├── PIPELINE.md                      # Detailed pipeline documentation
├── run.bat                          # Phase 1: Full pipeline (18 steps)
└── run_phase_2.bat                  # Phase 2: Post-review pipeline (18 steps)
```

---

# ⚡ Quick Start

```bat
REM 0. (One-time) Generate real future domains from WEF/O*NET/McKinsey
python ingest_future_domains.py

REM 0b. (Optional) Scrape fresh job data: cd job_scraping && python scrape_english_jobs.py
REM     Default pipeline uses job_scraping/output/english_jobs.csv

REM 1a. Quick run with job_scraping data (preprocess + pipeline in one step, with dedup):
python run_with_job_scraping.py --dedupe

REM 1b. Or Phase 1 — Full pipeline (18 steps: extraction → trends → recommendations → gold set)
run.bat
REM   run.bat includes deduplication (--dedupe) and LLM-only ablation run automatically

REM 2. (Optional) Label gold set: uvicorn gold_labeling_ui.app:app --reload
REM    Then python merge_gold_labels.py — see DATA/labels/LABELING_PROTOCOL.md

REM 3. Expert review — start the web UI (optional but recommended)
uvicorn review_ui.app:app --reload
REM Open http://127.0.0.1:8000/?reviewer_id=alice

REM 4. Phase 2 — Post-review (18 steps: calibration → re-generation → holdout validation → evaluation)
run_phase_2.bat

REM 5. Label top-20 recommendations (results/recommendations.csv → expert_priority column)
python recommendations.py --evaluate

REM 6. (Optional) Dashboard: upload DATA/samples/*.csv, run department pipeline
REM    Includes: printable report, sparklines, score explainability
uvicorn dashboard.app:app --reload

REM 7. (Optional) Multi-run: rename results → results_run1, repeat, then aggregate
python aggregate_results.py --run_dirs results_run1 results_run2 --output_dir results_aggregated
REM 8. (Optional) Generate all plots from aggregated data:
REM    plot_aggregated.bat
REM    or: python aggregate_results.py ... --plot
```

**Larger data:** Edit `run.bat` or run `python run_with_job_scraping.py --sample_size 5000 --dedupe` (or `--sample_size 0` for no limit).

**Indonesian postings:** `python run_with_job_scraping.py --include-indonesian --dedupe` — merges `indonesian_jobs.csv` with English data, auto-enables translation.

**Resume interrupted run:** `pipeline_orchestrator.py` supports `resume=True` — skips steps whose output files already exist.

See [PIPELINE.md](PIPELINE.md) for detailed steps, data flow, and troubleshooting.
See [CALCULATIONS.md](CALCULATIONS.md) for ranking, voting, weighting, and evaluation formulas.
See [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) for evaluation framework and metrics.

---

# 🖥 Dashboard — Public Surface + Admin

After the 2026 reframe to a **competency recommendation system**, the dashboard
exposes two distinct surfaces:

### Public surface (`/`) — anonymous, KKNI-aligned

- **Landing** at `/` with stage chips (SMA / SMK / D3 / S1 / S2 / S3) mapped to KKNI levels
- **Browse** at `/competencies` — filter by stage, KKNI level, future domain, or full-text
- **Detail** at `/competencies/{id}` — full description, related hard skills, soft-skill profile, education-demand chart
- **Coverage** at `/coverage` — upload a curriculum CSV/JSON, get an instant gap report
- **About** at `/about` — methodology + KKNI explainer
- **Light signup** at `/signup` — saves curriculum analyses (no school upload, no pipeline ops)

### Admin surface (`/dashboard/admin/*`) — locked to data ops

- **Schools / Departments / Users / Runs** management (Bidang locked to "Teknologi Informasi")
- **Publish** at `/dashboard/admin/publish` — snapshot a `results/` directory as the canonical run that powers the public site
- **Inter-rater reliability** snapshot

### Run

```bat
uvicorn dashboard.app:app --reload
```

- Public site: `http://127.0.0.1:8000/`
- Admin login: `http://127.0.0.1:8000/dashboard/login` — default admin: `admin@local` / `admin123`

See **[docs/PUBLIC_UI.md](docs/PUBLIC_UI.md)** for the full URL map and publish flow.

---

# 🔬 Review UI (Internal / Development)

For **internal or development** reviews (no auth, single output dir):

```bat
uvicorn review_ui.app:app --reload
```

Open `http://127.0.0.1:8000/?reviewer_id=alice` — feedback goes to `feedback_store/` (project root).

| Aspect | review_ui | Dashboard |
|--------|-----------|-----------|
| Purpose | Internal / dev | Production (schools) |
| Auth | None (URL param) | Login required |
| Reviewer ID | `?reviewer_id=` | Logged-in user email |
| Data | Default `results/`, `feedback_store/` | Per-department `data/schools/.../` |

**Notes:**
- Dashboard runs are isolated under `data/schools/{school_id}/departments/{department_id}/`
- Existing `run.bat` and `run_phase_2.bat` workflows remain unchanged

---

# 🚀 System Overview

This project is a **competency recommendation system** for curriculum design.
Skill/knowledge extraction is the infrastructure that feeds two products:

1. **Ranked curriculum-gap recommendations** (top-20 hard skills the curriculum
   should add, prioritized by demand × empirical trend × future relevance).
2. **Hard-skill competency statements** with, per competency, a list and
   description of the soft skills a learner needs to perform it.

LLM-first extraction: a single LLM call per job extracts skills (verb-led
action phrases), knowledge (noun phrases — tools, technologies, concepts), and
classifies skill type (Hard/Soft). Domain is assigned post-extraction via SBERT
cosine similarity to a curated future-domain taxonomy (`future_domains.csv`).
BERT is retained as an optional ablation (`--extraction-mode hybrid`) but is no
longer the default. Bloom-level classification was removed from the pipeline
in pipeline-redesign-v2 Phase 1.3 (Bloom decisions are now left to curriculum
stakeholders).

## **Main Stages**

### **1. Data Acquisition & Cleaning**

* **Default source:** `job_scraping/output/english_jobs.csv` (12 months of job postings; config.JOBS_SCRAPING_CSV)
* **Optional Indonesian source:** `job_scraping/output/indonesian_jobs.csv` — merged via `--include-indonesian` flag; auto-translated + deduplicated
* Scraped job postings (IT / Software / Game Development)
* **Job deduplication** (`--dedupe`): MD5 fingerprint of (title + company + first 500 chars of description); removes near-duplicates before sentence splitting
* Cleaning markdown noise (** \ // etc.)
* Sentence splitting (bullet boundaries, paragraph breaks for JobBERT 128-token limit)
* Every sentence carries **job_id + job_date**

### **2. LLM-first Extraction (BERT optional)**

**Default extraction mode after the 2026 reframe:** `llm_only` — a single LLM
call per job description extracts skills (verb-led action phrases) and
knowledge (noun phrases) and classifies skill type (Hard / Soft). BERT is
retained for ablation (`--extraction-mode hybrid`).

* **Verb-noun discrimination is load-bearing.** "designing UI/UX" is a SKILL;
  "UI/UX" alone is KNOWLEDGE. The pipeline preserves this distinction
  end-to-end because downstream stages (competency generation, KKNI labeler)
  rely on the verb as a signal of cognitive level. Single-word soft skills
  ("passion", "self-starter") are also valid SKILL items per SkillSpan
  annotation conventions.
* **Sentence-level provenance.** Every extracted item carries `(job_id,
  sentence_id, sentence_text, extractor_source)` so any output can be traced
  back to source. (Phase 1.1 of pipeline-redesign-v2.)
* **Sentence relevance filter.** Before extraction, a zero-shot LLM filter
  drops irrelevant sentences (benefits, location, boilerplate). Persistent
  SHA-256 cache makes re-runs cost zero. (Phase 1.2.)
* **LLM-based extractor** for structured JSON (configurable: DeepSeek, GPT, Gemini, Claude, etc.)
* **JobBERT + CRF** preserved as an ablation path (`--extraction-mode hybrid` or `bert_only`); a Skill-LLM (LoRA-fine-tuned LLaMA 3.1 8B) replacement is in flight (Phase 1.5).
* **Semantic agreement** using SBERT embeddings (used in fusion when hybrid mode is active)
* **Skill normalization** (`skill_normalizer.py`) — SBERT greedy clustering (threshold 0.82); canonical = most frequent variant, tiebreak = shortest string
* **Prompt versioning** — All LLM prompts are SHA-256 fingerprinted at runtime; combined hash stored in `run_metadata.json` for reproducibility

### **3. Skill / Knowledge Taxonomy**

* **Skills** — verb-led action phrases (Hard) or single-word personality / behavioral terms (Soft)
* **Knowledge** — noun phrases representing tools, technologies, frameworks, theoretical concepts
* **Semantic density scoring** for confidence weighting
* Bloom-level classification was removed in Phase 1.3 (Req 1) — Bloom decisions belong with curriculum stakeholders, not the pipeline

### **4. Curriculum Mapping**

* Compare skills/knowledge with **SMK Software & Game Dev curriculum**
* Component mapping via SBERT
* Compute:

  * coverage percentage
  * component-level heatmaps

### **5. Future-of-Work Integration**

* Normalizes real forecast sources (WEF, O*NET, McKinsey) via `ingest_future_domains.py`
* Maps skills/knowledge to domains using SBERT cosine similarity
* Computes:

  ```
  future_weight = similarity(skill, domain) × trend_score
  ```
* Includes **mapping uncertainty** (top1-top2 similarity margin)
* Identifies future-critical skills, declining skills, and curriculum gaps

### **6. Time Trend Analysis**

* FDR-controlled (Benjamini-Hochberg) emerging/declining skill detection
* Outputs q-values (not just raw p-values) to control false discovery rate
* Stability analysis across multiple seeds and min_jobs thresholds
* **Longitudinal holdout validation** (`skill_trend_holdout_validation.py`) — Holds out last N months, trains FDR trend model on earlier data, measures direction accuracy and slope correlation on held-out period (RQ3)

### **7. Competency Generator (LLM)**

* Uses verified skills + future context + empirical trend signals
* **Domain-based batching:** Groups skills by future domain before LLM calls, so each batch contains thematically related skills (reduces forced groupings)
* Fallbacks: normalized-key lookup for coverage gaps; on-the-fly embedding lookup for unmapped skills; "Uncertain" batch for low-confidence domain assignments
* Produces competency IDs, titles, descriptions, related skills, future relevance notes

### **8. Curriculum Recommendations**

* Ranked skill gap priorities combining: **demand, empirical trend, future_weight** (coverage is for insights only, not prioritization)
* Schools use the system to design better curriculum; existing curriculum may be outdated
* Evidence traces per recommendation (job_ids, trend stats, domain info)
* Ablation study (remove one signal at a time; optional `with_coverage` variant)
* **Coverage ablation** (`--coverage-ablation`) — Sweeps w_coverage = 0.0/0.10/0.20/0.30; reports Jaccard overlap vs no-coverage baseline to empirically validate the 0.0 default
* Expert evaluation: Precision@20, NDCG@20

### **9. Export for Review & Human-in-the-Loop**

Creates sampled CSVs for expert validation (500 skills, 200 knowledge, 100 competencies).

**Review workflow (single or multi-reviewer):**
1. Phase 1 (`run.bat`) exports review tables and gold-set labels automatically
2. Start the review web app: `uvicorn review_ui.app:app --reload`
3. **Multi-reviewer:** Each reviewer opens `http://localhost:8000/?reviewer_id=alice`
4. Review in browser; feedback auto-saves to `feedback_store/`
5. Phase 2 (`run_phase_2.bat`) imports feedback, calibrates scoring, re-generates, and evaluates

### **10. Scientific Evaluation**

* **Gold set labeling** (`DATA/labels/`) for ground-truth extraction quality
* **Extraction evaluation**: Precision per source (BERT/LLM/Hybrid) with Wilson CIs (recall not estimable)
* **Calibrated verification**: AUC-ROC, Brier score, calibration curve, cross-validated AUC
* **Domain mapping validation**: Top-1 accuracy vs expert labels
* **Longitudinal holdout validation**: Direction accuracy and slope correlation on held-out trend data (RQ3)
* **Coverage ablation**: Empirical justification for coverage weight = 0.0 default
* **Reproducibility**: Run metadata with dataset hash, model versions, seeds, and LLM prompt hashes

---

# 🧪 Experimental Workflow

The system supports **multiple independent runs** for robust evaluation.

### **1. Run an experiment (e.g., sample size = 1000)**

```bat
run.bat
```

For **larger data**: `python run_with_job_scraping.py --sample_size 5000` (or `--sample_size 0` for no limit).

After the run completes, rename the results folder:

```
results → results_run1
```

Repeat:

```
results_run2
results_run3
...
```

### **2. Aggregate runs**

```bash
python aggregate_results.py --run_dirs results_run1 results_run2 results_run3 --output_dir results_aggregated
```

### **3. Generate final plots, competencies, and review tables**

Set `OUTPUT_DIR = "results_aggregated"` in `config.py`
Then run:

```bash
python plot_generator.py
python future_weight_mapping.py
python verify_skills.py
python generate_competencies.py
python export_for_review.py
python skill_time_trend_analysis.py --only_hard
```

---

# 📊 Visualizations & Analytics

The system generates:

### **Hybrid model comparison**

* JobBERT vs LLM vs Hybrid
* Skill/Knowledge counts
* Confidence score distributions

### **Top-N clusters**

* Hard skills
* Knowledge items
* Soft skills
* Skills demanded but **not covered** by curriculum (insight)
* Skills "future-critical" but underrepresented

### **Time trend analysis**

* Emerging vs declining skills
* Based on `job_date`
* Monthly trend slopes

### **Future-of-work analytics**

* future_weight histogram
* Top future-weighted skills and knowledge
* Emerging skills coverage (covered vs not covered; insight only)

### **Coverage (insight only)**

* Coverage distribution across jobs
* Coverage improvement: Hybrid vs base models
* Coverage is not used for prioritization; schools use recommendations to design better curriculum.

---

# 🧠 Competency Generation (the product)

`generate_competencies.py` produces hard-skill competency statements (with
anti-hallucination: prompt rule + post-validation filter for `related_skills`):

* JSON competency framework
* **Hard-skill input only** (default after 2026 reframe). Soft skills surface
  per-competency via `soft_skills_required` and `soft_skills_description`.
  Pass `--include-soft-in-competencies` to use the legacy mixed-input behavior.
* **At most 12 competencies per batch** (hard cap; LLM is told to coalesce
  overlapping themes rather than split). Constant: `COMPETENCIES_PER_BATCH_CAP`.
* **Domain-based batching (default):** Skills are grouped by `best_future_domain` before chunking. Low-confidence mappings (similarity below 0.45 or mapping_margin below 0.05) go to "Uncertain"; unmapped skills get on-the-fly domain assignment or "Unmapped". The "Uncertain" / "Unmapped" pseudo-domains are **sub-clustered with SBERT** so each LLM call sees a thematically coherent group instead of a mixed bag. Small domain batches can be merged when domains are strongly similar (cosine ≥ 0.7). Use `--no-batch-by-domain` for legacy sequential chunking.
* **Three-stage deduplication:** (1) normalized-title exact match, (2) SBERT
  semantic title similarity ≥ 0.85 AND related-skill Jaccard ≥ 0.40,
  (3) optional related-skill Jaccard merge (`--merge-overlap-threshold`).
  Soft-skill fields are absorbed (set union for the list, first non-empty
  for the description).
* Each competency includes:

  * `id`, `title`, `description` (single verb-led sentence — measurable, operational learning outcome)
  * `related_skills` (hard skills only, drawn from input)
  * `future_relevance` statement
  * `soft_skills_required` — list of 3–6 soft skills a learner needs
  * `soft_skills_description` — single sentence describing how those soft skills
    support the competency (rendered verbatim in the school report)
  * `batch_domain` (when domain batching is enabled)

This output can be directly used in a **curriculum redesign document** or **expert workshop**.

---

# 📝 Pipeline Diagrams

The planned v2 architecture is in **[docs/PIPELINE_DIAGRAM.md](docs/PIPELINE_DIAGRAM.md)**
in two formats:

* **Mermaid source** — renders inline on GitHub / Notion / draw.io. Use when
  you need a precise, version-controlled, editable diagram.
* **Natural-language prompt** — paste into ChatGPT image / DALL-E / Midjourney
  / Whimsical AI / Eraser.io for marketing-style figures (slides, paper).

For the legacy file-by-file flow (which files produce which outputs), see
**[PIPELINE.md](PIPELINE.md)**.

---

# 💼 Roadmap — Phase 2 of pipeline-redesign-v2

After Phase 1.5 (Skill-LLM LoRA fine-tune) lands, the queued Phase 2 work
reshapes the generation flow on top of the new data layer:

| Sub-phase | Work | Spec |
|---|---|---|
| **2.1** | Replace `domain_batching.py` with HDBSCAN + agglomerative clustering, winner per batch by SBERT cohesion. New file: `skill_clustering.py`. | Req 5 |
| **2.2** | Rewrite `generate_competencies.py` to be cluster-driven and provenance-aware. Each competency carries `contributing_item_ids`, `source_job_ids`, `source_sentences`. | Req 6 |
| **2.3** | New `kkni_labeler.py` — post-hoc SBERT match competency to KKNI levels 1–9 (Perpres 8/2012). Informational only, does not enter ranking. | Req 2 |
| **2.4** | Aggregate education-level histogram per competency from contributing job postings. | Req 2 |
| **2.5** | New `evaluate_competency_quality.py` — grounding score, coherence score, coverage score; flags hallucinated competencies. | Req 7 |
| **2.6** | Public detail page gets a "Why this competency?" provenance chain; browse filters add KKNI level + education stage. | Req 8 |

Longer-horizon ideas (post-Phase 2):

* Train a **domain-specific SBERT** for stronger skill-domain matching
* **Semantic search** over generated competencies (for the public dashboard)
* Scaling tests with >10,000 job postings
* Incorporate additional national / regional forecast sources

---

# 🙌 Citation & Acknowledgment

If you use this pipeline or insights from this project, please cite:

```
[Astuti, Listyanti Dewi]. Future-aware hybrid skill extraction for curriculum intelligence. (2025)
```

---

# 📬 Contact

For questions or collaboration:

* **Author:** Listyanti Dewi Astuti
* **Affiliation:** SMK Negeri 12 Malang / Universitas Negeri Malang
* **Email:** [your email here]

---
