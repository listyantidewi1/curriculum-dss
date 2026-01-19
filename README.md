# 🎓 Future-Aware Hybrid Skill Extraction Pipeline

### **A Curriculum-Intelligence System for Vocational Education (IT / Software / Game Development)**

This repository contains a full **research AI pipeline** designed to:

* Extract **skills** and **knowledge** from job postings
* Fuse outputs from **JobBERT + LLMs**
* Map results to **educational taxonomies**
* Evaluate **curriculum coverage**
* Integrate **future-of-job forecasts** (WEF / McKinsey style)
* Generate **competency statements** for curriculum development
* Support **expert review** through structured exports

The system is modular, reproducible, and supports multi-run **experimental aggregation**.

---

# 🔍 Repository Structure

```
skill-extraction/
│
├── pipeline.py                      # Main hybrid extraction pipeline
├── config.py                        # Global configuration
├── plot_generator.py                # Visual analytics
├── verify_skills.py                 # Skill verification tiers
├── generate_competencies.py         # Future-aware competency generator (LLM)
├── export_for_review.py             # Human-in-the-loop review tables
├── enrich_with_dates.py             # Attach job_date → extraction outputs
├── skill_time_trend_analysis.py     # Time-series trend analysis
├── future_weight_mapping.py         # Maps knowledge → future domains
├── aggregate_results.py             # Aggregates multiple experiment runs
│
├── data/
│   ├── preprocessing/
│   │   ├── jobs_english_only_dataset.csv
│   │   ├── job_sentences_with_dates_clean.csv
│   │   ├── jobs_metadata.csv
│   │   └── ...
│   └── future_domains_dummy.csv     # Dummy WEF/McKinsey-style dataset
│
├── results/                         # Output of a single run
├── results_run1/                    # Snapshot of run 1
├── results_run2/
├── results_run3/
├── results_aggregated/              # Aggregated results across runs
│
└── run.bat                          # Automated execution pipeline
```

---

# 🚀 System Overview

This project introduces a **multi-layer curriculum intelligence pipeline** combining NLP, educational taxonomy, labour-market analysis, and future-of-work forecasting.

## **Main Stages**

### **1. Data Acquisition & Cleaning**

* Scraped job postings (IT / Software / Game Development)
* Cleaning markdown noise (** \ // etc.)
* Sentence splitting
* Every sentence carries **job_id + job_date**

### **2. Hybrid Extraction (JobBERT + GPT)**

* **JobBERT + CRF** for BIO-tagged skill/knowledge spans
* **GPT-based extractor** for structured JSON
* **Semantic agreement** using SBERT embeddings
* **Fusion Engine** merges both with:

  * confidence tiers
  * semantic density
  * hard vs soft skill classification

### **3. Taxonomy Layer**

* Hard vs soft skills
* Bloom’s taxonomy for **hard skills only**
* Semantic density scoring

### **4. Curriculum Mapping**

* Compare skills/knowledge with **SMK Software & Game Dev curriculum**
* Component mapping via SBERT
* Compute:

  * coverage percentage
  * HOT (Analyze-Evaluate-Create) distributions
  * component-level heatmaps

### **5. Future-of-Work Integration**

* Reads WEF/McKinsey-style domains from `future_domains_dummy.csv`
* Computes:

  ```
  future_weight = similarity(skill, domain) × trend_score
  ```
* Identifies:

  * future-critical skills
  * declining skills
  * curriculum gaps for future-ready design

### **6. Competency Generator (LLM)**

* Uses verified skills + future context
* Produces:

  * competency IDs
  * titles
  * descriptions
  * related skills
  * future relevance notes

### **7. Export for Review**

Creates CSVs for expert validation:

* `expert_review_jobs.csv`
* `expert_review_skills.csv`
* `expert_review_knowledge.csv`
  (with future weight + domain relevance)

---

# 🧪 Experimental Workflow

The system supports **multiple independent runs** for robust evaluation.

### **1. Run an experiment (e.g., sample size = 1000)**

```bat
run.bat
```

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

* JobBERT vs GPT vs Hybrid
* Skill/Knowledge counts
* Confidence score distributions

### **Bloom taxonomy distribution**

* For hard skills only
* Across JobBERT, GPT, Hybrid

### **Curriculum heatmap**

* Curriculum components (Y-axis)
* Bloom levels (X-axis)

### **Top-N clusters**

* Hard skills
* Knowledge items
* Soft skills
* Skills demanded but **not covered** by curriculum
* Skills "future-critical" but underrepresented

### **Time trend analysis**

* Emerging vs declining skills
* Based on `job_date`
* Monthly trend slopes

### **Future-of-work analytics**

* future_weight histogram
* top future-critical knowledge
* declining domains

---

# 🧠 Competency Generation

`generate_competencies.py` produces:

* JSON competency framework
* 10–25 competencies per batch
* Each includes:

  * id
  * title
  * description
  * **related skills**
  * **future relevance statement**

This output can be directly used in a **curriculum redesign document** or **expert workshop**.

---

# 📝 Pipeline Diagrams

All diagrams can be generated using the provided prompts in `/docs/prompts/`
(Or directly pasted into an AI image generator.)

### Includes:

* Full pipeline architecture
* Checkpoint diagrams:

  * preprocessing
  * hybrid extraction
  * taxonomy mapping
  * future-of-jobs layer
  * competency generation & review

---

# 💼 Future Work

* Incorporate **real** WEF/McKinsey datasets instead of dummy files
* Train a **domain-specific SBERT** model for improved matching
* Build a small **web interface** for expert review
* Add **semantic search** over extracted competencies
* Testing with >10,000 job postings

---

# 🙌 Citation & Acknowledgment

If you use this pipeline or insights from this project, please cite:

```
[Your Name]. Future-aware hybrid skill extraction for curriculum intelligence. (2025)
```

---

# 📬 Contact

For questions or collaboration:

* **Author:** Listyanti Dewi Astuti
* **Affiliation:** SMK Negeri 12 Malang / Universitas Negeri Malang
* **Email:** [your email here]

---
