# Dashboard — Public Surface + Admin

FastAPI app exposing two surfaces:

1. **Public** at `/` — anonymous competency browser (stage chips, KKNI levels) and curriculum coverage analyzer.
2. **Admin** at `/dashboard/admin/*` — data ops: schools, users, pipeline runs, **publish a canonical run**.

The legacy `/dashboard/school/*` routes are retained for accounts created before the 2026 reframe but no new school accounts can be created.

For the full URL map, data flow, and publish architecture see **[../docs/PUBLIC_UI.md](../docs/PUBLIC_UI.md)**.

---

## Quick Start

**Before dashboard simulation:** Run the pipeline from project root:

```bat
run.bat              REM Phase 1: 18 steps (extraction → trends → recommendations → gold labeling → expert review)
REM [Optional] Expert review via: uvicorn review_ui.app:app --reload
run_phase_2.bat      REM Phase 2: 18 steps (feedback → re-generation → holdout validation → evaluation)
```

Then start the dashboard:

```bat
uvicorn dashboard.app:app --reload
```

Open `http://127.0.0.1:8000/dashboard/login`

- **Admin**: `admin@local` / `admin123` (seeded on first run)
- **School**: Create via Admin → Users (role=school, select school)

---

## Features

### Admin

| Page | Purpose |
|------|---------|
| Schools | Create schools |
| Departments | Create departments (name, vocational field) |
| Users | Create users (admin or school; school users need school_id) |
| Runs | All pipeline runs; inter-rater reliability snapshot |
| IRR | Cohen's Kappa, agreement %, shared items (when multi-reviewer) |

### School

| Page | Purpose |
|------|---------|
| Upload | Job postings (CSV), curriculum (CSV/JSON) per department. Sample files: `DATA/samples/jobs_sample.csv`, `DATA/samples/curriculum_sample.csv` |
| Runs | Trigger pipeline run; view status; supports checkpoint/resume (interrupted runs skip completed steps) |
| Results | Skills, knowledge, competencies (ranked); aggregation toggle; **trend sparklines** per skill; **"Why?"** score explainability per skill |
| Report | **Printable PDF-quality report**: skill gaps table, knowledge gaps, competency proposals, reproducibility metadata. `Print / Save as PDF` button |
| Review | In-dashboard review (skills, knowledge, competencies) with sub-tabs |
| Insights | Plots with descriptions; click to enlarge |
| How it works | Methodology page |

### New Dashboard Routes

| Route | Description |
|-------|-------------|
| `GET /dashboard/school/report` | Printable curriculum gap report (HTML, optimized for print/PDF) |
| `GET /dashboard/api/sparklines` | Monthly demand frequency per top-N skills as JSON; used by Results page SVG charts |
| `GET /dashboard/api/explain_score` | Score component breakdown for a single skill: demand, trend, future — values, weights, contributions |

---

## Multi-Reviewer & Reviewer Identity

- **Multiple users per school**: Admin creates multiple users with same `school_id`
- **Reviewer ID**: Logged-in user's **email** (unique in DB; no collision across schools)
- **Feedback storage**: Per `(item_id, reviewer_id)` in department `feedback_store/`
- **Merge**: Majority vote when aggregating; see [CALCULATIONS.md](../CALCULATIONS.md)

---

## Ranking Modes (Results Page)

| Mode | Description |
|------|-------------|
| **model_only** | Pipeline scores only; human feedback shown but not used for ordering |
| **human_adjusted** | Pipeline scores + expert verification boost/penalty |

Formulas: [CALCULATIONS.md](../CALCULATIONS.md) §4–6.

---

## Cross-School Aggregation

When "Aggregate with same vocational field across schools" is checked:

- Results combine data from all departments with the same `vocational_field`
- Contributor metadata (school, department, runs, uploads) is shown for transparency
- Review statuses and future weights are merged (any verified wins; max future_weight)

---

## Paths

| Scope | Path |
|-------|------|
| Department data | `data/schools/{school_id}/departments/{department_id}/` |
| Uploads | `uploads/` |
| Preprocessing | `preprocessing/` |
| Results | `results/` |
| Feedback | `feedback_store/` |
| Fallback | Project `results/` when department has no runs |

---

## Dependencies

```
fastapi
uvicorn
jinja2
python-multipart
pandas
```

Install: `pip install -r dashboard/requirements.txt`

---

## Checkpoint / Resume

When a department pipeline run is interrupted, re-triggering the run will skip steps whose output files already exist (`pipeline_orchestrator.py` with `resume=True`). Checkpointed steps include:

| Step | Output marker |
|------|---------------|
| `pipeline.py` | `advanced_skills.csv` |
| `verify_skills.py` | `verified_skills.csv` |
| `enrich_with_dates.py` | `advanced_skills_with_dates.csv` |
| `skill_time_trend_analysis.py` | `skill_time_trends.csv` |
| `generate_competencies.py` | `competency_proposals.json` |
| `recommendations.py` | `recommendations.csv` |
| `evaluate_extraction.py` | `extraction_evaluation_report.json` |

---

## Non-Invasive Design

- Dashboard does **not** modify `DATA/`, `results/`, or `feedback_store/` at project root
- Pipeline runs via `pipeline_orchestrator.py` with path overrides
- `run.bat` and `run_phase_2.bat` remain unchanged
