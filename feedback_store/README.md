# Feedback Store

Canonical storage for human-in-the-loop review feedback. Used by the pipeline for adaptation (type overrides, human-verified filtering, competency few-shot examples). Bloom overrides were removed in pipeline-redesign-v2 Phase 1.3.

## Locations

| Context | Path |
|---------|------|
| **Default run** (review_ui, run.bat) | `feedback_store/` (project root) |
| **Dashboard** (per department) | `data/schools/{school_id}/departments/{department_id}/feedback_store/` |

Both use the same schema. `import_feedback.py` reads from a configurable path (default: project `feedback_store/`).

## Schema

| File | Description |
|------|--------------|
| `skill_feedback.csv` | review_id, reviewer_id, human_valid, human_type, human_notes (human_bloom removed in Phase 1.3) |
| `knowledge_feedback.csv` | review_id, reviewer_id, human_valid, human_notes |
| `competency_feedback.csv` | competency_id, reviewer_id, human_quality, human_relevant, human_notes |
| `human_verified_skills.csv` | Merged (majority vote); skills where human_valid=valid |
| `human_verified_knowledge.csv` | Merged; knowledge where human_valid=valid |
| `competency_assessments.json` | `{ competency_id: { quality, relevant, notes } }` |
| ~~`bloom_corrections.json`~~ | **Removed in pipeline-redesign-v2 Phase 1.3.** |
| `type_corrections.json` | `{ skill_text: Hard|Soft|Both }` |
| `inter_rater_report.json` | Cohen's Kappa, agreement %, when ≥2 reviewers |
| `calibrated_threshold.json` | From validate_parameters.py (AUC, Brier) |

## Multi-Reviewer

- One row per `(item_id, reviewer_id)` in `*_feedback.csv`
- `import_feedback.py` merges via **majority vote**; see [CALCULATIONS.md](../CALCULATIONS.md) §3

## Usage

1. Run export scripts to create `expert_review_*.csv` in OUTPUT_DIR
2. Use **review_ui** (`?reviewer_id=alice`) or **dashboard** (school review) to submit feedback
3. Run `import_feedback.py` to merge feedback → human_verified_*.csv, corrections, IRR
4. Run pipeline with `--use_human_verified` and competency gen with `--human_verified_only`

## RAG / Fine-Tuning Extension

- **RAG**: `human_verified_skills` + high-quality competencies (quality >= 4, relevant == yes) can be embedded and indexed.
- **Fine-tuning**: Build dataset `(skills, future_context) -> competency JSON` from human-approved competencies.
