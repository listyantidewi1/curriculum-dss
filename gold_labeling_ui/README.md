# Gold Labeling UI

Web UI for ground-truth labeling of the gold set. Supports **multi-reviewer** labeling with auto-save and inter-rater reliability (Cohen's Kappa).

## Quick Start

```bat
uvicorn gold_labeling_ui.app:app --reload
```

Open `http://127.0.0.1:8000/?labeler_id=alice`

Use different `labeler_id` per reviewer (e.g. `?labeler_id=dewi`).

## Prerequisites

Run `export_gold_set.py` first (step 11 of Phase 1) to create:
- `DATA/labels/gold_skills.csv`
- `DATA/labels/gold_knowledge.csv`
- `DATA/labels/gold_future_domain.csv`

## Workflow

| Step | Action |
|------|--------|
| 1 | Run Phase 1 (`run.bat`) through step 11 |
| 2 | Start `uvicorn gold_labeling_ui.app:app --reload` |
| 3 | Each reviewer opens with `?labeler_id=NAME` |
| 4 | Label all items (Skills, Knowledge, Domain tabs) |
| 5 | **Overlap set**: First 20 items in each tab must be labeled by ALL reviewers |
| 6 | Run `python merge_gold_labels.py` |
| 7 | Run `python evaluate_extraction.py` and `python evaluate_future_mapping.py` |

## Storage

Labels are stored in `DATA/labels/gold_labels/`:
- `skill_labels.csv` — gold_id, labeler_id, is_correct, type_label, notes
- `knowledge_labels.csv` — gold_id, labeler_id, is_correct, notes
- `domain_labels.csv` — gold_id, labeler_id, true_domain_id, notes

Each (gold_id, labeler_id) pair is stored separately so reviewers do not overwrite each other.

## Scientific Rigor

- **Overlap set**: First 20 items per category for Cohen's Kappa
- **Majority vote**: Applied by `merge_gold_labels.py` before evaluation
- **Conservative rule**: Tie on is_correct → no
- **IRR**: Computed from `gold_labels/` when 2+ reviewers label overlap items

## Labeling Protocol

See [DATA/labels/LABELING_PROTOCOL.md](../DATA/labels/LABELING_PROTOCOL.md).
