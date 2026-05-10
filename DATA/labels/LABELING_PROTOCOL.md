# Gold Set Labeling Protocol

## Purpose
Create ground-truth labels for evaluating extraction quality, scoring
calibration, and future-domain mapping accuracy. Supports multi-reviewer
labeling with majority vote and inter-rater reliability (Cohen's Kappa).

## Labeling Options

| Method | Use Case |
|--------|----------|
| **Gold Labeling UI** | Multi-reviewer, auto-save, overlap set for IRR |
| **Manual CSV edit** | Single reviewer, direct edit of gold_*.csv |

## Gold Labeling UI (Recommended for Multi-Reviewer)

```bat
uvicorn gold_labeling_ui.app:app --reload
```

Open `http://127.0.0.1:8000/?labeler_id=alice` (use different labeler_id per reviewer).

Labels are stored in `DATA/labels/gold_labels/` per (gold_id, labeler_id).

**After all reviewers finish:**
```bat
python merge_gold_labels.py
```
Then run `evaluate_extraction.py` and `evaluate_future_mapping.py` (they use `*_merged.csv` automatically).

## Files to Label

| File | Columns to Fill | Items |
|------|----------------|-------|
| gold_skills.csv | is_correct, type_label | ~150 |
| gold_knowledge.csv | is_correct | ~100 |
| gold_future_domain.csv | true_domain_id | ~100 |

> **Note (pipeline-redesign-v2 Phase 1.3):** the `bloom_label` column was
> removed from the gold-skills schema in May 2026. If you have a local
> copy of `gold_skills.csv` from before that change, the `bloom_label`
> column is now ignored by the loader; you can safely leave existing
> values alone or drop the column entirely.

## Instructions

### gold_skills.csv
For each row, read the `skill` text and the `job_id` context.

- **is_correct**: Was this skill correctly extracted from the job text?
  - `yes` — the text is a real skill/competence mentioned in the posting
  - `no` — garbage, fragment, not a skill, or hallucinated
- **type_label**: What type of skill is this?
  - `Hard` — technical, domain-specific (verb-led action phrase like "designing UI/UX")
  - `Soft` — interpersonal, transferable (single-word personality terms like "passion" are also valid)
  - `Both` — hybrid (e.g., "technical communication")
  - Leave blank if unsure
- ~~`bloom_label`~~: removed in pipeline-redesign-v2 Phase 1.3. Skip this column if your CSV still has it.
- **labeler_id**: Your reviewer ID (e.g., "dewi", "alice")
- **notes**: Optional free text

### gold_knowledge.csv
For each row, read the `knowledge` text and the `job_id` context.

- **is_correct**: Was this knowledge item correctly extracted?
  - `yes` — real knowledge/concept from the posting
  - `no` — garbage, fragment, or hallucinated
- **labeler_id**: Your reviewer ID
- **notes**: Optional

### gold_future_domain.csv
For each row, read the `item_text` and the `pipeline_domain` assigned by the
pipeline.

- **true_domain_id**: Which future domain best fits this item?
  - Use domain IDs from `future_domains.csv` (or `future_domains_dummy.csv`)
  - Write `none` if the item does not map to any domain
  - Write `unclear` if you cannot decide
- **labeler_id**: Your reviewer ID
- **notes**: Optional

## Overlap Items (IRR)
Items marked with `is_overlap = True` in the gold set CSV must be labeled by
**all** reviewers to compute inter-rater reliability (Cohen's Kappa for 2
raters; Fleiss' Kappa for 3+). The Gold Labeling UI marks these with an
"Overlap (IRR)" badge.

Overlap items are selected via **stratified random sampling** (not first-N)
at export time to avoid order bias. The default overlap size is 30 items
per gold set (configurable via `--overlap_n`). The export script uses
`config.RANDOM_SEED` so the overlap set is reproducible.

## Majority Vote
When multiple reviewers label the same item:
- **is_correct**: yes if count(yes) > count(no); else no (conservative tie-break)
- **type_label**, **true_domain_id**: mode (most frequent)

## Pre-Labeling Calibration

Before starting, all reviewers should:

1. **Read this protocol** in full.
2. **Review 5 practice items** together (not counted in the gold set) to
   align on edge cases and ambiguous extractions.
3. **Discuss disagreements** on the practice items and agree on decision
   rules for common edge cases (see below).

## Edge-Case Decision Rules

| Situation | Decision |
|-----------|----------|
| Extracted text is a valid skill but truncated (e.g., "machine learn") | Mark `no` — fragments are invalid extractions |
| Text combines two skills (e.g., "Python and Java") | Mark `yes` if both are real skills mentioned in the posting |
| Skill is too generic to be actionable (e.g., "work") | Mark `no` |
| Skill is valid but the type label is wrong | Mark `yes` for is_correct; correct type_label (Bloom column was removed in Phase 1.3) |
| Multi-domain skills (e.g., "data visualization" fits both Analytics and UX) | Choose the **primary** domain based on the item's typical use context; add the secondary domain in notes |
| Domain mapping: skill fits no listed domain well | Use `none` |
| Domain mapping: cannot decide between two close domains | Use `unclear` and explain in notes |

## Quality Checks
- Label all items (do not skip)
- If in doubt, mark `no` for is_correct (conservative)
- For manual CSV: save as UTF-8 (do not change column order)
- Aim for at least 90% completion rate before submitting
