# Review UI — Internal / Development

Standalone web UI for expert review during internal or development workflows. No authentication; uses URL parameter for reviewer identity.

---

## Quick Start

```bat
uvicorn review_ui.app:app --reload
```

- Single reviewer: `http://127.0.0.1:8000/` (reviewer_id=default)
- Multi-reviewer: `http://127.0.0.1:8000/?reviewer_id=alice`

---

## Data Paths

- **Output/templates**: `config.OUTPUT_DIR` (default: `results/`)
- **Feedback**: `feedback_store/` (project root)

---

## Features

- **Skills**: Validity, type (Hard/Soft/Both), notes
- **Knowledge**: Validity, notes
- **Competencies**: Quality (1–5), relevance (yes/no/partial), notes
- **Show unreviewed only**: Filter per reviewer
- **Auto-save**: Debounced save on change
- **Progress**: Reviewed count per panel

---

## vs Dashboard

| Aspect | review_ui | Dashboard |
|--------|-----------|-----------|
| Purpose | Internal / dev | Production |
| Auth | None | Login |
| Reviewer ID | URL param | User email |
| Data | Default results/ | Per-department |

See [PIPELINE.md](../PIPELINE.md) §4b and [dashboard/README.md](../dashboard/README.md).
