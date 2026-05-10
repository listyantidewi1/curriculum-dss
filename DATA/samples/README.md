# Sample CSVs for Dashboard Simulation

These files are extracted from actual pipeline data for testing and simulation.

| File | Source | Use |
|------|--------|-----|
| `jobs_sample.csv` | SkillSpan `DATA/train.json` | Upload to dashboard for job postings simulation |
| `curriculum_sample.csv` | `config.CURRICULUM_COMPONENTS` | Upload to dashboard for curriculum simulation |

## Regenerating samples

```bash
python scripts/create_sample_csvs.py
```

## Job postings format

- **description** (required): Full job description text
- **id** (optional): Job identifier
- **date_posted** (optional): YYYY-MM-DD for trend analysis

## Curriculum format

- **component_id**: Unique curriculum component ID
- **component_name**: Human-readable name
- ~~**bloom_level**~~: removed in pipeline-redesign-v2 Phase 1.3. The
  curriculum loader now ignores this column when present, so legacy
  CSVs that still carry it continue to load without error per Req 1.5.
- **phrases**: Comma-separated learning phrases
