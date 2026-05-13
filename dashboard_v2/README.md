# Dashboard v2 — Phase 2.6 minimal public UI

Streamlit app that renders the v2 competency pipeline output for browsing.

## Quick start

```bash
# 1. Make sure you have a pipeline output in results/competency_v2_pipeline_<tag>/
python scripts/run_full_v2_pipeline.py --tag latest

# 2. Launch the dashboard
streamlit run dashboard_v2/app.py
```

By default the app auto-picks the most recent `results/competency_v2_*` directory.
Override with:

```bash
streamlit run dashboard_v2/app.py -- --run-dir results/competency_v2_pipeline_e2e_v1
```

Or via env var:

```bash
COMPETENCY_V2_RUN_DIR=results/competency_v2_pipeline_e2e_v1 streamlit run dashboard_v2/app.py
```

## What you see

### Sidebar filters
- KKNI level (multi-select; from Phase 2.3 labeller)
- Min future weight
- Min grounding score (default 0.80, the v2 Req-7 gate)
- Text search across title / description / skills / rationale
- Sort options

### Per-competency detail
- Title + description
- KKNI level (Phase 2.3 SBERT labeller) — source tag shows `llm_suggested` vs `sbert_labeller`
- Future weight + empirical trend
- **Grounding score** (Phase 2.5 canonical; method = substring / sbert / mixed)
- Source jobs count + unique-sentences count
- Related skills list
- Soft skills required + description
- **Education-level demand** (Phase 2.4): which education stages demanded this competency, with %s
- **Rationale** with "Read more ▾" toggle (Phase 2.2 reasoning logging)
- **Batch reasoning** with "View reasoning ▾" toggle (Phase 2.2 CoT)
- **Source cluster details** ▾ (Phase 2.1 origin)
- **Provenance** ▾ — full list of source sentences

## Architecture

Single-file Streamlit app. No database. Reads JSON files directly from the
pipeline output directory:

```
<run-dir>/
  competencies.json         — list[CompetencyV2.to_dict()]
  batch_reasonings.json     — list[BatchReasoning.to_dict()]
  clusters.json             — list[Cluster.to_dict()]
  pipeline_report.json      — aggregate metrics (optional, for header)
```

Data is cached via `st.cache_data` per run-dir. Re-runs of the pipeline that
produce a new directory will trigger a fresh load.

## What this does NOT include yet

- Curriculum upload + coverage report (Phase 2.6 sub-deliverable, Req 8)
- User rating feature (light-signup, research signal only; Phase 2.6 sub-deliverable)
- Authentication
- Edit / annotate competencies
- Mobile-responsive styling

Those are queued for the next sprint slot.
