# Skill-LLM → `pipeline.py` integration

Skill-LLM 8B LoRA is the chosen Layer-1 sentence-level extractor per
[`docs/EXTRACTOR_DECISION.md`](../../docs/EXTRACTOR_DECISION.md) (2026-05-12).
At 4-bit quantization the model needs ~10–12 GB VRAM for inference — not
fittable on consumer 4 GB GPUs. The production deployment story is therefore
**offline batch inference**: process all sentences once on a borrowed GPU
(Kaggle, HF Inference Endpoint, or cloud), persist results to a JSONL, and
have the pipeline load that file at runtime instead of running the model live.

This document is the end-to-end workflow for that.

---

## End-to-end workflow

```
                                 ┌─────────────────────────────────────────┐
[1] Local: preprocess            │ preprocess_jobs_pipeline.py             │
                                 │   raw jobs → jobs_sentences.csv         │
                                 │   (one row per sentence, with           │
                                 │    sentence_id from Phase 1.1 provenance)│
                                 └─────────────────────────────────────────┘
                                              │
                                              ▼
                                 ┌─────────────────────────────────────────┐
[2] Local: export Kaggle input   │ scripts/export_sentences_for_skill_llm.py│
                                 │   jobs_sentences.csv → skill_llm_input.jsonl
                                 │   (sentence_id + sentence_text)         │
                                 └─────────────────────────────────────────┘
                                              │
                              upload as Kaggle Dataset
                                              ▼
                                 ┌─────────────────────────────────────────┐
[3] Kaggle: batch inference      │ baseline_versions/skill_llm/kaggle/      │
                                 │   run_inference_on_kaggle.py            │
                                 │   loads LoRA adapter + input sentences   │
                                 │   runs Skill-LLM per-sentence            │
                                 │   → skill_llm_extractions.jsonl          │
                                 └─────────────────────────────────────────┘
                                              │
                          download to local repo
                                              ▼
                                 ┌─────────────────────────────────────────┐
[4] Local: pipeline runs         │ pipeline.py --extraction-mode           │
                                 │     skill_llm_offline                   │
                                 │   loads pre-computed extractions as     │
                                 │   Layer 1, runs Layer 2 (OpenRouter)    │
                                 │   live, fuses                           │
                                 └─────────────────────────────────────────┘
```

---

## Step-by-step

### Step 1 — Preprocess (local, ~1 min for 1k jobs)

```bash
python preprocess_jobs_pipeline.py
```

Produces `DATA/preprocessing/data_prepared/jobs_sentences.csv` with columns
`job_id, sentence_id, sentence, ...` per Phase 1.1 provenance.

### Step 2 — Export Kaggle-ready input (local, ~5 sec)

```bash
# Full corpus
python scripts/export_sentences_for_skill_llm.py

# Smoke test (first 50 sentences)
python scripts/export_sentences_for_skill_llm.py --max-sentences 50

# Drop duplicate sentence_texts to save Kaggle compute (one extraction per
# distinct sentence, then pipeline.py expands by lookup):
python scripts/export_sentences_for_skill_llm.py --dedupe
```

Produces `skill_llm_input.jsonl` — one JSON object per line:
```
{"sentence_id": "job001_0001", "sentence_text": "Strong Python skills required."}
```

### Step 3 — Run inference on Kaggle (~15–90 min depending on size)

#### One-time setup (per fresh Kaggle workspace)

1. **Create or open a Kaggle notebook.** Accelerator: GPU T4 ×2.
2. **Add the LoRA-adapter Kaggle Dataset.** The trained adapter from the
   successful eval run lives in the Output of the previous Save Version
   (see [`docs/EXTRACTOR_DECISION.md`](../../docs/EXTRACTOR_DECISION.md));
   add that Save-Version Output as Input. Kaggle mounts it at
   `/kaggle/input/<notebook-slug>/skill_llm/lora_adapter/`.
3. **Add the input-sentences Kaggle Dataset.** Upload `skill_llm_input.jsonl`
   as a new Kaggle Dataset (or use "+ Add Data → Upload" if a quick
   one-off). It mounts at `/kaggle/input/<dataset-slug>/skill_llm_input.jsonl`.
4. **Add HF_TOKEN to Secrets** and toggle it on for the notebook.

#### Per-run

1. **Paste the contents of `baseline_versions/skill_llm/kaggle/run_inference_on_kaggle.py`**
   into the notebook's main cell.
2. **Edit two lines** at the top of the script to match your input mount paths:
   ```python
   ADAPTER_INPUT_DIR    = Path("/kaggle/input/<adapter-dataset-slug>")
   SENTENCES_INPUT_PATH = Path("/kaggle/input/<input-dataset-slug>/skill_llm_input.jsonl")
   ```
3. **Set the SAMPLE knob** for a smoke test on the first run:
   ```python
   SAMPLE = 50  # then None for the full batch
   ```
4. **Save Version → Save & Run All (Commit).** Walk away.

When it finishes the Output tab will contain `skill_llm_extractions.jsonl`.

#### Expected runtime

| Sentence count | Wall-clock on T4 fp16 (single GPU) |
|---|---|
| 50 (smoke test) | ~5 min |
| 1,000 | ~10 min |
| 5,000 | ~45 min |
| 10,000 | ~90 min |
| 20,000 | ~3 h |

The script writes flushed output every 500 sentences, so a kernel crash
mid-run preserves partial progress in `/kaggle/working/skill_llm_extractions.jsonl`.

### Step 4 — Download and run the pipeline

1. **Download** `skill_llm_extractions.jsonl` from the Kaggle Output tab into
   `results/skill_llm_extractions.jsonl` (the default path; configurable via
   `AdvancedPipelineConfig.SKILL_LLM_EXTRACTIONS_PATH`).

2. **Set extraction mode and run:**
   ```bash
   # Edit pipeline.py:
   #   AdvancedPipelineConfig.EXTRACTION_MODE = "skill_llm_offline"
   # OR override at the CLI if your runner supports it (run.bat / run_with_job_scraping.py)
   python pipeline.py
   ```

3. **Verify** by checking `pipeline_summary.txt` — the run metadata should
   list `extractor_source` counts for `"skill_llm"` (Layer 1) and the LLM
   model name (Layer 2), with their fusion counts.

---

## File reference

| Path | Role |
|---|---|
| `extractors/skill_llm_offline.py` | Local loader; reads the pre-computed JSONL and serves it as Layer 1 |
| `scripts/export_sentences_for_skill_llm.py` | Helper: jobs_sentences.csv → Kaggle-ready JSONL |
| `baseline_versions/skill_llm/kaggle/run_inference_on_kaggle.py` | Kaggle-side script; runs the trained LoRA on each sentence |
| `baseline_versions/skill_llm/skill_llm/lora_adapter/` | Trained LoRA weights (kept local, not committed; ~105 MB) |
| `pipeline.py` (modified) | Routes Layer 1 to `extract_with_skill_llm_offline` when `EXTRACTION_MODE = "skill_llm_offline"` |
| `results/skill_llm_extractions.jsonl` | The pre-computed extractions file the pipeline reads at runtime |

---

## Smoke test procedure (≤ 15 min, recommended before any production run)

1. `python scripts/export_sentences_for_skill_llm.py --max-sentences 50`
2. Upload `skill_llm_input.jsonl` to Kaggle, run inference with `SAMPLE = 50`.
3. Download output to `results/skill_llm_extractions.jsonl`.
4. Set `AdvancedPipelineConfig.EXTRACTION_MODE = "skill_llm_offline"` and
   `SAMPLE_SIZE = 5` in pipeline.py for a tiny run.
5. Run `python pipeline.py` and check the log for:
   - `[SkillLLMOfflineExtractor] loaded N records …` — confirms file loaded
   - `extractor_source: skill_llm` appearing on items in `advanced_skills.csv`
6. If everything passes, expand SAMPLE → None on Kaggle, re-run for the full
   corpus, then SAMPLE_SIZE → desired number in pipeline.py.

---

## Coverage diagnostics

If `pipeline_summary.txt` shows fewer skill_llm-tagged items than expected,
check coverage with:

```python
from extractors.skill_llm_offline import SkillLLMOfflineExtractor
ext = SkillLLMOfflineExtractor("results/skill_llm_extractions.jsonl")
# Pass the list of sentence_ids your run actually used (from jobs_sentences.csv)
print(ext.coverage(expected_sentence_ids=[...]))
```

Coverage < 1.0 means the Kaggle batch ran on fewer sentences than the
pipeline is now seeing — typically because:
- `--dedupe` was used at export time and the pipeline is querying duplicates by sentence_id (the loader's fallback to sentence_text handles this)
- The input JSONL was truncated by `--max-sentences` at export
- The Kaggle Save Version run hit the time limit (the partial-output file is still usable)

---

## What about live inference?

Live (in-process) Skill-LLM inference is **not viable on the user's local 4 GB
RTX 3050**. The minimum viable path post-sprint is one of:

1. **HF Inference Endpoint** — upload the adapter to HF Hub, deploy a T4-backed
   inference endpoint (~$0.60/hr), point pipeline.py at it via a new
   `SkillLLMHfEndpointExtractor` backend (not yet implemented; ~1 day to add).
2. **Vast.ai / RunPod A100 spot instance** — spin up for batch run, shut down.
   Same data flow as Kaggle but on a paid spot GPU.
3. **Modal / serverless** — pay-per-second, scale-to-zero.

None of these are required for the v2 sprint deadline (June 8). Kaggle batch
is sufficient for the stability experiment, user testing, and paper figures.

---

## Audit trail

- 2026-05-12: integration scaffold landed (`extractors/skill_llm_offline.py`,
  `scripts/export_sentences_for_skill_llm.py`,
  `baseline_versions/skill_llm/kaggle/run_inference_on_kaggle.py`, this doc,
  `pipeline.py` routing).
- Next: end-to-end smoke test on 50 sentences, then production batch run on
  full job-postings corpus for the stability experiment (Phase 2.6 prep).
