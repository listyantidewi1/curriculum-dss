# `jjzha_replicate` — JobBERT skill / knowledge extractor replication

**Phase 1.4 of pipeline-redesign-v2.** Audit + scaffold for replicating
[jjzha/jobbert_skill_extraction](https://huggingface.co/jjzha/jobbert_skill_extraction)
and [jjzha/jobbert_knowledge_extraction](https://huggingface.co/jjzha/jobbert_knowledge_extraction)
as our pipeline's BERT extraction backbone.

See [`AUDIT.md`](AUDIT.md) for the line-by-line comparison against our existing
`jobbert_crf` (multitask + CRF) and `v3_stl` (single-task + CRF) baselines, and
the rationale for every hyperparameter in [`config.py`](config.py).

---

## Headline numbers (in-repo, before this work)

| Setup | Skill F1 | Knowledge F1 |
|---|---|---|
| `baseline_versions/jobbert_crf` (multitask + CRF) | 0.5369 | 0.6952 |
| `baseline_versions/v3_stl` (STL + CRF) | 0.5629 | 0.7181 |
| **Phase 1.4 target** | **≥ 0.70** | **≥ 0.80** |

The audit identifies the architectural deltas (CRF removal, two single-task
softmax heads, demo-style `aggregate_span` post-merge) likely responsible for
the gap. This package implements them.

---

## Files

| File | Purpose |
|---|---|
| `AUDIT.md` | Line-by-line comparison against the published config; what to replicate and why. |
| `config.py` | All hyperparameters in one place. Single source of truth. |
| `data_utils.py` | SkillSpan loader + `SingleTaskSkillSpanDataset` for HF `BertForTokenClassification`. First-subword-only label alignment by default; `LABEL_ALL_SUBWORDS=True` toggles the legacy "all-subwords" alignment for ablation. |
| `train.py` | Single-task trainer (one run per task). Plain softmax token classification, no CRF. Saves a HF `save_pretrained` checkpoint. |
| `eval.py` | Re-evaluates a trained checkpoint or a HF Hub id on SkillSpan dev + test. Token-level seqeval F1 — apples-to-apples with our existing baselines. |
| `eval_published.py` | Reproduces jjzha's exact demo Space inference path (HF `pipeline(aggregation_strategy="first")` + the `aggregate_span` post-merger) and reports span-set F1 against gold spans. |
| `outputs/` | Per-task metrics, training logs, and the `published_*` direct-eval results land here. |

---

## How to run (GPU / training environment)

This stage requires GPU + the published `jjzha/jobbert-base-cased` backbone
(~440 MB) and is best run on the user's training rig, not in the dev shell.

```bash
# from the project root, activate the existing project venv first

# 1. Train both single-task models. ~15-30 min each on a single A100.
python baseline_versions/jjzha_replicate/train.py --task skill
python baseline_versions/jjzha_replicate/train.py --task knowledge

# 2. (Optional but recommended) Direct-evaluate the published checkpoints
#    so we have an upper-bound number to compare our trained replicate against.
python baseline_versions/jjzha_replicate/eval.py \
    --task skill --hf jjzha/jobbert_skill_extraction
python baseline_versions/jjzha_replicate/eval.py \
    --task knowledge --hf jjzha/jobbert_knowledge_extraction

# 3. (Optional) Reproduce the demo's exact inference pipeline numbers.
python baseline_versions/jjzha_replicate/eval_published.py \
    --task skill --hf jjzha/jobbert_skill_extraction
python baseline_versions/jjzha_replicate/eval_published.py \
    --task knowledge --hf jjzha/jobbert_knowledge_extraction

# 4. Re-evaluate our existing baselines (no retraining) for the matrix.
#    These commands live in their respective package dirs and produce the
#    metrics_test.txt files already present in their outputs/ folders, so
#    only re-run them if you want fresh numbers.
```

---

## What each output looks like

`outputs/<task>/metrics_test.txt` from `train.py`:

```
=== TEST METRICS ===
precision: 0.7XXX
recall:    0.6XXX
f1:        0.7XXX
```

`outputs/<task>_pipeline/metrics_test_pipeline.txt` from `eval_published.py`:

```
=== TEST PIPELINE METRICS — hf:jjzha/jobbert_skill_extraction ===
# inference: HF pipeline(aggregation_strategy='first') + aggregate_span
gold_spans: 1562
pred_spans: 1XXX
true_pos:   1XXX
precision:  0.7XXX
recall:     0.6XXX
span_f1:    0.7XXX
```

---

## Open work after the GPU runs come back

Per the plan §1.4 sub-tasks 4-6, all gated on the eval matrix landing:

1. **Pick the winner per task.** Compare token-level seqeval F1 across:
   - existing `jobbert_crf/outputs/metrics_test.txt`
   - existing `v3_stl/jobbert_*_crf/outputs/metrics_test.txt`
   - new `jjzha_replicate/outputs/<task>/metrics_test.txt`
   - new `jjzha_replicate/outputs/published_<task>/metrics_test.txt`
2. **Update `config.py`** in the project root: split `MULTITASK_MODEL_DIR` into
   per-task paths (`SKILL_MODEL_DIR`, `KNOWLEDGE_MODEL_DIR`) pointing at the
   chosen winners, and update `pipeline.py:ModelManager.load_bert_model()` to
   load both as separate `BertForTokenClassification` checkpoints (no CRF).
3. **Stretch interventions** *(if and only if the replicate misses targets)*:
   - `LABEL_ALL_SUBWORDS = True` in `config.py` — re-train, compare.
   - `DROPOUT = 0.2` — re-train, compare.
   - Cosine LR schedule (extend `train.py` to import `get_cosine_schedule_with_warmup`).
4. **Write `REPLICATION_REPORT.md`** in this directory: F1 matrix, which
   variant won, hyperparameter table, a short discussion of what closed the
   gap (CRF removal? STL split? subword alignment?), and any stretch
   interventions that were necessary.
