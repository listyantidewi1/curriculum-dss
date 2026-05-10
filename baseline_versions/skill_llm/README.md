# `skill_llm` — Skill-LLM-style LoRA fine-tune (Phase 1.5)

Replication of [Skill-LLM (Herandi et al. 2024, AAAI 2025)](https://arxiv.org/abs/2410.12052)
on SkillSpan: a LoRA fine-tune of LLaMA 3.1 8B Instruct that emits structured JSON
distinguishing **verb-led skill spans** from **noun knowledge spans**.

This package is the successor to [`baseline_versions/jjzha_replicate/`](../jjzha_replicate/).
The decision to switch backbones is documented in
[`../jjzha_replicate/REPLICATION_REPORT.md`](../jjzha_replicate/REPLICATION_REPORT.md);
the architectural rationale + the **verb-preservation invariant** that gates this
work are documented in [`AUDIT.md`](AUDIT.md).

## Why we're doing this (one-paragraph summary)

`jjzha_replicate` showed the published `jjzha/jobbert_skill_extraction` checkpoint
scores 0.519 / 0.653 on SkillSpan test — comparable to vanilla BERT. The literature
SOTA for SkillSpan is **Skill-LLM at 0.543 / 0.742** (Skill-LLM Table 2). That's the
new ceiling we're chasing. Beyond F1, the Skill-LLM output format **explicitly
preserves the verb in skill spans** (e.g. `"designing UI/UX"` under SKILL versus
`"UI/UX"` under KNOWLEDGE) — critical for the downstream competency generator and
KKNI labeler, which both lose information if the verb is stripped.

## Targets (revised from `requirements.md` Req 3)

| Metric | Original target (`requirements.md` Req 3) | New target (literature-grounded) |
|---|---|---|
| Skill F1 | ≥ 0.70 | **≥ 0.54** (matches Skill-LLM SOTA = 0.543) |
| Knowledge F1 | ≥ 0.80 | **≥ 0.74** (matches Skill-LLM SOTA = 0.742) |
| Total span F1 | — | **≥ 0.65** (matches Skill-LLM SOTA = 0.648) |
| Verb-preservation gate | — | **model's short-SKILL rate ≤ training baseline + 0.10** (see AUDIT.md "CRITICAL INVARIANT") |

## Files

| File | Purpose |
|---|---|
| `AUDIT.md` | Architectural comparison vs `jjzha_replicate` + verb-preservation invariant. |
| `config.py` | Single source of truth: backbone, LoRA recipe, training hyperparameters, prompt template, verb-failure threshold. |
| `prepare_data.py` | SkillSpan → Skill-LLM chat-format JSONL. Emits `datasets/{train,dev,test}.jsonl`. Runs a verb-preservation sanity check on the prepared training set. |
| `train.py` | LoRA fine-tune via PEFT + transformers + bitsandbytes 4-bit. Saves `lora_adapter/`. |
| `eval.py` | Generate JSON outputs on dev / test, parse, compute span-set F1 against gold, run the verb-preservation diagnostic. |
| `outputs/` | Per-split metrics + raw model outputs (jsonl). |

## Pre-flight (one-time)

```bash
# Project venv must include these on top of what's already there
pip install transformers>=4.45 datasets peft>=0.13 bitsandbytes accelerate>=0.34 tqdm

# LLaMA 3.1 is gated on HuggingFace. Either:
huggingface-cli login        # log in with a Meta-approved account, OR
# switch BASE_MODEL_NAME in config.py to a non-gated alternative:
#   "Qwen/Qwen2.5-7B-Instruct"
#   "mistralai/Mistral-7B-Instruct-v0.3"
#   "unsloth/llama-3-8b-Instruct-bnb-4bit"
```

## Run book

```bash
# 1. Convert SkillSpan to Skill-LLM chat JSONL.
#    Sanity-check at the end reports verb-preservation rate on the training set.
python baseline_versions/skill_llm/prepare_data.py

# 2. LoRA fine-tune. ~1-2h on a single A100; ~3-4h on a 24GB consumer card.
python baseline_versions/skill_llm/train.py

# 3. Evaluate the fine-tuned adapter on dev + test.
python baseline_versions/skill_llm/eval.py

# 4. (Optional) sanity check against the un-finetuned base model.
python baseline_versions/skill_llm/eval.py --adapter "" --output-subdir base_only
```

After step 3 you'll have `outputs/trained/metrics_{dev,test}.txt` with the F1
matrix + verb-preservation diagnostic. Decision rules:

- **Total F1 ≥ 0.65 AND verb-preservation gate passes** → promote to `pipeline.py`
  per the integration plan below. The gate passes when the model's short-SKILL
  rate stays within `VERB_FAILURE_TOLERANCE_DELTA` (default 0.10) of the
  training baseline (≈14% on SkillSpan, recorded in
  `datasets/training_stats.json`).
- **Total F1 < 0.65** → diagnose with the raw outputs JSONL: are JSON parse
  failures the bottleneck (try larger `INFERENCE_MAX_NEW_TOKENS`), or are
  span-text mismatches the bottleneck (try `INFERENCE_DO_SAMPLE = True` with
  low temperature for less brittle outputs)?
- **Verb-preservation gate fails** → DO NOT promote. The fine-tune has drifted
  toward emitting tech nouns ("UI/UX", "Java") under SKILL. Re-check the
  `metrics_*.txt` for the actual rate vs threshold; if `verb_short_rate` is
  far above `verb_fail_threshold`, retrain with more epochs or a higher LoRA
  rank.

## Pipeline integration (after the matrix passes)

The integration is a drop-in replacement for the BERT path in
`pipeline.py:ModelManager.extract_with_bert`:

1. Add a new `LLM_EXTRACTOR_MODEL` config in [`config.py`](../../config.py)
   pointing at the LoRA adapter directory.
2. Extend `ModelManager` with `extract_with_skill_llm(text)` that calls the
   model the same way `eval.py:generate` does. Returns `(skills, knowledge)`
   raw dicts where each dict carries `text`, `confidence` (= 1.0; LLM doesn't
   emit per-span scores), AND the `context` field for offset disambiguation.
3. The Phase 1.1 sentence-level provenance plumbing already accepts these
   raw dicts; the `context` field will make the LLM-side substring matching
   in `_pinpoint_llm_to_sentences` deterministic instead of best-effort.
4. Toggle `EXTRACTION_MODE` from `"llm_only"` (the post-2026-reframe default
   that still uses the existing OpenRouter LLM at full posting) to a new
   mode `"skill_llm"` that uses this fine-tuned model per-sentence.
5. The existing fusion engine (`AdvancedFusionEngine.fuse_skills_advanced`)
   continues to work unchanged — Skill-LLM produces `SkillItem` instances
   the same way the BERT path does.

The verb-led skill spans the new model emits feed `generate_competencies.py`
unchanged — the existing prompt already expects verb-led skills (it asks the
LLM to write competencies from action-oriented input).

## What the verb-preservation diagnostic actually catches

`eval.py` counts predicted SKILL spans shorter than `VERB_PRESERVATION_MIN_TOKENS`
(2 tokens) and compares the rate to the training-set baseline written by
`prepare_data.py` to `datasets/training_stats.json`.

SkillSpan's training data has ~14% single-token SKILL spans because the SKILL
category covers soft skills like `passion`, `empathetic`, `self-starter`,
`team-player` — those are legitimate single-token skills, not noun leaks. So
a hard 5% gate would always fail on a well-trained model.

The gate passes when the model's short-SKILL rate stays within
`VERB_FAILURE_TOLERANCE_DELTA = 0.10` of the training baseline. On SkillSpan
that means up to ~24% short-SKILL is OK; 25%+ is suspicious. Failure modes
that drive the rate up:

- LoRA rank too low (< 32) — adapter can't represent the structured JSON well
  enough and falls back to whatever pre-training memorised.
- Training cut short (< 2 epochs on SkillSpan) — model hasn't fully internalised
  the skill / knowledge distinction yet.
- Bad chat template — if `apply_chat_template` is misconfigured, loss masking
  in `train.py:tokenise_record` may mask the JSON output too, training nothing.

The `outputs/trained/raw_outputs_*.jsonl` files preserve every generated
string, parsed or not, so post-mortem analysis is easy.

## When to switch backbones

The default backbone is `meta-llama/Llama-3.1-8B-Instruct` to match the paper
exactly. If you want to compare:

```python
# in config.py
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"          # no Meta auth
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # smaller, no auth
BASE_MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"  # pre-quantised
```

`prepare_data.py` and `eval.py` are backbone-agnostic; only `train.py` cares,
and only because PEFT needs to know which `target_modules` exist (q_proj /
v_proj are present in all four models above).

After running with a different backbone, write the resulting F1 numbers into
[`../jjzha_replicate/REPLICATION_REPORT.md`](../jjzha_replicate/REPLICATION_REPORT.md)
under "F1 matrix" so the comparison stays in one place.
