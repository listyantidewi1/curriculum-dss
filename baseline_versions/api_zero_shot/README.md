# api_zero_shot — OpenRouter zero-shot SkillSpan baseline

Run modern instruction-tuned LLMs (GPT-4o-mini, Claude Haiku, Llama 3.1 70B,
DeepSeek v3.2) against the SkillSpan dev/test split via OpenRouter and measure
the same strict span-set F1 + verb-preservation diagnostic used by
`baseline_versions/skill_llm/`.

Purpose: head-to-head comparison with the Skill-LLM 8B LoRA fine-tune. If a
zero-shot or few-shot API model lands within striking distance of Skill-LLM's
F1, the deployment story flips from "we need 16 GB GPUs" to "we just call an
API," which materially changes the production architecture for the Indonesian
SMK competency-recommendation system.

## Quick start

```bash
cd baseline_versions/api_zero_shot

# Smoke test (50 examples, ~30 sec, ~$0.01)
python eval.py --model openai/gpt-4o-mini --split test --sample 50

# Full test split (~1 h, ~$0.50 for GPT-4o-mini)
python eval.py --model openai/gpt-4o-mini --split test

# Few-shot with 3 demos drawn from the training set
python eval.py --model anthropic/claude-3.5-haiku --split test --few-shot 3
```

Outputs go to `outputs/metrics_{split}_{model-slug}.txt` and
`outputs/raw_outputs_{split}_{model-slug}.jsonl`.

## API key

Same convention as the rest of the repo (see `generate_competencies.py:256`):

1. `OPENROUTER_API_KEY` env var, OR
2. fall back to `api_keys/jatevo.txt`

## Models in the curated menu

| Model slug | Notes |
|---|---|
| `openai/gpt-4o-mini` | Cheapest, strong JSON adherence. ~$0.50 for full test split. |
| `openai/gpt-4o` | Premium. ~$5 for full test split. |
| `anthropic/claude-3.5-haiku` | Cheap, strong reasoning. ~$1.80 for full test split. |
| `anthropic/claude-3.5-sonnet` | Premium Claude. ~$15 for full test split. |
| `meta-llama/llama-3.1-70b-instruct` | Open-weight, mid-cost. ~$0.80 for full test split. |
| `deepseek/deepseek-v3.2` | Used elsewhere in repo for competency gen. ~$0.30 for full test split. |

Pass any other OpenRouter slug via `--model` — the warning is a hint, not a gate.

## Why these costs

GPT-4o-mini: `$0.15 / 1M input tokens, $0.60 / 1M output tokens`. Each SkillSpan
example sends ~250 input tokens (system prompt + sentence) and gets back ~80
output tokens (JSON). Test split is 3,569 examples. Rough math: $0.13 input +
$0.17 output ≈ $0.30–$0.50 depending on sentence length.

For Claude Haiku 3.5: `$0.80 / 1M in, $4 / 1M out` → ~$1.80 total for the same
3,569 examples. Sonnet is ~10× that.

## How this stacks up against Skill-LLM (RQ1 decision rubric)

The Skill-LLM paper (Herandi 2024, AAAI 2025) reports SkillSpan test F1:
- skill_f1 = 0.543
- knowledge_f1 = 0.742
- total_f1 ≈ 0.66 (micro-average)

Decision logic after running both eval suites:

| Outcome | Production extractor |
|---|---|
| API zero-shot F1 ≥ Skill-LLM F1 − 0.02 | Ship the API path. 16 GB GPU unnecessary. |
| API zero-shot F1 within 0.05 of Skill-LLM AND verb-preservation OK | Probably still ship API — operational simplicity wins for SMK deployment. |
| API zero-shot F1 ≥ 0.10 below Skill-LLM | Ship Skill-LLM via Kaggle batch or HF Inference Endpoint. |
| Few-shot k=3 closes the gap | Ship API + 3-shot prompt. Still no GPU needed. |

The verb-preservation diagnostic is a hard gate either way: if the API model
collapses verb-led skills to bare nouns (e.g. "designing UI/UX" → "UI/UX" under
SKILL), the model is unfit for our hard-skill-vs-knowledge discrimination
regardless of F1. Threshold: `verb_short_rate` must stay under
`0.144 + 0.10 = 0.244`.

## Output format

`outputs/metrics_test_openai-gpt-4o-mini.txt` looks like:

```
=== TEST METRICS -- openai/gpt-4o-mini (zero-shot via OpenRouter) ===
# strict span-set F1 (Skill-LLM Table 2 definition)
# few-shot k = 0, sample = all

skill_precision: 0.5234
skill_recall:    0.4891
skill_f1:        0.5057
knowledge_precision: 0.6512
knowledge_recall:    0.7102
knowledge_f1:        0.6794
total_precision: 0.5891
total_recall:    0.6011
total_f1:        0.5950

--- diagnostics ---
api_failures:        2 / 3569
json_parse_failures: 5 / 3569
skill_predictions:   2104
skill_short_spans:   312 (< 2 tokens)
verb_short_rate:     0.1483
training_baseline:   0.1440
verb_fail_threshold: 0.2440 (baseline + tolerance 0.10)
OK: verb-preservation diagnostic passed.
```

The numbers above are illustrative, not measured.

## Why the prompt is more verbose than Skill-LLM's training prompt

Skill-LLM was fine-tuned on the schema, so its training-time prompt is
minimal ("extract skill entities and knowledge entities"). Zero-shot models
have never seen the schema, so the system prompt here adds:

1. Explicit SKILL-vs-KNOWLEDGE definitions with examples.
2. The verb-noun discrimination rule (the domain-critical quality bar).
3. A JSON schema specification.
4. A "no commentary, no markdown fences" rule (zero-shot models sometimes
   wrap JSON in ``` blocks despite `response_format={"type": "json_object"}`).

This is the standard fair-comparison adjustment: zero-shot prompts get more
instruction, fine-tunes get more data. Both are graded on the same F1 metric.

## Sequential inference, not concurrent

`eval.py` runs API calls sequentially via tqdm. At ~1 s per call, the full test
split takes ~1 h. If you need it faster, OpenRouter handles ~10 concurrent
requests easily — a `ThreadPoolExecutor` wrapper around `call_with_retry` would
cut runtime to ~6 min. Not added by default to keep the code minimal; happy to
add it if eval time becomes a bottleneck.
