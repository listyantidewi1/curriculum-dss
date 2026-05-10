# AUDIT — Skill-LLM-style LoRA fine-tune (Phase 1.5)

**Why we're here:** [`baseline_versions/jjzha_replicate/REPLICATION_REPORT.md`](../jjzha_replicate/REPLICATION_REPORT.md)
shows the published `jjzha/jobbert_skill_extraction` checkpoint scores 0.519 / 0.653
on SkillSpan test — comparable to vanilla BERT, well below the 0.70 / 0.80 targets
in `requirements.md` Req 3. The SkillSpan SOTA is **Skill-LLM** (Herandi et al. 2024,
arXiv:2410.12052) at **0.543 / 0.742** — an 8B-parameter LLaMA 3 fine-tuned with LoRA.
This package replicates that approach.

## Source paper

Herandi, Li, Liu, Hu, Cai. **Skill-LLM: Repurposing General-Purpose LLMs for Skill
Extraction.** AAAI 2025. https://github.com/herandy/Skill-LLM

## Reported numbers

| Method | Skill F1 | Knowledge F1 | Total |
|---|---|---|---|
| **Skill-LLM (LoRA LLaMA 3 8B)** | **0.543** | **0.742** | **0.648** |
| Fine-tuned GLiNER (166M) | 0.496 | 0.655 | 0.584 |
| jobSpanBERT (best published BERT) | 0.563 | 0.619 | 0.589 |
| `jjzha_replicate` (our trained baseline) | 0.5050 | 0.6570 | 0.581 |

Skill-LLM is the new SOTA on knowledge (+0.085 over jobSpanBERT) and competitive on
skill (within 0.020 of jobSpanBERT, basically tied).

---

## CRITICAL INVARIANT — verb preservation in skill spans

A persistent failure mode in skill extraction tools is **collapsing a verb-led action
phrase down to its noun head**, conflating skill with knowledge. Concretely:

| Source sentence | What we want | What we don't want |
|---|---|---|
| "responsible for designing UI/UX" | SKILL = "designing UI/UX" | SKILL = "UI/UX" |
| "experience with Java" | KNOWLEDGE = "Java" | SKILL = "Java" |
| "implementing and promoting QA topics" | SKILL = "implementing and promoting QA topics", KNOWLEDGE = "QA" | SKILL = "QA" |

This distinction is **load-bearing** for the downstream pipeline:

- The competency generator turns SKILL items into measurable learning outcomes
  ("Design UI/UX components by ..."). A noun-only skill ("UI/UX") cannot be turned
  into a competency without re-inferring a verb, which the LLM does inconsistently.
- The Bloom-free curriculum mapping (post pipeline-redesign-v2) relies on the verb
  to signal action level (design / analyze / evaluate / create). Strip the verb and
  the downstream KKNI labeler loses its strongest cue.

**Skill-LLM's output format makes this distinction explicit:**

```json
{
  "SKILL": [
    {"skill_span": "implementing and promoting all QA relevant topics",
     "context": "for implementing and promoting all QA relevant topics on"}
  ],
  "KNOWLEDGE": [
    {"skill_span": "QA",
     "context": "all QA relevant"}
  ]
}
```

`SKILL.skill_span` is the **verb-led action phrase**; `KNOWLEDGE.skill_span` is the
**noun**. The model is trained on SkillSpan, where this distinction is part of the
gold annotation (paper §"Methodology"). We preserve the same prompt template and
output schema verbatim — see `config.py:SYSTEM_PROMPT` + `prepare_data.py`.

`eval.py` includes a diagnostic that **counts every SKILL item shorter than 2 tokens
or starting with a non-verb POS tag** as a verb-preservation failure, and reports
the rate alongside span-set F1. The CI gate is "verb-failure rate < 5%". This is a
guardrail against the model learning to emit nouns under the SKILL key during
fine-tuning.

---

## Architecture comparison vs `jjzha_replicate`

| Dimension | `jjzha_replicate` | `skill_llm` |
|---|---|---|
| Backbone | `jjzha/jobbert-base-cased` (110M, BERT) | `meta-llama/Llama-3.1-8B-Instruct` (8B, LLaMA 3) |
| Adapter | none — full fine-tune | LoRA (rank 64, q_proj + v_proj) |
| Quantisation | none | 4-bit NF4 via bitsandbytes (training + inference) |
| Task formulation | token classification, BIO labels | seq2seq JSON generation |
| Output schema | per-token B/I/O | structured JSON with skill_span + context per item |
| Decoding | argmax over softmax logits | constrained-greedy / beam decode of JSON |
| Skill vs knowledge | separate single-task heads (or multitask 5-class) | single model, two output keys |
| Verb preservation | implicit (labelled at training time, not enforced at inference) | **explicit** in output schema |
| Span boundaries | implicit (BIO transitions) | explicit (string match against `context` window) |

The Skill-LLM formulation does more work per inference call (generating up to a few
hundred tokens of JSON instead of classifying ~50 tokens) but gives us:

1. **Verb preservation as a structural property** of the output, not an emergent
   property of training data balance.
2. **Disambiguation when the same word appears twice in a sentence**: the `context`
   field ("for implementing and promoting all QA relevant topics on") gives an
   exact substring to map back to character offsets, so the existing Phase 1.1
   sentence-level provenance plumbing in `pipeline.py` becomes deterministic
   instead of best-effort substring matching.
3. **A natural place to extend the schema**: future revisions can add an
   `is_action_oriented: bool` field to make the verb constraint machine-checkable.

## Training recipe (reproducing Skill-LLM Table 2 exactly)

From Skill-LLM §"Experimental Setup":

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | LLaMA 3 8B Instruct | paper |
| LoRA rank | 64 | paper |
| LoRA alpha | 128 (2× rank) | standard |
| LoRA dropout | 0.05 | standard |
| LoRA target modules | `q_proj`, `v_proj` | paper |
| Learning rate | 2e-4 | paper |
| Batch size | 4 | paper |
| Gradient accumulation | 4 (effective batch 16) | added for stability on 24 GB GPUs |
| Epochs | 2 | paper |
| Weight decay | 0.0 | paper |
| Warmup ratio | 10% | paper |
| LR schedule | cosine | paper |
| Quantisation | NF4 (bnb_4bit) | added; 8B fits on 24 GB only with 4-bit |

**Why LLaMA 3.1 8B Instruct over LLaMA 3 8B Instruct:** the paper used 3.0 (released
April 2024); 3.1 was released July 2024 with the same architecture and 128k context.
Drop-in replacement, marginally better instruction following.

**Alternative backbones** (no HF gating; for users without a Meta access token):

- `Qwen/Qwen2.5-7B-Instruct` — same parameter scale, no auth, similar instruction
  following on JSON output. Set `BASE_MODEL_NAME` in `config.py`.
- `mistralai/Mistral-7B-Instruct-v0.3` — slightly smaller, also no auth.
- `unsloth/llama-3-8b-Instruct-bnb-4bit` — pre-quantised LLaMA 3 from Unsloth,
  no Meta auth needed.

The LoRA recipe transfers across all three. Numbers reported in the
`REPLICATION_REPORT.md` should specify which backbone was used.

## Open questions

- **GLiNER as a lighter alternative (166M params, 0.496 / 0.655).** Skill-LLM Table 2.
  This is half the size of jobbert_crf and beats it on knowledge by 4 points. If the
  8B fine-tune turns out to be too slow at inference for `pipeline.py`'s job-by-job
  loop, GLiNER is the backup. Out of scope for this audit; would be a Phase 1.5b.
- **Dataset augmentation.** The paper notes "Future research could explore... using
  LLMs to generate synthetic data." Out of scope for the initial replicate; revisit
  if the verb-preservation diagnostic on real Indonesian SMK postings shows
  out-of-distribution failures.
- **Inference latency.** ~8B at 4-bit on A100 is ≈ 30 ms / sentence with batched
  decoding. For ~10K postings × ~30 sentences = 300K calls, that's ~2.5 hours
  end-to-end — acceptable but not fast. Consider batched generation or vLLM if
  this becomes a bottleneck.
