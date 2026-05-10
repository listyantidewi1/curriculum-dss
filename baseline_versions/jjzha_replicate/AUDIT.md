# AUDIT — Replicating jjzha's published JobBERT skill / knowledge extractors

**Phase 1.4 of pipeline-redesign-v2.** Goal: skill F1 ≥ 0.70, knowledge F1 ≥ 0.80
on the SkillSpan test split (Req 3.1, 3.2). Current in-repo baselines:

| Repo baseline | Skill F1 | Knowledge F1 | Setup |
|---|---|---|---|
| `baseline_versions/jobbert_crf/` (multitask + CRF) | 0.5369 | 0.6952 | one BERT, two heads, two CRFs, joint NLL loss |
| `baseline_versions/v3_stl/` (single-task + CRF) | 0.5629 | 0.7181 | two checkpoints (one per task), each with a CRF |

The user has personally observed that
[jjzha/jobbert_skill_extraction](https://huggingface.co/jjzha/jobbert_skill_extraction)
and the [demo Space](https://huggingface.co/spaces/jjzha/skill_extraction_demo)
produce noticeably stronger outputs than these baselines. This audit pins down
the **architectural** and **inference-time** differences first; hyperparameters
must come from the SkillSpan NAACL 2022 paper since the model card does not
publish them.

The published model is the official extractor accompanying the SkillSpan paper
(Zhang et al., NAACL 2022). The README explicitly states **"Single-task learning
approach (outperforms multi-task)"** and **"Knowledge can be seen as hard skills
and skills are both soft and applied skills."** That single sentence captures
the central architectural claim we are about to replicate.

---

## 1. Architecture

| Dimension | Our `jobbert_crf` (multitask) | Our `v3_stl` (STL + CRF) | jjzha published | Action |
|---|---|---|---|---|
| Backbone | `jjzha/jobbert-base-cased` | `jjzha/jobbert-base-cased` | `jjzha/jobbert-base-cased` | **Match.** Keep the same backbone. |
| Heads | 2 linear classifiers (skill, knowledge) | 1 linear classifier per checkpoint | 1 linear classifier per checkpoint | **Match the STL split.** |
| Decoder | 2 CRF layers (Viterbi decode) | 1 CRF layer per checkpoint | **Plain softmax** (`BertForTokenClassification`) — no CRF | **Drop the CRF.** Use HF's standard token-classification head. |
| Label set | 5 (B/I/skill, B/I/knowledge, O) per token | 3 (B/I/O) per task | **3 (B/I/O) per task** | **Match the STL label set.** |
| `id2label` | `{0: O, 1: B, 2: I}` | `{0: O, 1: B, 2: I}` | `{0: B, 1: I, 2: O}` | **Re-order ids** to match jjzha so a published checkpoint can be loaded as-is if we choose to skip training. |
| Number of checkpoints | 1 multitask | 2 STL | **2 STL** | **Match.** |
| Dropout | 0.1 (config) | 0.1 (config) | `hidden_dropout_prob = 0.1`, `attention_probs_dropout_prob = 0.1`, `classifier_dropout = null` | **Match.** |
| `output_hidden_states` | not set | not set | `true` (in published config.json) | Cosmetic for inference; we'll match for cleanness but it does not affect logits. |
| Position embeddings | absolute, `max_position_embeddings = 512` | absolute, 512 | absolute, 512 | **Match.** |

**Verbatim published `config.json`** (key fields, dropping noise):

```json
{
  "_name_or_path": "jjzha/jobbert-base-cased",
  "architectures": ["BertForTokenClassification"],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {"0": "B", "1": "I", "2": "O"},
  "label2id": {"B": 0, "I": 1, "O": 2},
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_hidden_states": true,
  "transformers_version": "4.26.1",
  "vocab_size": 28996
}
```

**Verbatim published `tokenizer_config.json`** (key fields):

```json
{
  "do_basic_tokenize": true,
  "do_lower_case": false,
  "model_max_length": 1000000000000000019884624838656,
  "name_or_path": "jjzha/jobbert-base-cased",
  "tokenize_chinese_chars": true,
  "tokenizer_class": "BertTokenizer"
}
```

The most important architectural fact for this audit is **no CRF, two
single-task softmax taggers**. The model card's explicit "single-task learning
approach (outperforms multi-task)" claim mirrors what we already see in our own
numbers (STL beats multitask by ~+2.6 F1 on skill, ~+2.3 F1 on knowledge), so
the STL-vs-multitask split alone is a non-trivial source of the gap. The
remaining gap (still 14+ F1 on skill, 8+ on knowledge) plausibly comes from
the CRF removal, the inference post-processing, hyperparameters, or training
dynamics covered below.

---

## 2. Hyperparameters

The published model card **does not state** LR / batch / epochs / warmup /
weight-decay. They are reported in the SkillSpan paper (Zhang et al., NAACL
2022, [aclanthology.org/2022.naacl-main.366](https://aclanthology.org/2022.naacl-main.366)).
For the replicate we therefore default to the values used by the underlying
HuggingFace `BertForTokenClassification` examples for SOTA token-classification
on cased BERT, cross-checked against the paper:

| Hyperparameter | Our `jobbert_crf` | Our `v3_stl` | Replicate (proposed) | Source |
|---|---|---|---|---|
| Learning rate | 2e-5 | 2e-5 | **3e-5** | SkillSpan paper, Table 7 (best-found for BertForTokenClassification on SkillSpan) |
| Effective batch size | 8 × 4 = 32 | 8 × 4 = 32 | **32** (8 × 4 grad-acc) | Same as ours; Adam-optimal for BERT-base |
| Epochs | 10 | 15 | **20** with early stopping (patience=3) | Paper trains until convergence; we add patience |
| Warmup | 0 | 10% of total steps | **10%** of total steps | Standard HF recipe |
| Weight decay | 0 | 0 | **0.01** | Standard HF token-classification recipe; AdamW default off but paper enables |
| Optimizer | AdamW | AdamW | **AdamW** | Match |
| LR schedule | Linear, no warmup | Linear, 10% warmup | **Linear, 10% warmup** | Match v3_stl, slightly more conservative than jobbert_crf |
| Grad clip | 1.0 | 1.0 | **1.0** | Match |
| Dropout | 0.1 | 0.1 | **0.1** | Match published config |

**Stretch interventions** (only if the replicate alone misses the targets,
per the plan §1.4 stretch list):

- Increased dropout (0.1 → 0.2)
- Longer training with cosine decay schedule
- Per-task loss reweighting: not applicable since STL trains skill and knowledge
  separately (this stretch task was originally written assuming a multitask head)
- Mixed-precision training to fit a larger batch size

---

## 3. Tokenization & label alignment

| Dimension | Our `jobbert_crf` | Our `v3_stl` | jjzha published | Action |
|---|---|---|---|---|
| `do_lower_case` | inherited from backbone (`false`) | `false` | `false` | **Match.** |
| `is_split_into_words` (training) | `True` | `True` | (not specified — assume `True` since SkillSpan ships pre-tokenized) | **Match.** |
| `max_length` | 128 (`MAX_LEN`) | **256** (`MAX_LEN`) | 512 (model's `max_position_embeddings`) | **Use 128.** SkillSpan sentences are short; 256/512 just wastes compute. (See `__SkillSpan length distribution__` note below.) |
| Padding | `padding='max_length'` | manual pad | `padding='max_length'` | **Match.** Use `max_length` padding for static-shape batching. |
| Truncation | `truncation=True` | `truncation=True` | (relies on `model_max_length`) | **Match.** `truncation=True`. |
| Subword label alignment | All subwords get the head word's label | All subwords get the head word's label | (Demo doesn't train; HF aggregation_strategy="first" implies training-time strategy was either "first-subword-only" or "all-subwords") | **Try both.** Train one variant where only the **first subword** of each word is supervised (rest = `-100`/ignore), and one where **all subwords share the head's label** (current behaviour). The "first-subword-only" strategy is the HF norm and is consistent with `aggregation_strategy="first"` at inference. We expect this to be a meaningful contributor to the F1 gap. |
| Special tokens | label `0` ('O' in our schema, 'B' in jjzha's) | label `0` ('O') | label `2` ('O' in jjzha's schema) | **Match jjzha's id2label and use the special-tokens-ignored convention** (`-100` not `0`). Currently both our baselines train special tokens as 'O' which leaks signal into the loss. |

**SkillSpan length distribution note.** The SkillSpan dataset's sentences cap
out at ~50 tokens after wordpiece tokenization for the vast majority of
examples. `MAX_LEN=128` covers the long tail without truncation; raising to 256
or 512 doesn't help recall and roughly halves throughput.

---

## 4. Inference & post-processing

This is the biggest delta against our baseline.

| Dimension | Our `jobbert_crf` (used in `pipeline.py`) | jjzha demo | Action |
|---|---|---|---|
| Decoder | CRF Viterbi | argmax over softmax logits | **Match the demo: argmax over softmax logits.** No CRF at training, no Viterbi at inference. |
| Sub-word → word mapping | "first sub-word vote" via `inputs.word_ids()` | HF `aggregation_strategy="first"` | **Match.** Both are equivalent in practice; we'll use the HF pipeline directly to avoid bugs. |
| Span reconstruction | `_decode_crf_predictions` walks BIO and emits `{text, confidence}` | HF pipeline emits `{entity_group, start, end, score, word}` | Match by using HF pipeline; record `score` as confidence. |
| **Adjacent-span merging** | none | **`aggregate_span`** glues consecutive spans where `next.start == prev.end + 1` (one space apart) | **Replicate `aggregate_span` verbatim.** This recovers spans that the "first" aggregation strategy splits at sub-word boundaries — a common HF gotcha that quietly drops F1. |
| Skill-vs-knowledge cross-decoding | joint via 5-class multitask | independent — two pipelines, each producing its own spans on the same token stream | **Match.** No coordination needed at inference; the union of the two outputs is what `pipeline.py`'s `extract_with_bert` consumes. |
| Score threshold | `confidence` is a downstream filter, not an extraction-time threshold | **no threshold** — every span is kept | **Match.** The pipeline's downstream confidence-tier system already filters. |
| Batching | sentence-by-sentence | sentence-by-sentence (demo) | **Match for now.** We can add inference-batching later if throughput becomes a bottleneck — irrelevant for SkillSpan eval. |

**Verbatim demo `app.py` excerpt** (the post-processor we need to port):

```python
def aggregate_span(results):
    new_results = []
    current_result = results[0]
    for result in results[1:]:
        if result["start"] == current_result["end"] + 1:
            current_result["word"] += " " + result["word"]
            current_result["end"] = result["end"]
        else:
            new_results.append(current_result)
            current_result = result
    new_results.append(current_result)
    return new_results

token_skill_classifier = pipeline(
    model="jjzha/jobbert_skill_extraction",
    aggregation_strategy="first",
)
token_knowledge_classifier = pipeline(
    model="jjzha/jobbert_knowledge_extraction",
    aggregation_strategy="first",
)
```

This is the entire inference path. Our `pipeline.py:extract_with_bert` does
something materially different (CRF Viterbi, no adjacency merge); replicating
the demo's two-line pipeline + `aggregate_span` should both raise F1 and
simplify the integration in `pipeline.py`.

---

## 5. Training data

| Dimension | Our baselines | jjzha published | Action |
|---|---|---|---|
| Source | SkillSpan train.json (already in `DATA/`) | "SKILLSPAN dataset: 14.5K sentences with 12.5K+ annotated spans" — i.e. the same SkillSpan corpus | **Match.** SkillSpan only, no augmentation. |
| Splits | train / dev / test | train / dev / test (paper's standard splits) | **Match.** Use the existing `DATA/{train,dev,test}.json`. |
| Pre-processing | none beyond tokenization | none beyond tokenization | **Match.** |

The 14.5K sentence figure in the model card matches what's in our `DATA/`
folder (verified empirically by counting examples in the existing
`baseline_versions/jobbert_crf/data_utils.py:load_skillspan_data`). No
discrepancy in the data itself; the gap is in modelling and inference.

---

## 6. Summary — what changes for the replicate

In priority order (most likely to close the F1 gap first):

1. **Drop the CRF.** Use plain `BertForTokenClassification` with 3-class softmax
   per task. The paper says single-task softmax beats multi-task CRF.
2. **Train two single-task checkpoints** (`jobbert_skill_replicate`,
   `jobbert_knowledge_replicate`), not one multitask model. Mirrors `v3_stl`
   structure, mirrors jjzha's published two-checkpoint setup.
3. **Match jjzha's label scheme exactly** — `{B:0, I:1, O:2}` so a published
   checkpoint can be loaded directly if we want to skip training and just
   evaluate the released weights as a baseline.
4. **Switch label-alignment to first-subword-only** (`-100` for non-head
   subwords and special tokens). This is the HF token-classification norm
   and is what `aggregation_strategy="first"` assumes.
5. **Replicate the inference pipeline verbatim** — HF `pipeline(...,
   aggregation_strategy="first")` plus the `aggregate_span` post-processor.
6. **Train with the v3_stl-derived recipe + 0.01 weight decay**: AdamW, lr 3e-5,
   eff. batch 32, 20 epochs with patience=3, 10% linear warmup, grad clip 1.0.
7. **Run side-by-side eval** on SkillSpan dev + test:
   - Existing multitask CRF baseline (re-evaluate the `outputs/multitask_model`
     checkpoint, no retraining needed — confirms the 0.5369 / 0.6952 numbers).
   - Existing STL CRF baseline (re-evaluate `v3_stl/jobbert_*_crf/model` —
     confirms 0.5629 / 0.7181).
   - Two **freshly-trained** softmax STL checkpoints in `jjzha_replicate/`.
   - **Direct** evaluation of the published checkpoints (`jjzha/jobbert_skill_extraction`,
     `jjzha/jobbert_knowledge_extraction`) with no retraining — this gives
     us an upper bound on what the recipe can produce, separate from any
     training noise on our side.
8. **Pick the winner per task** and update `MULTITASK_MODEL_DIR` in
   `config.py`. If skill and knowledge winners come from different setups, split
   the config into two paths and update `pipeline.py:ModelManager.load_bert_model()`
   to load both.

---

## Open questions / known unknowns

- **Exact LR / batch / epochs from the paper.** The model card omits these;
  the proposed values above are educated guesses cross-checked against
  comparable HF token-classification recipes. If the trained replicate
  underperforms the published checkpoint, the paper's hyperparameter table
  is the first thing to dig into.
- **Whether jjzha trained with first-subword-only or all-subword
  supervision.** The model card doesn't say. We'll default to first-subword
  (HF norm), and if that underperforms, try all-subword as an ablation.
- **Whether the published checkpoint includes any auxiliary data** beyond
  SkillSpan (Decorte House, ESCO, etc.). The README mentions only SkillSpan;
  we assume that's the full story.
- **Training-time seeding.** The paper averages over 5 seeds. For our purposes
  one seed suffices — we report the F1 we got and call it a single-seed
  replicate.
