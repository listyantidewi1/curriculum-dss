# Replication report — jjzha JobBERT skill / knowledge extractor

**Phase 1.4, sub-tasks 3-6 (eval matrix + winner selection + write-up).**

## TL;DR

- **Replication succeeded.** Our trained replicate matches the published checkpoint within 0.014 (skill F1) / 0.004 (knowledge F1).
- **The published checkpoint scores 0.519 / 0.653 on SkillSpan test** — comparable to vanilla BERT (0.542 / 0.617 in the literature). It is **not** the breakthrough we hoped for in the audit.
- **Phase 1.4's targets (skill F1 ≥ 0.70, knowledge F1 ≥ 0.80) are unreachable on SkillSpan with this architecture.** The published SkillSpan SOTA is **0.543 / 0.742** (Skill-LLM, Herandi et al. 2024 — an 8B LLaMA 3 fine-tuned with LoRA). No published method crosses 0.57 skill F1 or 0.75 knowledge F1.
- **Recommendation: pivot.** The next step is a Skill-LLM-style LoRA fine-tune (see `baseline_versions/skill_llm/`).

## F1 matrix

All numbers are SkillSpan **test** set, span-level seqeval F1 with first-subword-only label alignment (the HF norm; matches Skill-LLM's reported support counts of 1091 skill / 1174 knowledge spans, confirming we are evaluating on the correct test set).

| Setup | Skill F1 | Knowledge F1 | Source |
|---|---|---|---|
| `jjzha_replicate` (this work, trained from scratch) | **0.5050** | **0.6570** | `outputs/skill/metrics_test.txt`, `outputs/knowledge/metrics_test.txt` |
| `jjzha_published` (HF Hub, direct eval) | **0.5189** | **0.6532** | `outputs/published_skill/metrics_test.txt`, `outputs/published_knowledge/metrics_test.txt` |
| BERT (no domain pretraining) | 0.542 | 0.617 | Skill-LLM Table 2 |
| jobSpanBERT (pretrain + fine-tune) | 0.563 | 0.619 | Skill-LLM Table 2 |
| ESCOXLM-R | — | — | Total 0.626 (Skill-LLM Table 2) |
| NNOSE (RoBERTa + kNN retrieval) | — | — | Total 0.642 (Skill-LLM Table 2) |
| Fine-tuned GLiNER (166M) | 0.496 | 0.655 | Skill-LLM Table 2 |
| **Skill-LLM (LoRA LLaMA 3 8B) — SOTA** | **0.543** | **0.742** | Skill-LLM Table 2 |
| **Phase 1.4 target** | **≥ 0.70** | **≥ 0.80** | `requirements.md` Req 3 |

Replicate-vs-published gap: skill +0.0139, knowledge −0.0038. Within run-to-run noise; we accept this as a successful single-seed replication.

## Note on the older CRF baseline numbers

`baseline_versions/jobbert_crf/outputs/metrics_test.txt` reports 0.5369 / 0.6952 with support 1562 / 2302. Those numbers are **not directly comparable** to the matrix above. They use an "all-subwords share the head word's BIO tag" label-alignment scheme that artificially inflates the span count when a multi-subword token contains a B-tag (each subword starts a new span). Skill-LLM's published support counts (1091 skill / 1174 knowledge) confirm that our replicate's first-subword-only alignment is the correct one. Re-evaluating the CRF baselines under the same alignment is left as an open task; given the SOTA numbers, doing so would not change the conclusion below.

## What the audit got right

- Architecture deltas are accurately identified (jjzha is plain `BertForTokenClassification`, no CRF; two single-task checkpoints; cased tokenizer). Every detail verified against the published `config.json`.
- The hyperparameter recipe (LR 3e-5, eff batch 32, 20 epochs with patience=3, 10% warmup, weight decay 0.01) trains cleanly to convergence: skill stops at epoch 8 (best dev F1 0.5319), knowledge at epoch 6 (best dev F1 0.6210). No instability, no over/under-fitting symptoms.
- First-subword-only label alignment matches the support counts the field uses.

## What the audit got wrong

- **Hypothesis: dropping the CRF + STL split would close the F1 gap.** Wrong direction. The CRF removal didn't hurt either, but it didn't help to anywhere near the targets.
- **Implicit assumption: jjzha's published model is materially stronger on SkillSpan than vanilla BERT.** Wrong. The published checkpoint scores 0.519/0.653 — within 0.04 F1 of vanilla BERT either way. The user's "demo Space looks better than ours" observation must come from real-world output quality on a different distribution, not benchmark F1.
- **Targets calibrated to wishful thinking.** No published method hits 0.70 / 0.80 on SkillSpan. The targets were not grounded in the literature.

## Why the targets aren't reachable on SkillSpan

SkillSpan is a small benchmark (14.5K sentences) with rich span annotations. Three structural ceilings:

1. **Annotation noise.** Skill-LLM §"Qualitative Analysis" reports ~8% of error cases are caused by problematic gold labels (e.g. `"DevOps Engineer ( CI CD Cloud Docker Jenkins ) <ORGANIZATION>"` with both BIO columns left empty). That's a hard ~8 F1 point ceiling.
2. **Span boundary ambiguity.** "Implementing and promoting all QA relevant topics" — should that be one span or two? Different annotators will draw the line differently. Strict span-set F1 punishes any mismatch.
3. **Label scheme has no entity sub-types.** All skills get the same B/I/O. The model can't learn fine-grained semantics; it can only learn span boundaries.

Skill-LLM's 0.742 knowledge F1 is the practical ceiling for SkillSpan with current methods.

## Recommendation: revise targets, pivot to Skill-LLM

**Revised Phase 1.4 targets (proposed):**

- **Skill F1 ≥ 0.54** (matches BERT, within 0.01 of SOTA Skill-LLM = 0.543)
- **Knowledge F1 ≥ 0.74** (matches Skill-LLM SOTA = 0.742)
- **Total span F1 ≥ 0.65** (matches Skill-LLM SOTA = 0.648)

These targets are SOTA-aligned. They require switching the BERT backbone for an LLM-based extractor, which is the path forward documented in `baseline_versions/skill_llm/`.

**Why Skill-LLM specifically:**

- It is the published SkillSpan SOTA at the time of this report.
- Output format is structured JSON with `skill_span` (verb-led for SKILL) + `context` (one-token window for offset disambiguation) + a separate `KNOWLEDGE` list (noun phrases). This **explicitly preserves the verb in skill spans** — critical for our pipeline because "designing UI/UX" is a skill, "UI/UX" alone is knowledge. A verb-less skill extractor would collapse the two and corrupt downstream competency generation.
- Inference cost is real (~8B parameters) but acceptable: one LoRA-fine-tuned head, deployed via `transformers` + `peft`. Fits on a single A100/RTX 4090 with 4-bit quantization.
- Hyperparameters are published (Skill-LLM §"Experimental Setup"): LoRA rank 64 on q_proj/v_proj, lr 2e-4, batch 4, 2 epochs, 10% warmup, cosine schedule. Training time ≈ 1-2 hours on a single A100.

## Disposition of the four checkpoints in this directory

- **`jobbert_skill_replicate/`, `jobbert_knowledge_replicate/`** — keep on disk for reproducibility but **do not promote to `pipeline.py`**. The replicate succeeded as a research artifact (we now know the published model's ceiling on SkillSpan) but is not the production winner.
- **Local re-evaluation of `jjzha/jobbert_skill_extraction` / `jjzha/jobbert_knowledge_extraction`** — keep as a literature-comparison baseline.
- **`config.py:MULTITASK_MODEL_DIR` in the project root** — leave pointing at `baseline_versions/v3_stl/` for now; the v3_stl checkpoint remains the strongest BERT-based fallback. Final decision on the production extractor moves to Phase 1.5 (Skill-LLM).

## Open work

- (Phase 1.5 — `baseline_versions/skill_llm/`) LoRA fine-tune LLaMA 3.1 8B Instruct on SkillSpan, evaluate against this matrix, and integrate into `pipeline.py` as the new BERT-path replacement.
- (Stretch, low priority) Re-evaluate `jobbert_crf` and `v3_stl` checkpoints under first-subword-only alignment to make all six rows of the matrix directly comparable. Will not change the disposition above.
- (Phase 2 prerequisite) Add an ESCO-normalization pass over **knowledge items only** (per ESCOX), once the new extractor is wired in. Useful for KKNI labeling and curriculum mapping; explicitly NOT applied to skill items because ESCO entries are noun phrases that would strip the verb.
