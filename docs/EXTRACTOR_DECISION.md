# Layer-1 Extractor Decision

**Decision:** Skill-LLM (LoRA-fine-tuned LLaMA 3.1 8B Instruct) is the production Layer-1 extractor.

**Date:** 2026-05-12. Preliminary — finalized once Llama 3.1 70B and DeepSeek-V3.2 zero-shot evals complete (in flight). Decision is robust to those remaining results: the F1 gap between Skill-LLM and any zero-shot API is large enough (≥ 0.25 on the two measured candidates) that the remaining models would need to materially outperform their peers to change the outcome.

**Architecture context:** the pipeline uses a hybrid two-layer extractor (`.kiro/specs/pipeline-redesign-v2/requirements.md` Req 3.6 + Req 9.1). Layer 1 is sentence-level. Layer 2 is the full-posting LLM (DeepSeek-V3 via OpenRouter — unchanged by this decision). Production extraction = Layer 1 + Layer 2 + SBERT fusion. This document settles **Layer 1 only.**

---

## Results table (SkillSpan test, n = 3,569)

All metrics are strict span-set F1 per the Skill-LLM paper Table 2 definition.

| Candidate | Total F1 | Skill F1 | Knowledge F1 | verb_short_rate | parse_failures | Gate 1 |
|---|---|---|---|---|---|---|
| **Skill-LLM 8B LoRA (Kaggle)** | **0.6485** | **0.5810** | **0.7065** | 0.1644 ✓ | 1 / 3569 ✓ | ✅ **PASS** |
| GPT-4o-mini (zero-shot via OpenRouter) | 0.3958 | 0.2206 | 0.4921 | 0.1002 ✓ | 0 / 3569 ✓ | ❌ Total F1 < 0.55 |
| Claude Haiku 3.5 (zero-shot via OpenRouter) | 0.3594 | 0.3091 | 0.4035 | **0.3360** ✗ | **481 / 3569** ✗ | ❌ FAIL (3 of 5 sub-gates) |
| Llama 3.1 70B (zero-shot via OpenRouter) | _pending_ | | | | | _pending_ |
| DeepSeek-V3.2 (zero-shot via OpenRouter) | _pending_ | | | | | _pending_ |
| jjzha JobBERT (Phase 1.4 audit, for reference) | n/a | 0.5190 | 0.6531 | n/a | n/a | retired |

### Reference points

- **Skill-LLM paper SOTA** (Herandi et al. 2024 AAAI 2025): skill F1 0.543 / knowledge F1 0.742 / total F1 ≈ 0.66.
- **Our replication** (this run): skill F1 0.5810 / knowledge F1 0.7065 / total F1 0.6485. Slightly above the paper on skill, slightly below on knowledge — within reasonable replication variance, and **above the published SOTA on the harder skill axis**.

---

## Gate evaluation

Gate 1 from the v2 plan:

```
total_F1 ≥ 0.55
AND verb_short_rate ≤ 0.244     (baseline 0.144 + 0.10 tolerance)
AND parse_failure_rate ≤ 0.02
```

Per-candidate breakdown:

### Skill-LLM 8B LoRA ✅ ALL PASS
- total F1 = 0.6485 ≥ 0.55 ✓ (with substantial headroom)
- verb_short_rate = 0.1644 ≤ 0.244 ✓ (preserves verb-noun distinction)
- parse_failure_rate = 1/3569 = 0.028% ≤ 2% ✓
- Bonus: matches or beats published SOTA on skill F1

### GPT-4o-mini ❌ FAIL
- total F1 = 0.3958 < 0.55 — **fails primary quality gate by 0.15**
- verb_short_rate ✓, parse ✓
- The model is well-behaved (no parse failures, good verb preservation) but materially under-extracts skills (recall 0.175 vs Skill-LLM's 0.579). Zero-shot is insufficient for this task.

### Claude Haiku 3.5 ❌ FAIL (multi-sub-gate)
- total F1 = 0.3594 < 0.55 — fails primary gate
- **verb_short_rate = 0.3360 > 0.244** — collapses many verb-led actions to bare nouns; would corrupt downstream KKNI/competency stages
- **parse_failure_rate = 481/3569 = 13.5% > 2%** — JSON output is unreliable
- Three independent sub-gate failures. Not viable for production.

### Llama 3.1 70B and DeepSeek-V3.2 (pending)
Will be measured for completeness in the academic comparison table. Even if either reaches total F1 = 0.55, Skill-LLM at 0.6485 leads by ≥ 0.10 → per the v2 decision rule (`If best API < Skill-LLM − 0.05, ship Skill-LLM`), the decision is the same.

---

## Decision rule (from the v2 plan)

| Outcome | Action | Triggered? |
|---|---|---|
| API zero-shot ≥ Skill-LLM | Ship the API model | No (best measured API is 0.25 below) |
| API close to Skill-LLM (within 0.03) | Probably ship API | No |
| Skill-LLM materially better (≥ 0.05 over best API) | **Ship Skill-LLM via Kaggle batch or HF Inference Endpoint** | **Yes — by ≥ 0.25** |
| Two-stage (recall + verify) materially better | Ship two-stage | Not measured; not pursuing given the deadline |
| Nothing passes Gate 1 | Halt | No (Skill-LLM passes cleanly) |

**Action triggered: Ship Skill-LLM 8B LoRA as Layer 1.**

---

## Deployment implications

### Where Skill-LLM inference runs

**Short-term (during the v2 sprint, through 2026-06-08):**
- **Kaggle Save Version batch.** 30h/week free GPU quota. Sufficient for the stability experiment (Section RQ2b: N ∈ {500, 1000, 2500, 5000, 10000} × 3 seeds) plus weekly production batches up to ~50,000 postings/week.
- Workflow: upload job postings as a Kaggle Dataset → run `run_eval_only_on_kaggle.py`-style script that loads the LoRA adapter from the saved-version output and extracts on new jobs → download `extracted_skills.csv` + `extracted_knowledge.csv` → feed into local pipeline.

**Medium-term (post-deadline, periodic use):**
- **Vast.ai Tier 2 (RTX A6000 48 GB) spot instance**, ~$0.50/hr. Annual cost ~$26 at one weekly batch run. Or **HuggingFace Inference Endpoints** if managed infrastructure is preferred (~$0.60/hr T4 backend).

**Long-term (if user base or extraction volume grows):**
- Scheduled cron on a Vast.ai / RunPod A100 instance; spin up, batch process, spin down.

### What does NOT change

- **Layer 2** (full-posting LLM): DeepSeek-V3 via OpenRouter, exactly as today. Unchanged by this decision.
- **All downstream stages** (clustering, competency gen, KKNI labeler, etc.): unchanged.
- **Provenance invariant**: every extracted item carries `(job_id, sentence_id, sentence_text, extractor_source="skill_llm" | "llm_full_posting")`.

### What the api_zero_shot baselines are kept for

- **Paper baseline comparison** in RQ1. The metrics_test_*.txt files document that zero-shot underperforms fine-tuning by a wide margin (~0.25 F1) on this task, which is itself a publishable finding.
- **Fallback option** if Skill-LLM deployment becomes operationally infeasible (unlikely given Kaggle batch viability).
- **Reference numbers** if someone wants to re-evaluate with newer API models in the future.

---

## Replication evidence

- Skill-LLM training script: `baseline_versions/skill_llm/kaggle/run_on_kaggle.py`
- Skill-LLM eval-only script (post-training inference on adapter): `baseline_versions/skill_llm/kaggle/run_eval_only_on_kaggle.py`
- Skill-LLM raw outputs: `baseline_versions/skill_llm/outputs/trained/raw_outputs_test.jsonl` (3,569 examples) + same for dev
- API zero-shot script: `baseline_versions/api_zero_shot/eval.py`
- API raw outputs: `baseline_versions/api_zero_shot/outputs/raw_outputs_test_*.jsonl`

All evals use the SkillSpan test split (3,569 examples) loaded from `DATA/test.json`. Same prompt boundary token (`**`), same JSON output schema, same span-matching offset logic, same verb-preservation diagnostic. Apples-to-apples by construction.

---

## Open follow-ups (post-decision)

These do not block the decision but should be tracked:

1. **Finish Llama 70B and DeepSeek-V3.2 evals** for the paper's complete comparison table. ETA ~2h from 2026-05-12 morning (running now in the background).
2. **Integrate Skill-LLM into `pipeline.py`** as the active Layer-1 extractor. Replace the current `--extraction-mode hybrid` BERT path with a Skill-LLM-backed extractor. Estimate: 2–3 days.
3. **Stability experiment** (RQ2b): determine N\* across N ∈ {500, 1000, 2500, 5000, 10000} with 3 seeds. Estimate: 2–3 days wall-clock + 0.5 day analysis. Output figure: Jaccard top-20 vs N with ±std bars.
4. **Update v2 spec Req 3** to reflect the realized gate (total F1 ≥ 0.55 + verb-preservation + parse-failures), and to name Skill-LLM as the chosen Layer 1 (with API zero-shot retained as ablation comparison). _Done in this commit._
5. **Update `docs/PIPELINE_DIAGRAM.md`** to confirm Skill-LLM as the BERT-path label (already correct in current draft, but worth verifying). _Done in this commit._

---

## Audit trail

- 2026-05-11: Skill-LLM training on Kaggle T4 — ~2h training + adapter saved, eval crashed (BatchEncoding bug, fixed in commit `be193fa`).
- 2026-05-11/12: Skill-LLM eval-only resume run on Kaggle — full test + dev splits, completed overnight. Output downloaded 2026-05-12 morning.
- 2026-05-11: GPT-4o-mini full test split eval, ~1h wall-clock, ~$0.40 OpenRouter cost.
- 2026-05-11: Claude Haiku 3.5 full test split eval, ~1.5h wall-clock, ~$1.80 OpenRouter cost.
- 2026-05-12: Llama 3.1 70B and DeepSeek-V3.2 full test evals launched in background, pending completion.
