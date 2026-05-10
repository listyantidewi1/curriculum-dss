"""
eval.py — skill_llm

Evaluate a Skill-LLM-style fine-tuned checkpoint on SkillSpan dev / test.
Reports:

1. **Span-set F1** for SKILL and KNOWLEDGE separately, and their micro-average,
   following the strict span-equality definition used in the Skill-LLM paper
   (Table 2). A predicted span counts as correct iff it matches a gold span
   exactly on (skill_span text, character offsets in source sentence).

2. **Verb-preservation diagnostic** — fraction of predicted SKILL spans that
   have fewer than VERB_PRESERVATION_MIN_TOKENS tokens. This catches the
   failure mode where the model collapses verb-led action ("designing UI/UX")
   to noun head ("UI/UX") and emits the noun under SKILL. The threshold
   VERB_FAILURE_RATE_MAX is the CI gate; exceeding it indicates the fine-tune
   has degraded into a noun-only extractor and the resulting model should NOT
   be promoted to pipeline.py.

Usage:
    # against the LoRA adapter we just trained (default)
    python eval.py

    # against an arbitrary HF model id (e.g. a published checkpoint)
    python eval.py --base-model meta-llama/Llama-3.1-8B-Instruct \
                   --adapter ""        # disable adapter, eval base model only
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    DATASETS_DIR,
    HF_CACHE_DIR,
    INFERENCE_DO_SAMPLE,
    INFERENCE_MAX_NEW_TOKENS,
    INFERENCE_TEMPERATURE,
    OUTPUTS_DIR,
    SENTENCE_BOUNDARY_TOKEN,
    SYSTEM_PROMPT,
    USE_4BIT,
    VERB_FAILURE_RATE_MAX,
    VERB_PRESERVATION_MIN_TOKENS,
)


os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))


# --------------------------------------------------------------------------- #
# JSON output parsing
# --------------------------------------------------------------------------- #

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def parse_assistant_json(text: str) -> Optional[dict]:
    """Robust parse of the assistant's JSON output. Returns None if no parseable
    object is found. Skill-LLM §"Qualitative Analysis" reports this happens in
    ~1/3174 dev cases on their fine-tune; we match that order of magnitude."""
    text = text.strip()
    # First try the whole thing.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first {...} block.
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Span resolution: map predicted skill_span back to character offsets via context
# --------------------------------------------------------------------------- #


def find_span_offsets(
    sentence: str, skill_span: str, context: str
) -> Optional[Tuple[int, int]]:
    """Locate (start, end_exclusive) of skill_span within sentence, using the
    context window to disambiguate when skill_span appears multiple times.

    Strategy: find the context substring first, then locate skill_span within it.
    Falls back to the first occurrence of skill_span if context isn't found
    (paraphrased context = LLM didn't quote verbatim — rare on a fine-tuned model).
    """
    if not skill_span:
        return None
    sent_l = sentence.lower()
    sk_l = skill_span.lower()
    ctx_l = (context or "").lower()

    if ctx_l and ctx_l in sent_l:
        ctx_start = sent_l.index(ctx_l)
        within = ctx_l
        rel = within.find(sk_l)
        if rel != -1:
            start = ctx_start + rel
            return (start, start + len(skill_span))

    if sk_l in sent_l:
        start = sent_l.index(sk_l)
        return (start, start + len(skill_span))
    return None


def gold_spans_from_bio(
    tokens: List[str], tags: List[str]
) -> Set[Tuple[int, int]]:
    """Reduce a token-level BIO sequence to a set of (start_char, end_char_exclusive)
    spans. Char offsets are computed against `" ".join(tokens)` — the same
    sentence representation we send to the model in prepare_data.py."""
    spans: Set[Tuple[int, int]] = set()
    cursor = 0
    span_start = None
    span_end = None
    for tok, tag in zip(tokens, tags):
        tok_start = cursor
        tok_end = cursor + len(tok)
        cursor = tok_end + 1  # +1 for the joining space (skipped after last)
        if tag == "B":
            if span_start is not None:
                spans.add((span_start, span_end))
            span_start, span_end = tok_start, tok_end
        elif tag == "I" and span_start is not None:
            span_end = tok_end
        else:
            if span_start is not None:
                spans.add((span_start, span_end))
                span_start, span_end = None, None
    if span_start is not None:
        spans.add((span_start, span_end))
    return spans


def f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def build_quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


def load_model(base_model_name: str, adapter_path: Optional[str]):
    print(f"[INFO] base model: {base_model_name}")
    if adapter_path:
        print(f"[INFO] LoRA adapter: {adapter_path}")
    else:
        print("[INFO] no adapter — evaluating base model directly")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, cache_dir=HF_CACHE_DIR, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    kwargs = dict(
        cache_dir=HF_CACHE_DIR,
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        device_map="auto",
    )
    if USE_4BIT:
        kwargs["quantization_config"] = build_quantisation_config()
    base = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def build_user_message(tokens: List[str]) -> str:
    sentence = " ".join(tokens)
    return f"{SENTENCE_BOUNDARY_TOKEN} {sentence} {SENTENCE_BOUNDARY_TOKEN}"


@torch.no_grad()
def generate(model, tokenizer, tokens: List[str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(tokens)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
        do_sample=INFERENCE_DO_SAMPLE,
        temperature=INFERENCE_TEMPERATURE if INFERENCE_DO_SAMPLE else 1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, inputs.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #


def load_split(split: str) -> List[dict]:
    """Load the chat-format JSONL produced by prepare_data.py and re-attach
    the original tokens / tags from the SkillSpan source so we can score."""
    from config import DATA_DIR  # local import to avoid circular cycles

    src_path = DATA_DIR / f"{split}.json"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} not found")
    raw = json.loads(src_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                raw = v
                break
    return raw


def evaluate_split(
    model, tokenizer, split: str, output_dir: Path, label: str
) -> Dict[str, float]:
    examples = load_split(split)
    print(f"[INFO] evaluating {split}: {len(examples)} examples")

    skill_tp = skill_fp = skill_fn = 0
    knowl_tp = knowl_fp = knowl_fn = 0
    n_skill_pred = 0
    n_skill_short = 0
    n_parse_failures = 0

    raw_outputs: List[dict] = []

    for ex in tqdm(examples, desc=split):
        tokens = ex["tokens"]
        gold_skill = gold_spans_from_bio(tokens, ex.get("tags_skill", []))
        gold_knowl = gold_spans_from_bio(tokens, ex.get("tags_knowledge", []))
        sentence = " ".join(tokens)

        raw_text = generate(model, tokenizer, tokens)
        parsed = parse_assistant_json(raw_text)

        pred_skill: Set[Tuple[int, int]] = set()
        pred_knowl: Set[Tuple[int, int]] = set()
        if parsed is None:
            n_parse_failures += 1
        else:
            for item in parsed.get("SKILL", []) or []:
                if not isinstance(item, dict):
                    continue
                span_text = str(item.get("skill_span", "")).strip()
                ctx = str(item.get("context", "")).strip()
                offsets = find_span_offsets(sentence, span_text, ctx)
                if offsets:
                    pred_skill.add(offsets)
                # Verb-preservation diagnostic
                if span_text:
                    n_skill_pred += 1
                    if len(span_text.split()) < VERB_PRESERVATION_MIN_TOKENS:
                        n_skill_short += 1
            for item in parsed.get("KNOWLEDGE", []) or []:
                if not isinstance(item, dict):
                    continue
                span_text = str(item.get("skill_span", "")).strip()
                ctx = str(item.get("context", "")).strip()
                offsets = find_span_offsets(sentence, span_text, ctx)
                if offsets:
                    pred_knowl.add(offsets)

        skill_tp += len(pred_skill & gold_skill)
        skill_fp += len(pred_skill - gold_skill)
        skill_fn += len(gold_skill - pred_skill)
        knowl_tp += len(pred_knowl & gold_knowl)
        knowl_fp += len(pred_knowl - gold_knowl)
        knowl_fn += len(gold_knowl - pred_knowl)

        raw_outputs.append({
            "tokens": tokens,
            "raw_output": raw_text,
            "parsed": parsed,
        })

    s_p, s_r, s_f1 = f1(skill_tp, skill_fp, skill_fn)
    k_p, k_r, k_f1 = f1(knowl_tp, knowl_fp, knowl_fn)
    total_tp = skill_tp + knowl_tp
    total_fp = skill_fp + knowl_fp
    total_fn = skill_fn + knowl_fn
    t_p, t_r, t_f1 = f1(total_tp, total_fp, total_fn)

    verb_fail_rate = (n_skill_short / n_skill_pred) if n_skill_pred else 0.0

    # Write metrics + raw outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"metrics_{split}.txt", "w", encoding="utf-8") as f:
        f.write(f"=== {split.upper()} METRICS — {label} ===\n")
        f.write("# strict span-set F1 (Skill-LLM Table 2 definition)\n\n")
        f.write(f"skill_precision: {s_p:.4f}\n")
        f.write(f"skill_recall:    {s_r:.4f}\n")
        f.write(f"skill_f1:        {s_f1:.4f}\n")
        f.write(f"knowledge_precision: {k_p:.4f}\n")
        f.write(f"knowledge_recall:    {k_r:.4f}\n")
        f.write(f"knowledge_f1:        {k_f1:.4f}\n")
        f.write(f"total_precision: {t_p:.4f}\n")
        f.write(f"total_recall:    {t_r:.4f}\n")
        f.write(f"total_f1:        {t_f1:.4f}\n\n")
        f.write("--- diagnostics ---\n")
        f.write(f"json_parse_failures: {n_parse_failures} / {len(examples)}\n")
        f.write(f"skill_predictions:   {n_skill_pred}\n")
        f.write(
            f"skill_short_spans:   {n_skill_short} "
            f"(< {VERB_PRESERVATION_MIN_TOKENS} tokens)\n"
        )
        f.write(f"verb_failure_rate:   {verb_fail_rate:.4f}\n")
        f.write(f"verb_failure_max:    {VERB_FAILURE_RATE_MAX:.4f} (CI gate)\n")
        if verb_fail_rate > VERB_FAILURE_RATE_MAX:
            f.write(
                "WARN: verb-preservation diagnostic FAILED. The model is "
                "emitting too many noun-only SKILL spans. Do NOT promote this "
                "checkpoint to pipeline.py.\n"
            )

    with open(output_dir / f"raw_outputs_{split}.jsonl", "w", encoding="utf-8") as f:
        for rec in raw_outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[INFO] {split}: skill F1={s_f1:.4f} | knowledge F1={k_f1:.4f} "
        f"| total F1={t_f1:.4f} | verb_failure={verb_fail_rate:.1%}"
    )
    if verb_fail_rate > VERB_FAILURE_RATE_MAX:
        print(
            f"[WARN] verb-preservation diagnostic FAILED ({verb_fail_rate:.1%} > "
            f"{VERB_FAILURE_RATE_MAX:.1%}). See AUDIT.md."
        )

    return {
        "skill_f1": s_f1,
        "knowledge_f1": k_f1,
        "total_f1": t_f1,
        "verb_failure_rate": verb_fail_rate,
        "json_parse_failures": n_parse_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Skill-LLM checkpoint on SkillSpan.")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL_NAME)
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(ADAPTER_DIR),
        help="Path to LoRA adapter directory. Pass empty string to evaluate the base model only.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["dev", "test"],
        choices=["dev", "test"],
    )
    parser.add_argument("--output-subdir", type=str, default=None)
    args = parser.parse_args()

    adapter = args.adapter if args.adapter else None
    if adapter and not Path(adapter).exists():
        raise FileNotFoundError(f"adapter path {adapter} does not exist")

    label = f"adapter:{adapter}" if adapter else f"base:{args.base_model}"
    out_subdir = args.output_subdir or ("trained" if adapter else "base_only")
    output_dir = OUTPUTS_DIR / out_subdir

    model, tokenizer = load_model(args.base_model, adapter)

    for split in args.splits:
        evaluate_split(model, tokenizer, split, output_dir, label)


if __name__ == "__main__":
    main()
