"""
eval.py — api_zero_shot

Run SkillSpan extraction against an OpenRouter-hosted instruction-tuned LLM
and measure strict span-set F1 + the verb-preservation diagnostic.

Apples-to-apples with baseline_versions/skill_llm/eval.py:
- Same SkillSpan splits (DATA/dev.json, DATA/test.json)
- Same JSON output schema (SKILL/KNOWLEDGE arrays of {skill_span, context})
- Same strict span-set F1 metric (Skill-LLM Table 2 definition)
- Same verb-preservation gate (baseline 14.4% + 0.10 tolerance)
- Output written in the same metrics_<split>.txt format

Usage:
    # Zero-shot, full test split
    python eval.py --model openai/gpt-4o-mini --split test

    # Quick smoke test on first 50 examples
    python eval.py --model openai/gpt-4o-mini --split test --sample 50

    # Few-shot with 3 examples drawn deterministically from training set
    python eval.py --model anthropic/claude-3.5-haiku --split test --few-shot 3

    # Compare multiple models -- just run the script multiple times; each
    # writes to outputs/metrics_<split>_<model-slug>.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI
from tqdm import tqdm

from config import (
    DATA_DIR,
    DEFAULT_TRAIN_SHORT_RATE,
    MAX_RETRIES,
    MAX_TOKENS,
    OPENROUTER_BASE_URL,
    OPENROUTER_KEY_FILES,
    OUTPUTS_DIR,
    RANDOM_SEED,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE,
    SENTENCE_BOUNDARY_TOKEN,
    SUPPORTED_MODELS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    VERB_FAILURE_TOLERANCE_DELTA,
    VERB_PRESERVATION_MIN_TOKENS,
)


# --------------------------------------------------------------------------- #
# Client + retry
# --------------------------------------------------------------------------- #


def load_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        for key_file in OPENROUTER_KEY_FILES:
            if key_file.exists():
                api_key = key_file.read_text(encoding="utf-8").strip()
                if api_key:
                    print(f"[INFO] loaded OpenRouter API key from {key_file.name}")
                    break
        if not api_key:
            paths = ", ".join(str(p) for p in OPENROUTER_KEY_FILES)
            raise FileNotFoundError(
                f"OPENROUTER_API_KEY env var unset and no key file found at: {paths}"
            )
    if not api_key.startswith("sk-or-"):
        sys.stderr.write(
            f"[WARN] loaded key does not start with 'sk-or-' (OpenRouter prefix). "
            f"If you see 401 errors, check that the key file contains a valid "
            f"OpenRouter key (https://openrouter.ai/keys), not a Jatevo / OpenAI / etc. key.\n"
        )
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def call_with_retry(client: OpenAI, model: str, messages: List[dict]) -> Optional[str]:
    """Call the chat-completions endpoint with retry/backoff. Returns the raw
    string content of the assistant message, or None on terminal failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                timeout=REQUEST_TIMEOUT,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                time.sleep(wait)
    sys.stderr.write(f"[WARN] terminal API failure after {MAX_RETRIES + 1} attempts: {last_exc}\n")
    return None


# --------------------------------------------------------------------------- #
# SkillSpan loading + helpers (mirrors skill_llm/eval.py for self-containment)
# --------------------------------------------------------------------------- #


def load_skillspan(data_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for split, fname in (("train", "train.json"), ("dev", "dev.json"), ("test", "test.json")):
        path = data_dir / fname
        if not path.exists():
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                out[split] = content
            elif isinstance(content, dict):
                for v in content.values():
                    if isinstance(v, list):
                        out[split] = v
                        break
        except json.JSONDecodeError:
            data: List[dict] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data.append(json.loads(line))
            out[split] = data
        if split in out:
            print(f"[INFO] {split}: {len(out[split])} examples")
    return out


def build_user_message(tokens: List[str]) -> str:
    sentence = " ".join(tokens)
    return f"{SENTENCE_BOUNDARY_TOKEN} {sentence} {SENTENCE_BOUNDARY_TOKEN}"


def bio_to_spans(tags: List[str]) -> List[Tuple[int, int]]:
    spans, start = [], None
    for i, tag in enumerate(tags):
        if tag == "B":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "I" and start is not None:
            continue
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def span_with_context(tokens: List[str], start: int, end: int) -> Tuple[str, str]:
    skill_span = " ".join(tokens[start : end + 1])
    ctx_start = max(0, start - 1)
    ctx_end = min(len(tokens) - 1, end + 1)
    context = " ".join(tokens[ctx_start : ctx_end + 1])
    return skill_span, context


def gold_spans_from_bio(tokens: List[str], tags: List[str]) -> Set[Tuple[int, int]]:
    spans: Set[Tuple[int, int]] = set()
    cursor = 0
    span_start = span_end = None
    for tok, tag in zip(tokens, tags):
        tok_start = cursor
        tok_end = cursor + len(tok)
        cursor = tok_end + 1
        if tag == "B":
            if span_start is not None:
                spans.add((span_start, span_end))
            span_start, span_end = tok_start, tok_end
        elif tag == "I" and span_start is not None:
            span_end = tok_end
        else:
            if span_start is not None:
                spans.add((span_start, span_end))
                span_start = span_end = None
    if span_start is not None:
        spans.add((span_start, span_end))
    return spans


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def parse_assistant_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences just in case the model emits them despite instruction.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def find_span_offsets(sentence: str, skill_span: str, context: str) -> Optional[Tuple[int, int]]:
    if not skill_span:
        return None
    sent_l = sentence.lower()
    sk_l = skill_span.lower()
    ctx_l = (context or "").lower()
    if ctx_l and ctx_l in sent_l:
        ctx_start = sent_l.index(ctx_l)
        rel = ctx_l.find(sk_l)
        if rel != -1:
            start = ctx_start + rel
            return (start, start + len(skill_span))
    if sk_l in sent_l:
        start = sent_l.index(sk_l)
        return (start, start + len(skill_span))
    return None


def f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


# --------------------------------------------------------------------------- #
# Few-shot example sampling
# --------------------------------------------------------------------------- #


def build_target_dict(tokens: List[str], tags_skill: List[str], tags_knowledge: List[str]) -> dict:
    skills = []
    for s, e in bio_to_spans(tags_skill):
        span_text, context = span_with_context(tokens, s, e)
        skills.append({"skill_span": span_text, "context": context})
    knowledges = []
    for s, e in bio_to_spans(tags_knowledge):
        span_text, context = span_with_context(tokens, s, e)
        knowledges.append({"skill_span": span_text, "context": context})
    return {"SKILL": skills, "KNOWLEDGE": knowledges}


def sample_few_shot(train_examples: List[dict], k: int) -> List[dict]:
    """Sample k training examples deterministically. Prefer examples that have
    at least one SKILL AND one KNOWLEDGE span — these are the most informative
    for prompt-shaping."""
    rng = random.Random(RANDOM_SEED)
    informative = [
        ex for ex in train_examples
        if any(t in ("B", "I") for t in ex.get("tags_skill", []))
        and any(t in ("B", "I") for t in ex.get("tags_knowledge", []))
    ]
    pool = informative if len(informative) >= k else train_examples
    return rng.sample(pool, k=min(k, len(pool)))


def few_shot_messages(few_shot_examples: List[dict]) -> List[dict]:
    """Convert sampled training examples into a sequence of (user, assistant)
    message pairs to prepend before the actual query."""
    messages: List[dict] = []
    for ex in few_shot_examples:
        target = build_target_dict(
            ex["tokens"],
            ex.get("tags_skill", []),
            ex.get("tags_knowledge", []),
        )
        messages.append({"role": "user", "content": build_user_message(ex["tokens"])})
        messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
    return messages


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #


def evaluate(
    model: str,
    split: str,
    sample: Optional[int],
    few_shot: int,
    output_dir: Path,
) -> None:
    print(f"[INFO] model:    {model}")
    print(f"[INFO] split:    {split}")
    print(f"[INFO] few-shot: {few_shot}")
    print(f"[INFO] sample:   {sample if sample else 'all'}")

    client = load_openrouter_client()

    raw = load_skillspan(DATA_DIR)
    if split not in raw:
        raise FileNotFoundError(f"{split}.json not found in {DATA_DIR}")
    examples = raw[split]
    if sample is not None:
        examples = examples[:sample]

    fs_examples: List[dict] = []
    if few_shot > 0:
        if "train" not in raw:
            raise FileNotFoundError(f"--few-shot requires train.json in {DATA_DIR}")
        fs_examples = sample_few_shot(raw["train"], few_shot)
        print(f"[INFO] sampled {len(fs_examples)} few-shot examples (seed={RANDOM_SEED})")
    fs_messages = few_shot_messages(fs_examples)

    skill_tp = skill_fp = skill_fn = 0
    knowl_tp = knowl_fp = knowl_fn = 0
    n_skill_pred = n_skill_short = n_parse_failures = n_api_failures = 0
    raw_outputs: List[dict] = []

    for ex in tqdm(examples, desc=f"{split}/{model.split('/')[-1]}"):
        tokens = ex["tokens"]
        sentence = " ".join(tokens)
        gold_skill = gold_spans_from_bio(tokens, ex.get("tags_skill", []))
        gold_knowl = gold_spans_from_bio(tokens, ex.get("tags_knowledge", []))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(fs_messages)
        messages.append({"role": "user", "content": build_user_message(tokens)})

        raw_text = call_with_retry(client, model, messages)
        if raw_text is None:
            n_api_failures += 1
            n_parse_failures += 1  # treat API failure as a parse failure for accounting
            raw_outputs.append({"tokens": tokens, "raw_output": None, "parsed": None})
            continue

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

        raw_outputs.append({"tokens": tokens, "raw_output": raw_text, "parsed": parsed})

    s_p, s_r, s_f1 = f1(skill_tp, skill_fp, skill_fn)
    k_p, k_r, k_f1 = f1(knowl_tp, knowl_fp, knowl_fn)
    t_p, t_r, t_f1 = f1(
        skill_tp + knowl_tp,
        skill_fp + knowl_fp,
        skill_fn + knowl_fn,
    )
    verb_short = (n_skill_short / n_skill_pred) if n_skill_pred else 0.0
    verb_fail_threshold = DEFAULT_TRAIN_SHORT_RATE + VERB_FAILURE_TOLERANCE_DELTA
    verb_failed = verb_short > verb_fail_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("/", "-").replace(":", "-")
    metrics_path = output_dir / f"metrics_{split}_{model_slug}.txt"
    raw_path = output_dir / f"raw_outputs_{split}_{model_slug}.jsonl"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"=== {split.upper()} METRICS -- {model} (zero-shot via OpenRouter) ===\n")
        f.write(f"# strict span-set F1 (Skill-LLM Table 2 definition)\n")
        f.write(f"# few-shot k = {few_shot}, sample = {sample if sample else 'all'}\n\n")
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
        f.write(f"api_failures:        {n_api_failures} / {len(examples)}\n")
        f.write(f"json_parse_failures: {n_parse_failures} / {len(examples)}\n")
        f.write(f"skill_predictions:   {n_skill_pred}\n")
        f.write(
            f"skill_short_spans:   {n_skill_short} "
            f"(< {VERB_PRESERVATION_MIN_TOKENS} tokens)\n"
        )
        f.write(f"verb_short_rate:     {verb_short:.4f}\n")
        f.write(f"training_baseline:   {DEFAULT_TRAIN_SHORT_RATE:.4f}\n")
        f.write(
            f"verb_fail_threshold: {verb_fail_threshold:.4f} "
            f"(baseline + tolerance {VERB_FAILURE_TOLERANCE_DELTA:.2f})\n"
        )
        f.write(
            "WARN: verb-preservation diagnostic FAILED.\n" if verb_failed else
            "OK: verb-preservation diagnostic passed.\n"
        )

    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in raw_outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"\n[RESULT] {split} | {model}\n"
        f"  skill F1     = {s_f1:.4f}\n"
        f"  knowledge F1 = {k_f1:.4f}\n"
        f"  total F1     = {t_f1:.4f}\n"
        f"  verb_short   = {verb_short:.1%} (baseline {DEFAULT_TRAIN_SHORT_RATE:.1%}, "
        f"{'FAIL' if verb_failed else 'OK'})\n"
        f"  api_failures = {n_api_failures}, parse_failures = {n_parse_failures}\n"
        f"  metrics  -> {metrics_path}\n"
        f"  raw out  -> {raw_path}"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot / few-shot SkillSpan extraction via OpenRouter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported models:\n" + "\n".join(
            f"  {k}  -- {v}" for k, v in SUPPORTED_MODELS.items()
        ),
    )
    parser.add_argument(
        "--model", required=True,
        help="OpenRouter model slug, e.g. openai/gpt-4o-mini or anthropic/claude-3.5-haiku",
    )
    parser.add_argument(
        "--split", default="test", choices=("dev", "test"),
        help="SkillSpan split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only evaluate the first N examples (smoke test). Default: all.",
    )
    parser.add_argument(
        "--few-shot", type=int, default=0,
        help="Number of training examples to include as few-shot demos (default: 0 = pure zero-shot).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUTS_DIR,
        help="Where to write metrics_*.txt and raw_outputs_*.jsonl.",
    )
    args = parser.parse_args()

    if args.model not in SUPPORTED_MODELS:
        print(
            f"[WARN] {args.model} is not in the curated SUPPORTED_MODELS list. "
            f"Proceeding anyway -- this is a hint not a gate.",
            file=sys.stderr,
        )

    evaluate(
        model=args.model,
        split=args.split,
        sample=args.sample,
        few_shot=args.few_shot,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
