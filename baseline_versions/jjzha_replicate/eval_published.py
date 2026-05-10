"""
eval_published.py — jjzha_replicate

Reproduces jjzha's exact demo-Space inference pipeline (HF
`pipeline(..., aggregation_strategy="first")` + the demo's `aggregate_span`
post-merger) on a list of SkillSpan test sentences and computes span-set F1
against the gold spans.

This is a separate script from `eval.py` because the metric definition is
different:

    eval.py         — token-level seqeval BIO F1 from raw logits (apples-to-apples
                      with our other baselines and with `train.py`'s final eval).

    eval_published  — span-set F1 from the demo's aggregated span output. The
                      demo emits `{start, end, word, score}` dicts that we
                      compare against gold spans constructed from `tags_skill`
                      / `tags_knowledge`. This is the metric you'd report when
                      claiming "we replicated the demo's behaviour exactly."

Usage:
    python eval_published.py --task skill   --hf jjzha/jobbert_skill_extraction
    python eval_published.py --task knowledge --hf jjzha/jobbert_knowledge_extraction
    python eval_published.py --task skill   --checkpoint jobbert_skill_replicate
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

from transformers import pipeline

from config import (
    AGGREGATION_STRATEGY,
    DATA_DIR,
    HF_CACHE_DIR,
    OUTPUTS_DIR,
    REPLICATE_ROOT,
)
from data_utils import load_skillspan_data


os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))


def aggregate_span(results):
    """Verbatim port of jjzha/skill_extraction_demo:app.py:aggregate_span.

    Glues consecutive spans whose offsets are exactly contiguous
    (`next.start == prev.end + 1`). Required because HF's
    `aggregation_strategy="first"` sometimes splits a single B-…-I-…-I span at
    sub-word boundaries.
    """
    if not results:
        return []
    new_results = []
    current_result = dict(results[0])
    for result in results[1:]:
        if result["start"] == current_result["end"] + 1:
            current_result["word"] += " " + result["word"]
            current_result["end"] = result["end"]
        else:
            new_results.append(current_result)
            current_result = dict(result)
    new_results.append(current_result)
    return new_results


def gold_spans_from_tokens(tokens: List[str], tags: List[str]) -> List[Tuple[int, int]]:
    """Reconstruct (char_start, char_end_exclusive) gold spans from BIO tags
    aligned to a whitespace-joined version of `tokens`. Matches the offsets
    HF's pipeline produces when fed the same whitespace-joined string."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    span_start = None
    span_end = None

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        token_start = cursor
        token_end = cursor + len(tok)
        cursor = token_end + 1  # +1 for the space separator (skipped after last token)

        if tag == "B":
            if span_start is not None:
                spans.append((span_start, span_end))
            span_start, span_end = token_start, token_end
        elif tag == "I" and span_start is not None:
            span_end = token_end
        else:
            if span_start is not None:
                spans.append((span_start, span_end))
                span_start, span_end = None, None

    if span_start is not None:
        spans.append((span_start, span_end))
    return spans


def predicted_spans_from_pipeline(text: str, ner) -> List[Tuple[int, int]]:
    """Run the HF pipeline + aggregate_span on `text`, return (start, end) tuples."""
    raw = ner(text)
    if not raw:
        return []
    merged = aggregate_span(raw)
    return [(int(r["start"]), int(r["end"])) for r in merged]


def span_set_metrics(gold: List[Tuple[int, int]], pred: List[Tuple[int, int]]):
    """Set-equality F1 — a predicted span is correct iff its (start, end)
    tuple matches a gold span exactly. Mirrors the strict span F1 reported in
    the SkillSpan paper."""
    g = set(gold)
    p = set(pred)
    tp = len(g & p)
    fp = len(p - g)
    fn = len(g - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def evaluate(model_id: str, task: str, output_dir: Path, label: str) -> None:
    print(f"[INFO] Loading pipeline for {label} = {model_id}")
    ner = pipeline(
        task="token-classification",
        model=model_id,
        aggregation_strategy=AGGREGATION_STRATEGY,
        device=0,  # GPU if available; pipeline silently falls back to CPU
    )

    raw = load_skillspan_data(DATA_DIR)
    tag_field = "tags_skill" if task == "skill" else "tags_knowledge"

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        if split not in raw:
            continue

        total_gold = total_pred = total_tp = 0
        for ex in raw[split]:
            tokens = ex["tokens"]
            tags = ex.get(tag_field, [])
            text = " ".join(tokens)

            gold = gold_spans_from_tokens(tokens, tags)
            pred = predicted_spans_from_pipeline(text, ner)

            _, _, _, tp, _, _ = span_set_metrics(gold, pred)
            total_gold += len(gold)
            total_pred += len(pred)
            total_tp += tp

        precision = total_tp / total_pred if total_pred else 0.0
        recall = total_tp / total_gold if total_gold else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        out_path = output_dir / f"metrics_{split}_pipeline.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== {split.upper()} PIPELINE METRICS — {label} ===\n")
            f.write(f"# inference: HF pipeline(aggregation_strategy='{AGGREGATION_STRATEGY}') + aggregate_span\n")
            f.write(f"gold_spans: {total_gold}\n")
            f.write(f"pred_spans: {total_pred}\n")
            f.write(f"true_pos:   {total_tp}\n")
            f.write(f"precision:  {precision:.4f}\n")
            f.write(f"recall:     {recall:.4f}\n")
            f.write(f"span_f1:    {f1:.4f}\n")
        print(
            f"[INFO] {split} span_f1={f1:.4f} (gold={total_gold}, pred={total_pred}, tp={total_tp}) -> {out_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate jjzha's published checkpoint (or our replicate) using the demo Space's pipeline."
    )
    parser.add_argument("--task", choices=["skill", "knowledge"], required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str,
                     help="Local save_pretrained dir (relative to jjzha_replicate/ or absolute).")
    src.add_argument("--hf", type=str, help="HuggingFace Hub id.")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Sub-folder under outputs/. Defaults: <task>_pipeline / published_<task>_pipeline.")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = REPLICATE_ROOT / ckpt_path
        label = f"local:{ckpt_path.name}"
        out_subdir = args.output_subdir or f"{args.task}_pipeline"
        evaluate(str(ckpt_path), args.task, OUTPUTS_DIR / out_subdir, label)
    else:
        label = f"hf:{args.hf}"
        out_subdir = args.output_subdir or f"published_{args.task}_pipeline"
        evaluate(args.hf, args.task, OUTPUTS_DIR / out_subdir, label)


if __name__ == "__main__":
    main()
