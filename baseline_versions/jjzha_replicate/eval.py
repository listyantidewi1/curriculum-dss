"""
eval.py — jjzha_replicate

Evaluate an already-trained checkpoint (the ones produced by `train.py`) on
the SkillSpan dev + test splits. Two modes:

    --checkpoint <path>      evaluate a local HF save_pretrained directory
    --hf <hf-id>             evaluate a HuggingFace Hub id directly
                             (e.g. jjzha/jobbert_skill_extraction)

This script does NOT use HF's `pipeline(..., aggregation_strategy="first")` so
that span F1 is computed from the same token-classification logits that
training optimises — i.e. apples-to-apples vs `train.py`'s final eval. To
reproduce the demo Space's behaviour (pipeline + aggregate_span post-merge),
see `eval_published.py`, which is intentionally a separate script because the
metric definition there is span-set F1 against gold spans (not token-level
seqeval F1) and the two numbers are not directly comparable.

Usage:
    python eval.py --task skill --checkpoint jobbert_skill_replicate
    python eval.py --task knowledge --hf jjzha/jobbert_knowledge_extraction
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForTokenClassification

from config import (
    BATCH_SIZE,
    DATA_DIR,
    HF_CACHE_DIR,
    ID2LABEL,
    OUTPUTS_DIR,
    REPLICATE_ROOT,
)
from data_utils import IGNORE_INDEX, SingleTaskSkillSpanDataset, load_skillspan_data


os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))


def _decode(logits: torch.Tensor, labels: torch.Tensor):
    """Project logits → BIO tag sequences for seqeval, dropping IGNORE_INDEX positions."""
    preds = logits.argmax(dim=-1).cpu().numpy()
    labs = labels.cpu().numpy()
    true_seqs, pred_seqs = [], []
    for true_row, pred_row in zip(labs, preds):
        t_tags, p_tags = [], []
        for t, p in zip(true_row, pred_row):
            if int(t) == IGNORE_INDEX:
                continue
            t_tags.append(ID2LABEL[int(t)])
            p_tags.append(ID2LABEL[int(p)])
        if t_tags:
            true_seqs.append(t_tags)
            pred_seqs.append(p_tags)
    return true_seqs, pred_seqs


def evaluate(model_or_path: str, task: str, output_dir: Path, label: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading {label} = {model_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_or_path, cache_dir=HF_CACHE_DIR, use_fast=True
    )
    model = BertForTokenClassification.from_pretrained(
        model_or_path, cache_dir=HF_CACHE_DIR
    ).to(device)
    model.eval()

    raw = load_skillspan_data(DATA_DIR)
    results = {}
    for split in ("dev", "test"):
        if split not in raw:
            print(f"[WARN] {split} split missing; skipping")
            continue
        ds = SingleTaskSkillSpanDataset(raw[split], tokenizer, task)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        all_true, all_pred = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                t_seqs, p_seqs = _decode(out.logits, labels)
                all_true.extend(t_seqs)
                all_pred.extend(p_seqs)

        if not all_true:
            print(f"[WARN] no labelled tokens in {split}")
            continue
        p = float(precision_score(all_true, all_pred))
        r = float(recall_score(all_true, all_pred))
        f1 = float(f1_score(all_true, all_pred))

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"metrics_{split}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== {split.upper()} METRICS — {label} ===\n")
            f.write(f"precision: {p:.4f}\n")
            f.write(f"recall:    {r:.4f}\n")
            f.write(f"f1:        {f1:.4f}\n\n")
            f.write("--- TAG REPORT ---\n")
            try:
                f.write(classification_report(all_true, all_pred))
            except ValueError:
                f.write("Could not generate classification report.\n")
        print(f"[INFO] {split} F1={f1:.4f} -> {out_path}")
        results[split] = {"precision": p, "recall": r, "f1": f1}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a token-classification checkpoint on SkillSpan.")
    parser.add_argument("--task", choices=["skill", "knowledge"], required=True)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str,
                     help="Path to a local save_pretrained dir (relative to jjzha_replicate/, or absolute).")
    src.add_argument("--hf", type=str,
                     help="HuggingFace Hub id (e.g. jjzha/jobbert_skill_extraction).")
    parser.add_argument("--output_subdir", type=str, default=None,
                        help="Sub-folder under outputs/ to write metrics into. "
                             "Defaults to the task name when --checkpoint, "
                             "or 'published_<task>' when --hf.")
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = REPLICATE_ROOT / ckpt_path
        label = f"local:{ckpt_path.name}"
        out_subdir = args.output_subdir or args.task
        evaluate(str(ckpt_path), args.task, OUTPUTS_DIR / out_subdir, label)
    else:
        label = f"hf:{args.hf}"
        out_subdir = args.output_subdir or f"published_{args.task}"
        evaluate(args.hf, args.task, OUTPUTS_DIR / out_subdir, label)


if __name__ == "__main__":
    main()
