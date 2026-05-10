"""
train.py — jjzha_replicate

Trains a single-task `BertForTokenClassification` (no CRF) per the AUDIT.md
recipe. Run twice — once for `--task skill`, once for `--task knowledge` — to
produce the two checkpoints jjzha publishes.

Usage:
    python train.py --task skill
    python train.py --task knowledge

Outputs (per task) under `baseline_versions/jjzha_replicate/`:
    jobbert_<task>_replicate/
        config.json, pytorch_model.bin, tokenizer.json, ...   (HF save_pretrained format)
    outputs/<task>/
        metrics_dev.txt, metrics_test.txt, training_log.txt
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

from config import (
    BACKBONE_MODEL_NAME,
    BATCH_SIZE,
    DATA_DIR,
    DROPOUT,
    EPOCHS,
    GRAD_ACC_STEPS,
    GRAD_CLIP_NORM,
    HF_CACHE_DIR,
    ID2LABEL,
    KNOWLEDGE_MODEL_DIR,
    LABEL2ID,
    LEARNING_RATE,
    MIN_EPOCHS,
    NUM_LABELS,
    OUTPUTS_DIR,
    PATIENCE,
    RANDOM_SEED,
    SKILL_MODEL_DIR,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from data_utils import IGNORE_INDEX, SingleTaskSkillSpanDataset, load_skillspan_data


# Route HF caches into the project tree (not C:).
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_predictions(logits: torch.Tensor, labels: torch.Tensor):
    """Turn (batch, seq, num_labels) logits + (batch, seq) labels into two
    parallel lists of BIO tag sequences (true, predicted), ignoring positions
    where the gold label is IGNORE_INDEX (subword non-heads / special tokens)."""
    preds = logits.argmax(dim=-1)
    true_seqs, pred_seqs = [], []
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    for true_row, pred_row in zip(labels_np, preds_np):
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


def evaluate(model, dataloader, device, split_name: str, output_dir: Path) -> dict:
    """Full evaluation: writes a metrics_{split}.txt and returns a dict."""
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            t_seqs, p_seqs = _decode_predictions(out.logits, labels)
            all_true.extend(t_seqs)
            all_pred.extend(p_seqs)

    if not all_true:
        return {"split": split_name, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    p = float(precision_score(all_true, all_pred))
    r = float(recall_score(all_true, all_pred))
    f1 = float(f1_score(all_true, all_pred))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"metrics_{split_name}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"=== {split_name.upper()} METRICS ===\n")
        f.write(f"precision: {p:.4f}\n")
        f.write(f"recall:    {r:.4f}\n")
        f.write(f"f1:        {f1:.4f}\n\n")
        f.write("--- TAG REPORT ---\n")
        try:
            f.write(classification_report(all_true, all_pred))
        except ValueError:
            f.write("Could not generate classification report (no valid entity labels).\n")
    print(f"[INFO] {split_name} F1={f1:.4f} (P={p:.4f}, R={r:.4f}) -> {out_path}")
    return {"split": split_name, "precision": p, "recall": r, "f1": f1}


def evaluate_for_early_stop(model, dataloader, device) -> float:
    """Lightweight dev F1 for the early-stopping loop."""
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            t_seqs, p_seqs = _decode_predictions(out.logits, labels)
            all_true.extend(t_seqs)
            all_pred.extend(p_seqs)
    if not all_true:
        return 0.0
    return float(f1_score(all_true, all_pred))


def train(task: str) -> None:
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training jjzha-replicate single-task '{task}' on {device}")

    model_dir = SKILL_MODEL_DIR if task == "skill" else KNOWLEDGE_MODEL_DIR
    output_dir = OUTPUTS_DIR / task
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    raw = load_skillspan_data(DATA_DIR)
    for split in ("train", "dev", "test"):
        if split not in raw:
            raise RuntimeError(f"Missing SkillSpan split: {split}")

    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE_MODEL_NAME, cache_dir=HF_CACHE_DIR, use_fast=True
    )
    train_ds = SingleTaskSkillSpanDataset(raw["train"], tokenizer, task)
    dev_ds = SingleTaskSkillSpanDataset(raw["dev"], tokenizer, task)
    test_ds = SingleTaskSkillSpanDataset(raw["test"], tokenizer, task)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- model ---
    model = BertForTokenClassification.from_pretrained(
        BACKBONE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
        cache_dir=HF_CACHE_DIR,
    )
    model.to(device)

    # --- optimiser / schedule ---
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped, lr=LEARNING_RATE)
    total_steps = max(1, int(np.ceil(len(train_loader) * EPOCHS / GRAD_ACC_STEPS)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    # --- train loop ---
    log_path = output_dir / "training_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = []

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    global_step = 0

    print(f"[INFO] total_steps={total_steps}, warmup_steps={int(WARMUP_RATIO * total_steps)}")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / GRAD_ACC_STEPS
            epoch_loss += loss.item()
            loss.backward()

            global_step += 1
            if global_step % GRAD_ACC_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress.set_postfix(loss=f"{loss.item():.5f}")

        if global_step % GRAD_ACC_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(1, len(train_loader))
        dev_f1 = evaluate_for_early_stop(model, dev_loader, device)

        line = (
            f"epoch {epoch + 1:>3} | train_loss {avg_loss:.4f} | dev_f1 {dev_f1:.4f}"
        )
        print(f"[INFO] {line}")
        log_lines.append(line)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            log_lines.append(f"           *** new best (dev_f1={best_f1:.4f}) ***")
        else:
            epochs_no_improve += 1

        if epoch + 1 >= MIN_EPOCHS and epochs_no_improve >= PATIENCE:
            print(f"[INFO] early-stopping at epoch {epoch + 1}; best epoch was {best_epoch}")
            log_lines.append(f"early stop after {epoch + 1} epochs; best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] restored best model (epoch {best_epoch}, dev_f1={best_f1:.4f})")

    # --- save ---
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"[INFO] saved best model + tokenizer to {model_dir}")

    # --- final eval ---
    print(f"[INFO] final dev / test evaluation")
    dev_metrics = evaluate(model, dev_loader, device, "dev", output_dir)
    test_metrics = evaluate(model, test_loader, device, "test", output_dir)
    log_lines.append(f"final dev:  {dev_metrics}")
    log_lines.append(f"final test: {test_metrics}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train jjzha-replicate single-task BertForTokenClassification.")
    parser.add_argument("--task", type=str, choices=["skill", "knowledge"], required=True)
    args = parser.parse_args()
    train(args.task)


if __name__ == "__main__":
    main()
