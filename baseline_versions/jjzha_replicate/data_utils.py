"""
data_utils.py — jjzha_replicate

Loads SkillSpan train / dev / test JSON splits and produces a
torch.utils.data.Dataset of pre-tokenised, label-aligned tensors suitable for
HuggingFace `BertForTokenClassification` training (no CRF).

Two label-alignment strategies, gated by `LABEL_ALL_SUBWORDS`:
    False (default): only the FIRST subword of each word receives the head
                     word's BIO label; remaining subwords and special tokens
                     get -100 (PyTorch CrossEntropyLoss ignore index).
    True           : every subword receives the head word's BIO label;
                     special tokens still get -100.

The first variant matches HF's standard recipe and the implicit assumption
behind `pipeline(..., aggregation_strategy="first")`. See AUDIT.md §3.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from config import LABEL2ID, MAX_LEN, LABEL_ALL_SUBWORDS

IGNORE_INDEX = -100  # CrossEntropyLoss default ignore_index


def load_skillspan_data(data_dir: Path) -> dict:
    """Load SkillSpan splits from JSON or JSONL files in `data_dir`.

    Each split is a list of dicts with keys: tokens, tags_skill, tags_knowledge.
    """
    data_dir = Path(data_dir)
    files = {"train": "train.json", "dev": "dev.json", "test": "test.json"}
    dataset: dict = {}

    print(f"[INFO] Reading SkillSpan from: {data_dir}")
    for split, filename in files.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[WARN] {filename} not found in {data_dir}")
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
            if isinstance(content, list):
                dataset[split] = content
            elif isinstance(content, dict):
                # Some redistributions wrap the list under a single key.
                for value in content.values():
                    if isinstance(value, list):
                        dataset[split] = value
                        break
        except json.JSONDecodeError:
            # Fallback: JSONL, one example per line.
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            dataset[split] = data

        if split in dataset:
            print(f"[INFO]   {split}: {len(dataset[split])} examples")

    return dataset


class SingleTaskSkillSpanDataset(Dataset):
    """Single-task SkillSpan dataset for `BertForTokenClassification`.

    task: "skill" or "knowledge" — chooses which BIO column to read.
    """

    def __init__(self, data: List[dict], tokenizer, task: str):
        assert task in {"skill", "knowledge"}, f"unknown task: {task}"
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self._tag_field = "tags_skill" if task == "skill" else "tags_knowledge"

        self.input_ids: List[List[int]] = []
        self.attention_masks: List[List[int]] = []
        self.labels: List[List[int]] = []

        self._prepare()

    def _align_labels(self, tokens: List[str], tags: List[str]):
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_attention_mask=True,
            add_special_tokens=True,
        )

        word_ids = encoding.word_ids()
        label_ids: List[int] = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # [CLS] / [SEP] / pad — never supervise.
                label_ids.append(IGNORE_INDEX)
            elif word_idx != previous_word_idx:
                # First sub-word of a new word — always supervised.
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(LABEL2ID.get(tag, LABEL2ID["O"]))
            else:
                # Non-first sub-word.
                if LABEL_ALL_SUBWORDS:
                    tag = tags[word_idx] if word_idx < len(tags) else "O"
                    label_ids.append(LABEL2ID.get(tag, LABEL2ID["O"]))
                else:
                    label_ids.append(IGNORE_INDEX)
            previous_word_idx = word_idx

        return encoding["input_ids"], encoding["attention_mask"], label_ids

    def _prepare(self):
        for ex in self.data:
            tokens = ex["tokens"]
            tags = ex.get(self._tag_field, [])
            input_ids, attn_mask, label_ids = self._align_labels(tokens, tags)
            self.input_ids.append(input_ids)
            self.attention_masks.append(attn_mask)
            self.labels.append(label_ids)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
