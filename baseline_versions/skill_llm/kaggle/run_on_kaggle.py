"""
run_on_kaggle.py — Skill-LLM end-to-end pipeline for Kaggle.

Self-contained script that mirrors the local prepare_data.py / train.py /
eval.py trio. Drop into a single Kaggle notebook cell, set HF_TOKEN as a
Kaggle Secret, and click Run All. Trains LoRA on LLaMA 3.1 8B Instruct
(paper-matching), evaluates on SkillSpan dev + test, writes the LoRA
adapter and metrics to /kaggle/working/.

Setup (see README.md in this directory for screenshots):
    1. Create a Kaggle notebook.
    2. Accelerator → "GPU T4 ×2" or "GPU P100" (16 GB VRAM either way).
    3. Add-ons → Secrets → add `HF_TOKEN` with your HuggingFace token.
       Token must have read access to gated repos and you must have
       accepted the LLaMA 3.1 8B Instruct license at
       https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
    4. Upload SkillSpan train.json / dev.json / test.json as inputs to the
       notebook (drag-drop into the right sidebar, OR create a Kaggle
       Dataset and attach it).
    5. Adjust DATA_DIR below to match the path Kaggle gave you (it'll
       be /kaggle/input/<dataset-name>/ or /kaggle/input/<filename>).
    6. Paste this whole file into a Kaggle code cell and Run All.
       Total runtime: ~2.5 hours on T4, ~1.5 hours on P100.
    7. After it finishes: download outputs/trained/metrics_test.txt and
       lora_adapter/ from /kaggle/working/.

The script is intentionally a single file. It duplicates ~150 lines from
the local package (data utils, LoRA setup, span-F1 eval) so the notebook
runs without needing the rest of the repo.
"""
from __future__ import annotations

# =========================================================================
# Cell 1: install missing packages
# =========================================================================
# Kaggle's base image doesn't include bitsandbytes (or has an old version).
# Run this as a SEPARATE cell *before* the main script, then RESTART the
# kernel before running the main script — bitsandbytes registers CUDA
# kernels at import time and a stale torch session won't pick them up.
#
#     !pip install -q -U bitsandbytes>=0.46.1 peft>=0.13 accelerate>=0.34 datasets seqeval transformers
#     # then: Run -> Restart Kernel

# =========================================================================
# Cell 2: imports + constants
# =========================================================================
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Lock to a SINGLE GPU before any torch import. Required on Kaggle's
# "GPU T4 x2" accelerator: with both T4s visible, accelerate's device_map="auto"
# shards the 8B model across both 16 GB GPUs, and QLoRA gradient computation
# across the shard boundary causes silent CUDA crashes -> Kaggle kernel
# auto-restart -> training loops forever without progressing. Single-GPU is
# sufficient (8B at 4-bit + LoRA fits in ~10 GB on one T4).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Unbuffered Python output so Trainer's per-step log lines appear in Kaggle's
# Logs panel in near-real-time. Without this, Save Version runs can buffer
# multiple minutes of stdout/stderr before flushing, making it impossible to
# tell whether training is hung or just slow.
os.environ["PYTHONUNBUFFERED"] = "1"

# Install required packages INLINE before any torch / peft / transformers
# import. Kaggle's base image ships an older bitsandbytes and a separate
# "!pip install" cell needs a kernel restart to be picked up; running pip
# inline in the same Python process before the first heavy import sidesteps
# that footgun entirely. The Cell 1 !pip install above is now belt-and-
# -suspenders -- this block will succeed even if Cell 1 never ran.
import subprocess
import sys
print("[INFO] ensuring required packages are up to date...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "bitsandbytes>=0.46.1",
        "peft>=0.13",
        "accelerate>=0.34",
        "datasets",
        "seqeval",
        "transformers",
    ],
    check=True,
)
print("[INFO] package install complete")

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# --- paths (Kaggle-specific) ----------------------------------------------
# Point DATA_DIR at wherever Kaggle put your SkillSpan JSON files.
# Inspect /kaggle/input/ in the file browser to find the right path.
DATA_DIR = Path("/kaggle/input/skillspan")  # change if your dataset name differs

WORKING_ROOT = Path("/kaggle/working/skill_llm")
DATASETS_DIR = WORKING_ROOT / "datasets"
ADAPTER_DIR = WORKING_ROOT / "lora_adapter"
OUTPUTS_DIR = WORKING_ROOT / "outputs"
HF_CACHE_DIR = Path("/kaggle/working/hf_cache")

# Kaggle secrets — HF_TOKEN must be set in Add-ons → Secrets.
try:
    from kaggle_secrets import UserSecretsClient
    _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    print("[INFO] HF_TOKEN loaded from Kaggle secret")
except Exception as e:
    print(f"[WARN] could not read HF_TOKEN from Kaggle secrets: {e}")
    print("[WARN] make sure you've added HF_TOKEN under Add-ons → Secrets and enabled it for this notebook")

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))

# --- model / training config (paper recipe verbatim) ----------------------
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
# float16, not bfloat16: T4 (Turing, compute 7.5) has no bf16 tensor cores so
# bf16 falls back to slow software emulation and has triggered hard crashes in
# bitsandbytes >= 0.49 on Turing. P100 (Pascal, compute 6.0) also lacks bf16.
# Only A100/H100 (compute >= 8.0) have native bf16 — neither is on Kaggle free.
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_USE_DOUBLE_QUANT = True

LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACC_STEPS = 4
EPOCHS = 2
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.10
LR_SCHEDULER = "cosine"
GRAD_CLIP_NORM = 1.0
# SkillSpan sentences are short (median ~25 tokens, p99 ~120 tokens). 256
# fits the prompt + JSON target with margin and is ~4x faster + lower VRAM
# than 1024, which materially reduces both runtime and OOM risk on T4 16 GB.
MAX_SEQ_LEN = 256

INFERENCE_MAX_NEW_TOKENS = 512
INFERENCE_DO_SAMPLE = False
INFERENCE_TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are a helpful information extraction system. "
    "Your job is to extract skill entities and knowledge entities from the "
    "given sentence."
)
SENTENCE_BOUNDARY_TOKEN = "**"

VERB_PRESERVATION_MIN_TOKENS = 2
VERB_FAILURE_TOLERANCE_DELTA = 0.10

LABEL2ID = {"B": 0, "I": 1, "O": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# Cell 3: data preparation (mirrors prepare_data.py)
# =========================================================================


def load_skillspan(data_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for split, fname in (("train", "train.json"), ("dev", "dev.json"), ("test", "test.json")):
        path = data_dir / fname
        if not path.exists():
            print(f"[WARN] {fname} not found in {data_dir}; skipping {split}")
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


def build_target(tokens: List[str], tags_skill: List[str], tags_knowledge: List[str]) -> dict:
    skills = [
        {"skill_span": (sp := span_with_context(tokens, s, e))[0], "context": sp[1]}
        for s, e in bio_to_spans(tags_skill)
    ]
    knowledges = [
        {"skill_span": (sp := span_with_context(tokens, s, e))[0], "context": sp[1]}
        for s, e in bio_to_spans(tags_knowledge)
    ]
    return {"SKILL": skills, "KNOWLEDGE": knowledges}


def build_user_message(tokens: List[str]) -> str:
    sentence = " ".join(tokens)
    return f"{SENTENCE_BOUNDARY_TOKEN} {sentence} {SENTENCE_BOUNDARY_TOKEN}"


def to_chat(example: dict) -> dict:
    tokens = example["tokens"]
    target = build_target(
        tokens,
        example.get("tags_skill", []),
        example.get("tags_knowledge", []),
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(tokens)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def prepare_data() -> dict:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_skillspan(DATA_DIR)

    train_short = train_total = 0
    short_examples: List[str] = []

    for split in ("train", "dev", "test"):
        if split not in raw:
            continue
        records = [to_chat(ex) for ex in raw[split]]
        out_path = DATASETS_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[INFO] wrote {len(records)} chat records to {out_path}")

        if split == "train":
            for rec in records:
                target = json.loads(rec["messages"][2]["content"])
                for s in target.get("SKILL", []):
                    train_total += 1
                    if len(s["skill_span"].split()) < VERB_PRESERVATION_MIN_TOKENS:
                        train_short += 1
                        if len(short_examples) < 20:
                            short_examples.append(s["skill_span"])

    train_short_rate = (train_short / train_total) if train_total else 0.0
    stats = {
        "n_skill_spans": train_total,
        "n_skill_short_spans": train_short,
        "skill_short_rate": train_short_rate,
        "verb_preservation_min_tokens": VERB_PRESERVATION_MIN_TOKENS,
        "sample_short_skill_spans": short_examples,
    }
    (DATASETS_DIR / "training_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[DIAGNOSTIC] training-set SKILL distribution: "
        f"{train_short}/{train_total} spans have < {VERB_PRESERVATION_MIN_TOKENS} "
        f"tokens ({train_short_rate:.1%}). NOT a bug — these are SkillSpan's "
        f"annotated soft-skill spans."
    )
    return stats


# =========================================================================
# Cell 4: training (mirrors train.py)
# =========================================================================


def build_quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


def tokenise_record(record: dict, tokenizer) -> dict:
    messages = record["messages"]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_only = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    prompt_ids = tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    labels = list(full_ids)
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def train() -> None:
    set_seed(RANDOM_SEED)

    train_path = DATASETS_DIR / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found. Run prepare_data() first.")

    print(f"[INFO] base model: {BASE_MODEL_NAME}")
    print(f"[INFO] 4-bit quantisation: {USE_4BIT}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=HF_CACHE_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        cache_dir=HF_CACHE_DIR,
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        device_map="auto",
    )
    if USE_4BIT:
        model_kwargs["quantization_config"] = build_quantisation_config()

    print("[INFO] loading base model (~5 min on first run)")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **model_kwargs)
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias=LORA_BIAS,
        task_type=LORA_TASK_TYPE,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files = {"train": str(DATASETS_DIR / "train.jsonl")}
    if (DATASETS_DIR / "dev.jsonl").exists():
        data_files["dev"] = str(DATASETS_DIR / "dev.jsonl")
    raw = load_dataset("json", data_files=data_files, cache_dir=str(HF_CACHE_DIR))
    print(f"[INFO] train: {len(raw['train'])}")
    if "dev" in raw:
        print(f"[INFO] dev:   {len(raw['dev'])}")

    tokenised = raw.map(
        lambda rec: tokenise_record(rec, tokenizer),
        remove_columns=raw["train"].column_names,
        desc="tokenising",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUTS_DIR / "checkpoints"),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        max_grad_norm=GRAD_CLIP_NORM,
        # Log every step including step 1, not every 10 steps. With
        # gradient_accumulation=4, "step 10" is 40 forward+backward passes,
        # which on T4+fp16+4-bit can take 8+ minutes and looks like a hang.
        # Per-step logs give us a definitive signal that training is alive.
        logging_steps=1,
        logging_first_step=True,
        # CRITICAL for Kaggle Save Version mode: with disable_tqdm=False
        # (the default) Trainer routes the {'loss': ...} log dict through
        # tqdm.write(). In non-TTY environments tqdm auto-disables itself
        # and tqdm.write() becomes a silent no-op -- so per-step logs vanish
        # entirely and training looks hung even when it is progressing
        # normally. disable_tqdm=True swaps ProgressCallback for
        # PrinterCallback, which uses plain print() and works in batch mode.
        disable_tqdm=True,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch" if "dev" in tokenised else "no",
        # fp16, not bf16 — T4/P100 lack bf16 tensor cores (see BNB_4BIT_COMPUTE_DTYPE note).
        bf16=False,
        fp16=True,
        report_to=[],
        seed=RANDOM_SEED,
        gradient_checkpointing=True,
        # use_reentrant=True is the classic GC implementation. Combining
        # use_reentrant=False with bitsandbytes >= 0.49 and fp16 on Turing
        # has been observed to hang silently on the first backward pass
        # (see bitsandbytes-foundation/bitsandbytes#1340 et al). Reentrant
        # GC is slightly slower and uses marginally more VRAM but is the
        # stable path on this hardware stack.
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised.get("dev"),
        data_collator=collator,
    )

    print("[INFO] starting training (~1.5h on P100, ~2.5h on T4)")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"[INFO] saved LoRA adapter to {ADAPTER_DIR}")

    # Free memory before eval (the eval loop reloads with PeftModel.from_pretrained).
    del model, trainer
    torch.cuda.empty_cache()


# =========================================================================
# Cell 5: evaluation (mirrors eval.py + eval_published.py)
# =========================================================================


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def parse_assistant_json(text: str) -> Optional[dict]:
    text = text.strip()
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


def f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


@torch.no_grad()
def generate_one(model, tokenizer, tokens: List[str]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(tokens)},
    ]
    # transformers >= 4.50 returns a BatchEncoding (dict-like) when
    # return_tensors="pt"; older versions returned a raw tensor. Use
    # return_dict=True explicitly + **inputs spread + inputs["input_ids"]
    # so the code works on both old and new transformers without sniffing.
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
        do_sample=INFERENCE_DO_SAMPLE,
        temperature=INFERENCE_TEMPERATURE if INFERENCE_DO_SAMPLE else 1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate(splits: Tuple[str, ...] = ("dev", "test")) -> None:
    print(f"[INFO] loading model + adapter for evaluation")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, cache_dir=HF_CACHE_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_kwargs = dict(
        cache_dir=HF_CACHE_DIR,
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        device_map="auto",
    )
    if USE_4BIT:
        base_kwargs["quantization_config"] = build_quantisation_config()
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **base_kwargs)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    raw = load_skillspan(DATA_DIR)

    # Verb-preservation baseline (loaded from training stats)
    train_stats_path = DATASETS_DIR / "training_stats.json"
    train_short_rate = 0.144  # SkillSpan default; overridden below if file exists
    if train_stats_path.exists():
        try:
            stats = json.loads(train_stats_path.read_text(encoding="utf-8"))
            if stats.get("skill_short_rate") is not None:
                train_short_rate = float(stats["skill_short_rate"])
        except Exception:
            pass
    verb_fail_threshold = train_short_rate + VERB_FAILURE_TOLERANCE_DELTA

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    eval_dir = OUTPUTS_DIR / "trained"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in raw:
            print(f"[WARN] {split} split missing; skipping")
            continue
        examples = raw[split]
        print(f"[INFO] evaluating {split}: {len(examples)} examples")

        skill_tp = skill_fp = skill_fn = 0
        knowl_tp = knowl_fp = knowl_fn = 0
        n_skill_pred = n_skill_short = n_parse_failures = 0
        raw_outputs: List[dict] = []

        for ex in tqdm(examples, desc=split):
            tokens = ex["tokens"]
            sentence = " ".join(tokens)
            gold_skill = gold_spans_from_bio(tokens, ex.get("tags_skill", []))
            gold_knowl = gold_spans_from_bio(tokens, ex.get("tags_knowledge", []))

            raw_text = generate_one(model, tokenizer, tokens)
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
        verb_failed = verb_short > verb_fail_threshold

        with open(eval_dir / f"metrics_{split}.txt", "w", encoding="utf-8") as f:
            f.write(f"=== {split.upper()} METRICS — Skill-LLM (Kaggle, LLaMA 3.1 8B Instruct) ===\n")
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
            f.write(f"verb_short_rate:     {verb_short:.4f}\n")
            f.write(f"training_baseline:   {train_short_rate:.4f}\n")
            f.write(
                f"verb_fail_threshold: {verb_fail_threshold:.4f} "
                f"(baseline + tolerance {VERB_FAILURE_TOLERANCE_DELTA:.2f})\n"
            )
            f.write(
                "WARN: verb-preservation diagnostic FAILED.\n" if verb_failed else
                "OK: verb-preservation diagnostic passed.\n"
            )

        with open(eval_dir / f"raw_outputs_{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in raw_outputs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"[INFO] {split}: skill F1={s_f1:.4f} | knowledge F1={k_f1:.4f} "
            f"| total F1={t_f1:.4f} | short-SKILL={verb_short:.1%} "
            f"(baseline {train_short_rate:.1%})"
        )


# =========================================================================
# Cell 6: run end-to-end
# =========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("Step 1/3: prepare data")
    print("=" * 72)
    prepare_data()

    print("\n" + "=" * 72)
    print("Step 2/3: train LoRA")
    print("=" * 72)
    train()

    print("\n" + "=" * 72)
    print("Step 3/3: evaluate on dev + test")
    print("=" * 72)
    # Defensive wrapper: if eval crashes for any reason, the LoRA adapter
    # from training is still saved at ADAPTER_DIR and we want Save Version
    # to persist /kaggle/working/ outputs (which only happens on success).
    # Print the traceback for diagnosis but let the notebook exit cleanly.
    try:
        evaluate(splits=("dev", "test"))
    except Exception as e:
        import traceback
        print(f"[ERROR] evaluation failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"[INFO] LoRA adapter is saved at {ADAPTER_DIR}")
        print("[INFO] you can download it and re-run eval separately")

    print("\n" + "=" * 72)
    print("DONE.")
    print("=" * 72)
    print(f"Adapter:  {ADAPTER_DIR}")
    print(f"Metrics:  {OUTPUTS_DIR / 'trained'}")
    print()
    print("To download from Kaggle:")
    print("  Right sidebar → /kaggle/working/skill_llm/ → right-click → Download")
    print("  OR commit the notebook and grab them from the run's output panel.")
