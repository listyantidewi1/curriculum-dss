"""
train.py — skill_llm

LoRA fine-tune of LLaMA 3.1 8B Instruct on the chat-format SkillSpan JSONL
produced by `prepare_data.py`. Replicates Skill-LLM (Herandi et al. 2024).

The model learns to emit JSON with verb-led SKILL.skill_span and noun
KNOWLEDGE.skill_span — the verb-preservation invariant from AUDIT.md is
inherited from SkillSpan's gold annotations, NOT enforced at training time
beyond the data the model sees.

Outputs the LoRA adapter (small, ~50 MB) under config.ADAPTER_DIR. The base
model is loaded from HuggingFace at inference time and not duplicated.

Usage:
    python prepare_data.py     # if not already run
    python train.py
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    BATCH_SIZE,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    DATASETS_DIR,
    EPOCHS,
    GRAD_ACC_STEPS,
    GRAD_CLIP_NORM,
    HF_CACHE_DIR,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    LORA_TASK_TYPE,
    LR_SCHEDULER,
    MAX_SEQ_LEN,
    OUTPUTS_DIR,
    RANDOM_SEED,
    USE_4BIT,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


# Route all HF caches into the project tree so we don't fill C:.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


def tokenise_record(record: dict, tokenizer) -> dict:
    """Convert a {"messages": [...]} record into input_ids / labels for causal
    LM training. Apply the model's chat template; mask everything before the
    assistant turn so we only compute loss on the gold JSON output.
    """
    messages = record["messages"]
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
    prompt_ids = tokenizer(prompt_only, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

    labels = list(full_ids)
    # Mask the prompt tokens so loss is only over the assistant response.
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main() -> None:
    set_seed(RANDOM_SEED)

    train_path = DATASETS_DIR / "train.jsonl"
    dev_path = DATASETS_DIR / "dev.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run `python prepare_data.py` first."
        )

    print(f"[INFO] base model: {BASE_MODEL_NAME}")
    print(f"[INFO] 4-bit quantisation: {USE_4BIT}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, cache_dir=HF_CACHE_DIR, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        cache_dir=HF_CACHE_DIR,
        torch_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
    )
    if USE_4BIT:
        model_kwargs["quantization_config"] = build_quantisation_config()
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    print(f"[INFO] loading base model (this may take a while on first run)")
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

    data_files = {"train": str(train_path)}
    if dev_path.exists():
        data_files["dev"] = str(dev_path)
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
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if "dev" in tokenised else "no",
        bf16=True,
        report_to=[],
        seed=RANDOM_SEED,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised.get("dev"),
        data_collator=collator,
    )

    print("[INFO] starting training")
    trainer.train()

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"[INFO] saved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
