"""
run_eval_only_on_kaggle.py — resume Skill-LLM eval from a previous run.

Use this when a previous Save Version run trained successfully (LoRA adapter
saved to /kaggle/working/skill_llm/lora_adapter) but the eval step crashed.
This script skips Step 1 (prepare data) and Step 2 (train), then runs only
Step 3 (evaluate) against the adapter from the previous run.

Total runtime: ~30-45 min (eval only) vs ~2.5h for a full re-run.

Setup:
    1. Create a NEW Kaggle notebook.
    2. Accelerator -> "GPU T4 x2" (or P100). Same hardware as the trained run.
    3. Internet -> On.
    4. Secrets -> add HF_TOKEN, toggle on for this notebook.
    5. "+ Add Data" in the right sidebar -> "Notebook Output Files" tab ->
       find the failed Save Version that has the adapter -> "Add". Kaggle
       will mount its /kaggle/working/ contents at /kaggle/input/<slug>/.
       Look at the right sidebar to find the exact path.
    6. "+ Add Data" again -> add the SkillSpan dataset (same as the original
       training run). Path stays /kaggle/input/skillspan/.
    7. Set PREVIOUS_RUN_INPUT below to the path Kaggle assigned in step 5.
    8. Paste this whole file into a single code cell and Run All (or Save
       Version -> Save & Run All).

After it finishes: download outputs/trained/metrics_test.txt and
metrics_dev.txt from /kaggle/working/.
"""
from __future__ import annotations

# =========================================================================
# Cell 1: install missing packages (run as SEPARATE cell, then restart kernel)
# =========================================================================
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

# Single GPU lock before torch import (see run_on_kaggle.py for rationale).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
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
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- paths -----------------------------------------------------------------
# *** EDIT THIS *** to point at the failed notebook's output mount.
# Find the exact name by looking at /kaggle/input/ in the right sidebar after
# you attach the failed notebook output. Example values:
#   PREVIOUS_RUN_INPUT = Path("/kaggle/input/notebookXXXXXXXX")
#   PREVIOUS_RUN_INPUT = Path("/kaggle/input/skill-llm-run-1")
PREVIOUS_RUN_INPUT = Path("/kaggle/input/CHANGE_ME_TO_FAILED_NOTEBOOK_INPUT_NAME")

# These are derived; do not edit unless your previous run wrote elsewhere.
ADAPTER_DIR = PREVIOUS_RUN_INPUT / "skill_llm" / "lora_adapter"
DATASETS_DIR_INPUT = PREVIOUS_RUN_INPUT / "skill_llm" / "datasets"

# Output goes here (fresh /kaggle/working/).
WORKING_ROOT = Path("/kaggle/working/skill_llm")
OUTPUTS_DIR = WORKING_ROOT / "outputs"
HF_CACHE_DIR = Path("/kaggle/working/hf_cache")

# SkillSpan dataset path (same as the training run).
DATA_DIR = Path("/kaggle/input/skillspan")

# Validate paths before we burn 30 min.
assert ADAPTER_DIR.exists(), (
    f"Adapter not found at {ADAPTER_DIR}. "
    f"Did you attach the failed notebook output and set PREVIOUS_RUN_INPUT?"
)
assert (ADAPTER_DIR / "adapter_config.json").exists(), (
    f"adapter_config.json missing at {ADAPTER_DIR}"
)
assert DATA_DIR.exists(), f"SkillSpan data not found at {DATA_DIR}"
print(f"[INFO] adapter located at: {ADAPTER_DIR}")
print(f"[INFO] SkillSpan data at:  {DATA_DIR}")

# Kaggle HF_TOKEN.
try:
    from kaggle_secrets import UserSecretsClient
    _hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = _hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    print("[INFO] HF_TOKEN loaded from Kaggle secret")
except Exception as e:
    print(f"[WARN] could not read HF_TOKEN: {e}")

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))

# Model config -- must match the training run.
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_USE_DOUBLE_QUANT = True

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


# =========================================================================
# Cell 3: load SkillSpan + helpers (verbatim from run_on_kaggle.py)
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


def build_user_message(tokens: List[str]) -> str:
    sentence = " ".join(tokens)
    return f"{SENTENCE_BOUNDARY_TOKEN} {sentence} {SENTENCE_BOUNDARY_TOKEN}"


def build_quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


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


# =========================================================================
# Cell 4: evaluation (mirrors run_on_kaggle.py:evaluate)
# =========================================================================


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

    # Verb-preservation baseline (from training_stats.json in the previous run).
    train_stats_path = DATASETS_DIR_INPUT / "training_stats.json"
    train_short_rate = 0.144
    if train_stats_path.exists():
        try:
            stats = json.loads(train_stats_path.read_text(encoding="utf-8"))
            if stats.get("skill_short_rate") is not None:
                train_short_rate = float(stats["skill_short_rate"])
            print(f"[INFO] loaded verb baseline {train_short_rate:.3f} from {train_stats_path}")
        except Exception as e:
            print(f"[WARN] could not parse training_stats.json: {e}; using default 0.144")
    else:
        print(f"[WARN] training_stats.json not found at {train_stats_path}; using default 0.144")
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
            f.write(f"=== {split.upper()} METRICS - Skill-LLM (Kaggle, LLaMA 3.1 8B Instruct) ===\n")
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
# Cell 5: run eval only
# =========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("EVAL-ONLY MODE - resuming from previously trained adapter")
    print("=" * 72)
    print(f"  adapter:  {ADAPTER_DIR}")
    print(f"  data:     {DATA_DIR}")
    print(f"  output:   {OUTPUTS_DIR}")
    print("=" * 72)

    evaluate(splits=("dev", "test"))

    print("\n" + "=" * 72)
    print("DONE.")
    print("=" * 72)
    print(f"Metrics:  {OUTPUTS_DIR / 'trained'}")
    print()
    print("To download:")
    print("  Right sidebar -> /kaggle/working/skill_llm/outputs/trained/ -> Download")
