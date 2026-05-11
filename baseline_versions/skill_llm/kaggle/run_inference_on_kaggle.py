"""
run_inference_on_kaggle.py — production Skill-LLM batch inference on Kaggle.

Loads the trained LoRA adapter (from a Kaggle Dataset, attached as input) and
runs extraction on a JSONL of arbitrary sentences (also attached as a Kaggle
Dataset input). Outputs a JSONL file with extractions per sentence.

Use this for production job-posting extraction. For SkillSpan evaluation use
run_eval_only_on_kaggle.py instead.

Setup:
    1. Create a new Kaggle notebook (or re-use an existing one).
    2. Accelerator -> "GPU T4 x2". Internet -> On.
    3. Add-ons -> Secrets -> HF_TOKEN with Meta access to Llama 3.1 8B Instruct.
    4. + Add Data:
        a. The trained-adapter dataset (the output of the previous
           run_eval_only_on_kaggle.py run, which contains skill_llm/lora_adapter/
           and skill_llm/datasets/). Mounts at /kaggle/input/<adapter-dataset>/
        b. The sentences-to-extract dataset (produced locally by
           scripts/export_sentences_for_skill_llm.py — a single JSONL file).
           Mounts at /kaggle/input/<input-dataset>/
    5. Set ADAPTER_INPUT_DIR and SENTENCES_INPUT_PATH below to the paths
       Kaggle assigned at step 4 (check /kaggle/input/ in the right sidebar).
    6. Paste this file into a single notebook cell, "Save Version -> Save & Run All".

Expected wall-clock: depends on sentence count. At ~0.5s/sentence on T4 fp16,
  1,000 sentences ~ 8 min
  5,000 sentences ~ 40 min
 10,000 sentences ~ 85 min
Use the SAMPLE parameter to smoke-test before committing to a long run.

Output: /kaggle/working/skill_llm_extractions.jsonl
  One JSON object per line, schema documented in
  extractors/skill_llm_offline.py (local-side loader).
"""
from __future__ import annotations

# =========================================================================
# Cell 1: install missing packages (separate cell or rely on inline install)
# =========================================================================
#     !pip install -q -U bitsandbytes>=0.46.1 peft>=0.13 accelerate>=0.34 datasets transformers

# =========================================================================
# Cell 2: imports + constants
# =========================================================================
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Pin to one GPU before any torch import (see run_on_kaggle.py for rationale).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["PYTHONUNBUFFERED"] = "1"

# Install dependencies inline so kernel restart is not required.
import subprocess
print("[INFO] ensuring required packages are up to date...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "bitsandbytes>=0.46.1", "peft>=0.13", "accelerate>=0.34",
     "datasets", "transformers"],
    check=True,
)
print("[INFO] package install complete")

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- paths -----------------------------------------------------------------
# *** EDIT THESE TWO PATHS *** to match what Kaggle assigned.
# Find them by inspecting /kaggle/input/ in the right sidebar after attaching
# the datasets.
ADAPTER_INPUT_DIR = Path("/kaggle/input/CHANGE_ME_TO_ADAPTER_DATASET_NAME")
SENTENCES_INPUT_PATH = Path("/kaggle/input/CHANGE_ME_TO_SENTENCES_DATASET_NAME/skill_llm_input.jsonl")

# Resolved adapter location inside the adapter-dataset mount.
ADAPTER_DIR = ADAPTER_INPUT_DIR / "skill_llm" / "lora_adapter"

# Output goes to /kaggle/working/ (downloadable from the Output tab).
OUTPUT_PATH = Path("/kaggle/working/skill_llm_extractions.jsonl")
HF_CACHE_DIR = Path("/kaggle/working/hf_cache")

# Hard validation before the model load (saves wasted compute on bad paths).
assert ADAPTER_DIR.exists(), (
    f"Adapter not found at {ADAPTER_DIR}. "
    f"Did you attach the adapter dataset and set ADAPTER_INPUT_DIR?"
)
assert (ADAPTER_DIR / "adapter_config.json").exists(), (
    f"adapter_config.json missing at {ADAPTER_DIR}"
)
assert SENTENCES_INPUT_PATH.exists(), (
    f"Sentences input not found at {SENTENCES_INPUT_PATH}. "
    f"Did you attach the input-sentences dataset?"
)
print(f"[INFO] adapter:  {ADAPTER_DIR}")
print(f"[INFO] sentences:{SENTENCES_INPUT_PATH}")
print(f"[INFO] output:   {OUTPUT_PATH}")

# HF token from Kaggle secrets.
try:
    from kaggle_secrets import UserSecretsClient
    _hf = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = _hf
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf
    print("[INFO] HF_TOKEN loaded from Kaggle secret")
except Exception as e:
    print(f"[WARN] could not read HF_TOKEN: {e}")

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))

# --- model / inference config (must match training; do not edit casually) ---
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_COMPUTE_DTYPE = "float16"  # T4/P100 lack bf16 tensor cores
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

# Run knobs.
SAMPLE: Optional[int] = None  # set to e.g. 50 for a smoke test; None = process all
LOG_EVERY_N = 100              # print a progress checkpoint every N sentences
SAVE_EVERY_N = 500             # flush output file every N sentences (crash safety)

MODEL_TAG = "skill_llm_8b_lora_v1"


# =========================================================================
# Cell 3: helpers (mirrors run_eval_only_on_kaggle.py for self-containment)
# =========================================================================


def build_quantisation_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    )


def build_user_message(sentence: str) -> str:
    return f"{SENTENCE_BOUNDARY_TOKEN} {sentence} {SENTENCE_BOUNDARY_TOKEN}"


import re
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def parse_assistant_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
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


@torch.no_grad()
def generate_one(model, tokenizer, sentence: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(sentence)},
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
# Cell 4: load model + adapter
# =========================================================================

def load_skill_llm():
    print("[INFO] loading tokenizer + base model + adapter (5-15 min cold)")
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
    print("[INFO] model + adapter loaded")
    return model, tokenizer


# =========================================================================
# Cell 5: batch inference
# =========================================================================

def load_input_sentences(path: Path) -> List[dict]:
    """Each line is {"sentence_id": ..., "sentence_text": ...}."""
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] malformed JSONL at line {line_no}: {e}; skipping")
                continue
            if not rec.get("sentence_text"):
                continue
            out.append({
                "sentence_id": str(rec.get("sentence_id", "")).strip(),
                "sentence_text": str(rec["sentence_text"]).strip(),
            })
    return out


def run_inference():
    sentences = load_input_sentences(SENTENCES_INPUT_PATH)
    if SAMPLE is not None and SAMPLE > 0:
        sentences = sentences[:SAMPLE]
        print(f"[INFO] capped to first {SAMPLE} sentences (smoke-test mode)")
    print(f"[INFO] {len(sentences)} sentences to process")

    model, tokenizer = load_skill_llm()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    n_parse_fail = 0
    n_empty = 0

    # Open in line-buffered append mode and flush periodically so partial
    # progress survives a kernel kill.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for i, sent in enumerate(tqdm(sentences, desc="extracting")):
            raw_text = generate_one(model, tokenizer, sent["sentence_text"])
            parsed = parse_assistant_json(raw_text)
            if parsed is None:
                n_parse_fail += 1
                skill_arr, knowledge_arr = [], []
            else:
                skill_arr = parsed.get("SKILL") or []
                knowledge_arr = parsed.get("KNOWLEDGE") or []
                if not skill_arr and not knowledge_arr:
                    n_empty += 1

            rec = {
                "sentence_id": sent["sentence_id"],
                "sentence_text": sent["sentence_text"],
                "SKILL": skill_arr,
                "KNOWLEDGE": knowledge_arr,
                "model": MODEL_TAG,
                "extracted_at": started_at,
                "raw_output": raw_text if parsed is None else None,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % SAVE_EVERY_N == 0:
                fout.flush()
            if (i + 1) % LOG_EVERY_N == 0:
                print(
                    f"[INFO] processed {i + 1}/{len(sentences)} "
                    f"({n_parse_fail} parse failures, {n_empty} empty extractions so far)"
                )

    print("=" * 72)
    print(f"[DONE] processed {len(sentences)} sentences")
    print(f"  parse_failures: {n_parse_fail} ({n_parse_fail / max(1, len(sentences)) * 100:.2f}%)")
    print(f"  empty_extractions: {n_empty} ({n_empty / max(1, len(sentences)) * 100:.2f}%)")
    print(f"  output: {OUTPUT_PATH}")
    print("=" * 72)
    print()
    print("Next: download this file to your local repo at:")
    print("  results/skill_llm_extractions.jsonl")
    print("then run:")
    print("  python pipeline.py --extraction-mode skill_llm_offline")


# =========================================================================
# Cell 6: run
# =========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("Skill-LLM 8B LoRA — batch inference for production extraction")
    print("=" * 72)
    run_inference()
