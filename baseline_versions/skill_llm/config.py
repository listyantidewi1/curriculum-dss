"""
config.py — skill_llm

Hyperparameters reproducing Skill-LLM (Herandi et al. 2024, AAAI 2025) on SkillSpan.
See AUDIT.md for sourcing of every value.

Switch BASE_MODEL_NAME to a non-gated model (Qwen/Qwen2.5-7B-Instruct,
mistralai/Mistral-7B-Instruct-v0.3, unsloth/llama-3-8b-Instruct-bnb-4bit) if your
environment doesn't have a Meta HF access token.
"""
from pathlib import Path

# --- paths -----------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\Projects\skill-extraction")
DATA_DIR = PROJECT_ROOT / "DATA"  # SkillSpan train / dev / test JSON

PACKAGE_ROOT = PROJECT_ROOT / "baseline_versions" / "skill_llm"
HF_CACHE_DIR = PACKAGE_ROOT / "hf_cache"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"

# Prepared chat-format JSONL files (output of prepare_data.py)
DATASETS_DIR = PACKAGE_ROOT / "datasets"

# LoRA adapter (output of train.py)
ADAPTER_DIR = PACKAGE_ROOT / "lora_adapter"

# --- backbone --------------------------------------------------------------

# Skill-LLM paper used "LLaMA 3 8B Instruct" (April 2024). LLaMA 3.1 is the
# same architecture with 128k context and slightly better instruction
# following — the closest currently-available drop-in. Requires Meta-approved
# gated access (https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct);
# approval is usually instant for individual researchers.
#
# Alternatives (drop-in if approval is delayed and you want to start now):
#   meta-llama/Llama-3.2-3B-Instruct      (Meta-gated, smaller, same recipe)
#   Qwen/Qwen2.5-7B-Instruct              (no gating, comparable scale)
#   mistralai/Mistral-7B-Instruct-v0.3    (no gating, slightly smaller)
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# 4-bit NF4 quantisation (bitsandbytes). Required to fit 8B on 24 GB GPUs.
USE_4BIT = True
BNB_4BIT_QUANT_TYPE = "nf4"
# float16, not bfloat16: most consumer / free-tier GPUs (T4 Turing compute 7.5,
# P100 Pascal compute 6.0, RTX 30xx) lack bf16 tensor cores. bf16 there falls
# back to slow emulation and bitsandbytes >= 0.49 has triggered hard crashes on
# Turing+bf16. Native bf16 only exists on compute >= 8.0 (A100, H100, RTX 40xx).
# Switch to bfloat16 if and only if you confirmed your GPU is Ampere or newer.
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_USE_DOUBLE_QUANT = True

# --- LoRA recipe (Skill-LLM paper §"Experimental Setup") --------------------

LORA_RANK = 64
LORA_ALPHA = 128                    # 2 x rank, standard
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"

# --- training (Skill-LLM paper) --------------------------------------------

LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACC_STEPS = 4                  # effective batch 16
EPOCHS = 2
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.10
LR_SCHEDULER = "cosine"
GRAD_CLIP_NORM = 1.0

# --- tokenisation ----------------------------------------------------------

# Skill-LLM doesn't pin a sequence length. SkillSpan sentences are short
# (median ~25 tokens, p99 ~120 tokens); 256 fits the chat template + JSON target
# with margin and is ~4x faster + much lower VRAM than 1024. This materially
# reduces both runtime and OOM risk on 16 GB GPUs (T4, P100). Raise only if
# your input sentences are routinely > 200 tokens AND you have spare VRAM.
MAX_SEQ_LEN = 256

# --- inference -------------------------------------------------------------

INFERENCE_MAX_NEW_TOKENS = 512      # JSON outputs are typically <200 tokens
INFERENCE_TEMPERATURE = 0.0         # deterministic (greedy decode)
INFERENCE_DO_SAMPLE = False

# --- prompt template -------------------------------------------------------

# Verbatim from Skill-LLM Figure 1. **CRITICAL:** the SKILL.skill_span field is
# expected to be VERB-LED (e.g. "implementing and promoting all QA relevant
# topics") and the KNOWLEDGE.skill_span field is the noun-only form (e.g. "QA").
# eval.py runs a diagnostic that fails CI if the verb-preservation rate drops
# below the threshold in VERB_FAILURE_RATE_MAX. See AUDIT.md "CRITICAL INVARIANT".

SYSTEM_PROMPT = (
    "You are a helpful information extraction system. "
    "Your job is to extract skill entities and knowledge entities from the "
    "given sentence."
)

# Boundary markers added on either side of every input sentence so that the
# `context` field (one token of surrounding text) is well-defined even when a
# span sits at the start or end of the sentence. Verbatim from the paper.
SENTENCE_BOUNDARY_TOKEN = "**"

# --- verb-preservation diagnostic ------------------------------------------
#
# The point of this diagnostic is to catch the failure mode where the model
# collapses a verb-led action ("designing UI/UX") down to a tech noun ("UI/UX")
# and emits the noun under the SKILL key instead of the KNOWLEDGE key.
#
# A naive "is this SKILL span shorter than 2 tokens?" check is too tight on
# SkillSpan because the SKILL category natively includes single-token soft
# skills ("passion", "passionate", "empathetic", "self-starter"). On the
# training data ~14% of SKILL spans are single-token for this reason. We
# therefore compare the *eval-time* short-SKILL rate against the *training-time*
# rate (recorded in datasets/training_stats.json by prepare_data.py) and only
# flag the model when it materially exceeds the training distribution.

VERB_PRESERVATION_MIN_TOKENS = 2

# Tolerance: how far above the training-set short-SKILL rate the model is
# allowed to drift before we flag it as a verb-preservation failure. A value
# of 0.10 means: if training is 14% short, eval must stay below 24% short.
# Wider than the original 5% absolute gate; narrower than "anything goes".
VERB_FAILURE_TOLERANCE_DELTA = 0.10

# Path to the training-stats sidecar produced by prepare_data.py. Loaded at
# eval time to set the per-run baseline for the diagnostic.
TRAINING_STATS_PATH = DATASETS_DIR / "training_stats.json"

# --- reproducibility -------------------------------------------------------

RANDOM_SEED = 42
