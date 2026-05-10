"""
config.py — jjzha_replicate

Hyperparameters for replicating jjzha's published JobBERT skill / knowledge
extractors. See AUDIT.md for the full rationale behind every value here.

Two single-task checkpoints (one per task) are trained from `jjzha/jobbert-base-cased`
with a plain `BertForTokenClassification` head — no CRF — using SkillSpan only.
"""
from pathlib import Path

# --- paths -----------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\Projects\skill-extraction")
DATA_DIR = PROJECT_ROOT / "DATA"  # contains train.json / dev.json / test.json

REPLICATE_ROOT = PROJECT_ROOT / "baseline_versions" / "jjzha_replicate"
HF_CACHE_DIR = REPLICATE_ROOT / "hf_cache"
OUTPUTS_DIR = REPLICATE_ROOT / "outputs"

# Per-task checkpoint dirs
SKILL_MODEL_DIR = REPLICATE_ROOT / "jobbert_skill_replicate"
KNOWLEDGE_MODEL_DIR = REPLICATE_ROOT / "jobbert_knowledge_replicate"

# --- backbone / tokenizer --------------------------------------------------

BACKBONE_MODEL_NAME = "jjzha/jobbert-base-cased"

# Match jjzha's published id2label exactly so the published checkpoint
# (`jjzha/jobbert_skill_extraction`, `jjzha/jobbert_knowledge_extraction`) can
# be loaded as-is for direct evaluation.
LABELS = ["B", "I", "O"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# --- tokenisation ----------------------------------------------------------

MAX_LEN = 128                    # SkillSpan sentences are short; 256/512 wastes compute
LABEL_ALL_SUBWORDS = False       # False = first-subword-only (HF norm), -100 for the rest
                                 # True  = label every subword with the head word's tag

# --- training --------------------------------------------------------------

BATCH_SIZE = 8
GRAD_ACC_STEPS = 4               # effective batch = 32

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10              # 10% of total steps, linear warmup
GRAD_CLIP_NORM = 1.0

EPOCHS = 20
PATIENCE = 3                     # early-stop if dev F1 doesn't improve for N epochs
MIN_EPOCHS = 3

DROPOUT = 0.1                    # matches published config

# --- inference -------------------------------------------------------------

# HF token-classification aggregation strategy. "first" = take the first
# sub-word's predicted label as the label of the whole word; matches jjzha's
# demo Space exactly.
AGGREGATION_STRATEGY = "first"

# --- reproducibility -------------------------------------------------------

RANDOM_SEED = 42
