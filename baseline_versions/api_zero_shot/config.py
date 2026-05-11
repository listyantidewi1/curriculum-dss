"""
config.py — api_zero_shot

Configuration for OpenRouter zero-shot / few-shot SkillSpan extraction.
Pairs apples-to-apples with baseline_versions/skill_llm/ — same SkillSpan
JSON, same prompt structure (system + boundary-wrapped sentence + JSON
target), same strict span-set F1 metric and same verb-preservation
diagnostic.

The goal: measure whether a modern instruction-tuned LLM via API can
match or beat the Skill-LLM fine-tuned 8B baseline, at deployment-
relevant cost. If yes, we ship the API path (no local GPU needed). If
no, the Skill-LLM 8B path stays in the running despite operational cost.
"""
from pathlib import Path

# --- paths -----------------------------------------------------------------

PROJECT_ROOT = Path(r"D:\Projects\skill-extraction")
DATA_DIR = PROJECT_ROOT / "DATA"  # SkillSpan train / dev / test JSON

PACKAGE_ROOT = PROJECT_ROOT / "baseline_versions" / "api_zero_shot"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"

# --- OpenRouter client -----------------------------------------------------

# Same loading convention as generate_competencies.py / sentence_relevance_filter.py:
# env var OPENROUTER_API_KEY takes priority, then fall back to api_keys/jatevo.txt.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_FILE = PROJECT_ROOT / "api_keys" / "jatevo.txt"

# Models to evaluate. Pass --model <slug> on the CLI; this is the menu.
SUPPORTED_MODELS = {
    "openai/gpt-4o-mini":               "GPT-4o-mini  -- cheap, fast, strong JSON",
    "openai/gpt-4o":                    "GPT-4o       -- premium, expensive",
    "anthropic/claude-3.5-haiku":       "Claude Haiku -- cheap, strong reasoning",
    "anthropic/claude-3.5-sonnet":      "Claude Sonnet-- premium",
    "meta-llama/llama-3.1-70b-instruct":"Llama 3.1 70B-- open-weight, mid-cost",
    "deepseek/deepseek-v3.2":           "DeepSeek v3.2-- in use elsewhere in repo",
}

# --- request settings ------------------------------------------------------

TEMPERATURE = 0.0          # deterministic
MAX_TOKENS = 800           # JSON outputs are typically <300 tokens
REQUEST_TIMEOUT = 60       # seconds

# Mirror the retry pattern from sentence_relevance_filter.py.
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2     # exponential: 4s, 8s

# --- prompt template -------------------------------------------------------
#
# Slightly more verbose than Skill-LLM's training prompt because the model
# wasn't fine-tuned on this schema. We explicitly define SKILL vs KNOWLEDGE,
# emphasise verb preservation (the core domain quality bar), and pin the
# JSON shape. Sentence boundary token is identical so the "context" field
# semantics match Skill-LLM's training data.

SYSTEM_PROMPT = """You are an information extraction system for job postings. Given a sentence, extract entities of two types:

1. SKILL -- verb-led action phrases describing what someone DOES.
   Examples: "designing UI/UX", "managing stakeholders", "implementing test cases"
   Also includes standalone soft-skill words: "communication", "leadership", "passion"

2. KNOWLEDGE -- noun phrases naming tools, technologies, methodologies, or concepts.
   Examples: "Python", "Agile methodology", "machine learning", "AWS"

CRITICAL distinction: when a verb appears with a technical noun, the span is a SKILL, not KNOWLEDGE.
  "designing UI/UX"   -> SKILL  (verb-led action)
  "UI/UX"             -> KNOWLEDGE  (bare noun)
  "using Python"      -> SKILL
  "Python"            -> KNOWLEDGE

For each entity, also include "context" = the entity span plus one surrounding word on each side (when available at sentence start/end, fewer is fine).

Output format -- JSON only, no commentary, no markdown fences:
{
  "SKILL": [{"skill_span": "<exact text from sentence>", "context": "<entity + surrounding words>"}],
  "KNOWLEDGE": [{"skill_span": "<exact text from sentence>", "context": "<entity + surrounding words>"}]
}

Rules:
- Use empty lists if no entities of that type appear.
- skill_span text MUST match the source sentence character-for-character.
- Do NOT invent skills that aren't in the sentence.
"""

SENTENCE_BOUNDARY_TOKEN = "**"  # identical to Skill-LLM

# --- verb-preservation diagnostic (same definition as Skill-LLM) ----------

VERB_PRESERVATION_MIN_TOKENS = 2
VERB_FAILURE_TOLERANCE_DELTA = 0.10
# Skill-LLM's training_stats.json baseline; same SkillSpan train split so
# the 14.4% short-SKILL rate applies here too.
DEFAULT_TRAIN_SHORT_RATE = 0.144

# --- reproducibility -------------------------------------------------------

RANDOM_SEED = 42
