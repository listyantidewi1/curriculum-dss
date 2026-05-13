"""
translator.py

LLM-backed Indonesian → English translator for the v2 pipeline.

Used in two places per the sprint plan:
  1. **Job-posting ingestion**: when Indonesian job postings flow through
     `preprocess_jobs_pipeline.py`, each Indonesian sentence is translated
     to English BEFORE the relevance filter and Skill-LLM extraction.
     Preserves the original sentence in `sentence_original` for provenance.
  2. **Curriculum upload (dashboard)**: when a user uploads an Indonesian
     curriculum, sections are translated to English before SBERT matching
     against competencies (since competencies are English-language).

Routing: uses our standard LLM router (Jatevo for GPT, OpenRouter for
everything else). Default model is gpt-5.4-mini (matches Phase 2.2's
default — cheap, fast, no token-corruption bugs).

Caching: per-text SHA-256 cache to disk so retries are free.

Public API:

    from translator import translate_to_english, langdetect_id

    if langdetect_id(text) == "id":
        text_en = translate_to_english(text)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


TRANSLATOR_VERSION = "v2.0.0"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_CACHE_PATH = Path("cache") / "translations.json"


# --------------------------------------------------------------------------- #
# Language detection
# --------------------------------------------------------------------------- #


def langdetect_lang(text: str) -> str:
    """Detect language code via `langdetect`. Returns ISO 639-1 ("en", "id", ...)
    or "unknown" if detection fails or langdetect is unavailable.
    """
    if not text or not text.strip():
        return "unknown"
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


def langdetect_id(text: str) -> bool:
    """Shortcut: True if `text` is detected as Indonesian."""
    return langdetect_lang(text) == "id"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


_CACHE: Optional[dict] = None


def _load_cache(cache_path: Path) -> dict:
    global _CACHE
    if _CACHE is None:
        if cache_path.exists():
            try:
                _CACHE = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{text.strip()}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = (
    "You are a professional translator specializing in job postings and "
    "vocational curriculum documents. Translate the user's input from "
    "Indonesian to natural English. Preserve technical terms that are "
    "commonly used in English within the Indonesian software-engineering "
    "context (e.g., 'developer', 'database', 'API', 'cloud computing'). "
    "Output ONLY the translation. No commentary, no quote marks, no extra "
    "text. If the input is already English, return it unchanged."
)


# v9 sprint: reverse direction for the dashboard's Bahasa Indonesia toggle.
# Translates English competency text (titles, descriptions, rationales) back
# to Indonesian for users browsing in Bahasa. Same preservation rule —
# English technical terms (DevOps, CI/CD, API, etc.) stay verbatim because
# that's how Indonesian IT practitioners actually use them.
SYSTEM_PROMPT_EN_TO_ID = (
    "You are a professional translator specializing in software-engineering "
    "and vocational curriculum content. Translate the user's input from "
    "English to natural Bahasa Indonesia. KEEP English technical terms "
    "verbatim in their original casing — do NOT translate terms like "
    "'DevOps', 'CI/CD', 'API', 'cloud computing', 'machine learning', "
    "'database', 'framework', 'JavaScript', 'Python', 'AWS', 'Lambda', etc. "
    "These terms are used in English in Indonesian IT contexts. Output ONLY "
    "the translation. No commentary, no quote marks, no extra text. If the "
    "input is already Indonesian, return it unchanged."
)


def _call_llm(text: str, model: str, system_prompt: str = SYSTEM_PROMPT, timeout: float = 60.0) -> str:
    from llm_client_router import get_client_for_model, strip_provider_prefix

    client, _provider = get_client_for_model(model, request_timeout=timeout)
    api_model = strip_provider_prefix(model)
    resp = client.chat.completions.create(
        model=api_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=max(256, int(len(text) * 1.5)),
    )
    out = (resp.choices[0].message.content or "").strip()
    # Defensive: strip surrounding quotes if the model added them
    if (out.startswith('"') and out.endswith('"')) or (out.startswith("'") and out.endswith("'")):
        out = out[1:-1]
    return out


def translate_to_english(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    force: bool = False,
    skip_if_english: bool = True,
) -> str:
    """Translate `text` to English. Returns the translated string.

    - If `skip_if_english` and langdetect classifies the input as English,
      returns the original unchanged (no LLM call).
    - Cached per (model, text) via SHA-256.
    - Falls back to returning the original text on any error.
    """
    if not text or not text.strip():
        return text

    if skip_if_english and langdetect_lang(text) == "en":
        return text

    cache = _load_cache(cache_path)
    k = _key(text, model)
    if not force and k in cache:
        return cache[k]

    try:
        out = _call_llm(text, model)
        cache[k] = out
        _save_cache(cache_path, cache)
        return out
    except Exception as e:
        logger.warning("translation failed (%s); returning original text", e)
        return text


def translate_batch(
    texts: list,
    *,
    model: str = DEFAULT_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    skip_if_english: bool = True,
    progress: bool = False,
) -> list:
    """Translate a list of texts to English. Hits the per-text cache so most
    calls in a repeated run are free.
    """
    out = []
    n = len(texts)
    for i, t in enumerate(texts):
        translated = translate_to_english(
            t, model=model, cache_path=cache_path, skip_if_english=skip_if_english
        )
        out.append(translated)
        if progress and (i % 20 == 0 or i == n - 1):
            print(f"  translate_batch: {i+1}/{n}")
    return out


# --------------------------------------------------------------------------- #
# v9 sprint: reverse direction (EN → ID) for dashboard display
# --------------------------------------------------------------------------- #


def translate_to_indonesian(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
    force: bool = False,
    skip_if_indonesian: bool = True,
) -> str:
    """Translate `text` to Bahasa Indonesia, keeping English technical terms
    verbatim. Used by the dashboard language toggle and the post-Phase-2.2
    competency translation pass.

    Cached per (model + target='id' + text) so the EN→ID cache namespace is
    independent of the ID→EN cache (same model, same text would otherwise
    collide).
    """
    if not text or not text.strip():
        return text

    if skip_if_indonesian and langdetect_lang(text) == "id":
        return text

    cache = _load_cache(cache_path)
    # Namespace the cache key with "en2id" so it doesn't collide with the
    # id-to-en direction.
    k = hashlib.sha256(f"en2id\x00{model}\x00{text.strip()}".encode("utf-8")).hexdigest()
    if not force and k in cache:
        return cache[k]

    try:
        out = _call_llm(text, model, system_prompt=SYSTEM_PROMPT_EN_TO_ID)
        cache[k] = out
        _save_cache(cache_path, cache)
        return out
    except Exception as e:
        logger.warning("EN→ID translation failed (%s); returning original text", e)
        return text


def translate_competency_to_indonesian(
    competency: dict,
    *,
    model: str = DEFAULT_MODEL,
    cache_path: Path = DEFAULT_CACHE_PATH,
) -> dict:
    """Translate the user-facing fields of one competency dict to Indonesian.

    Adds `_id`-suffixed keys without mutating the originals:
        title          → title_id
        description    → description_id
        rationale      → rationale_id
        future_relevance → future_relevance_id
        soft_skills_description → soft_skills_description_id
        related_skills → related_skills_id (per-item)
        soft_skills_required → soft_skills_required_id (per-item)

    Leaves source_sentences alone — those stay in their original language
    so provenance is honest about what was actually in the posting.
    """
    out = dict(competency)
    for field in ("title", "description", "rationale", "future_relevance",
                  "soft_skills_description"):
        v = competency.get(field) or ""
        out[f"{field}_id"] = translate_to_indonesian(
            v, model=model, cache_path=cache_path,
        ) if v else ""
    # Per-list fields: translate each item separately so the cache hits per skill
    for field in ("related_skills", "soft_skills_required"):
        items = competency.get(field) or []
        out[f"{field}_id"] = [
            translate_to_indonesian(it, model=model, cache_path=cache_path)
            for it in items
        ]
    return out
