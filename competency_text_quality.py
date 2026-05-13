"""
competency_text_quality.py

Phase 2.5b — text-quality post-check for competency title / description /
rationale fields. Complements the grounding gate by catching failure modes
that the grounding gate cannot see, e.g. the DeepSeek-V3.2 "religious" token-
corruption bug (substitutes " and " → "religious" in 30+ places per response).

The grounding gate verifies *related_skills* against source_sentences but does
NOT inspect the LLM-authored title/description/rationale text. A competency
with hallucinated text but real skills can pass the grounding gate. This
module fills that gap with cheap heuristic checks.

Checks (in order, cheapest first):

    1. **Common-English-word density** — every English sentence over ~50
       chars should contain at least one of {the, a, an, and, or, of, in,
       for, to, with}. Zero stop-words in a 50+ char string is a strong
       signal of token corruption or non-English content.

    2. **Suspicious-word substitution** — known token-corruption patterns
       (e.g. "religious" appearing >= 3 times in a single competency's text
       fields is the DeepSeek-V3.2 signature; "religious" appearing once
       legitimately is fine).

    3. **Language detection** — if `langdetect` is installed, verify the
       competency text is primarily English (or Indonesian, since the
       pipeline also supports Indonesian curricula). Silent fallback if
       langdetect is unavailable.

    4. **Length sanity** — title 10-200 chars, description 30-500 chars.
       Very short or very long fields are usually broken.

Public API:

    from competency_text_quality import check_text_quality, TextQualityConfig

    passing, failing, report = check_text_quality(competencies, config=TextQualityConfig())

Each competency is mutated to carry a `text_quality_passed: bool` field
(stashed inside grounding_reasoning so we don't need to extend the schema
again). For now, this module is OPT-IN — callers must invoke it explicitly.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from competency_v2_schema import CompetencyV2

logger = logging.getLogger(__name__)


CHECK_VERSION = "v2.5b.0"


@dataclass
class TextQualityConfig:
    """Knobs for the text-quality post-check."""

    # Common English stopwords expected in any 50+ char English sentence
    expected_english_stopwords: tuple = (
        "the", "a", "an", "and", "or", "of", "in", "for", "to", "with",
        "is", "are", "as", "by", "on", "at", "from", "this", "that",
    )

    # Suspicious-word substitutions (the DeepSeek-V3.2 "religious" signature
    # is the canonical example; add more as we observe them).
    suspicious_repeated_words: tuple = ("religious",)
    suspicious_word_threshold: int = 3   # >= N occurrences in title+desc+rationale flags

    # Length sanity
    min_title_chars: int = 10
    max_title_chars: int = 200
    min_description_chars: int = 30
    max_description_chars: int = 500

    # Language detection — set to None to disable
    expected_languages: tuple = ("en", "id")  # English or Indonesian
    enable_langdetect: bool = True

    # The smallest English-text length we require stopwords on
    min_text_for_stopword_check: int = 50


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _check_stopword_density(text: str, config: TextQualityConfig) -> Optional[str]:
    """Return a violation message if too few stopwords for the text length."""
    if not text or len(text) < config.min_text_for_stopword_check:
        return None
    toks = set(_tokens(text))
    hits = toks & set(config.expected_english_stopwords)
    if not hits:
        return (
            f"zero common-English-stopwords in {len(text)}-char text "
            f"(probable token corruption or non-English content)"
        )
    return None


def _check_suspicious_words(combined_text: str, config: TextQualityConfig) -> Optional[str]:
    """Return a violation if any suspicious word appears >= threshold times."""
    if not combined_text:
        return None
    low = combined_text.lower()
    for word in config.suspicious_repeated_words:
        n = len(re.findall(rf"\b{re.escape(word)}\b", low))
        if n >= config.suspicious_word_threshold:
            return (
                f"suspicious word '{word}' appears {n} times "
                f"(threshold={config.suspicious_word_threshold}) — "
                f"likely token-corruption bug"
            )
    return None


def _check_language(text: str, config: TextQualityConfig) -> Optional[str]:
    if not config.enable_langdetect or not text:
        return None
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic
        lang = detect(text)
    except Exception:
        # silent fallback
        return None
    if lang not in config.expected_languages:
        return f"detected language '{lang}' not in expected {config.expected_languages}"
    return None


def _check_lengths(comp: CompetencyV2, config: TextQualityConfig) -> List[str]:
    out = []
    tl = len((comp.title or "").strip())
    if not (config.min_title_chars <= tl <= config.max_title_chars):
        out.append(f"title length {tl} outside [{config.min_title_chars}, {config.max_title_chars}]")
    dl = len((comp.description or "").strip())
    if not (config.min_description_chars <= dl <= config.max_description_chars):
        out.append(
            f"description length {dl} outside [{config.min_description_chars}, "
            f"{config.max_description_chars}]"
        )
    return out


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


def check_one(comp: CompetencyV2, config: TextQualityConfig) -> List[str]:
    """Run all text-quality checks on one competency.

    Returns a list of human-readable violation strings (empty = pass).
    """
    violations = []
    combined = " ".join([comp.title or "", comp.description or "", comp.rationale or ""])

    # 1. Suspicious-word substitutions (catches the "religious" bug class)
    v = _check_suspicious_words(combined, config)
    if v:
        violations.append(v)

    # 2. Per-field stopword density
    for field_name, text in (("description", comp.description), ("rationale", comp.rationale)):
        v = _check_stopword_density(text or "", config)
        if v:
            violations.append(f"{field_name}: {v}")

    # 3. Language detection (description + rationale combined)
    v = _check_language(combined, config)
    if v:
        violations.append(v)

    # 4. Length sanity
    violations.extend(_check_lengths(comp, config))

    return violations


def check_text_quality(
    competencies: Sequence[CompetencyV2],
    config: Optional[TextQualityConfig] = None,
) -> Tuple[List[CompetencyV2], List[CompetencyV2], dict]:
    """Apply text-quality checks across a batch.

    Returns (passing, failing, report). Each competency in `failing` has its
    `grounding_reasoning` field appended with the violation list so audit logs
    capture the rejection reason (we deliberately avoid adding new schema
    fields for an opt-in check).
    """
    if config is None:
        config = TextQualityConfig()

    passing = []
    failing = []
    per_comp_violations = {}

    for c in competencies:
        v = check_one(c, config)
        per_comp_violations[c.id] = v
        if v:
            tag = f"[TEXT-QUALITY-FAIL ({CHECK_VERSION})] " + " ; ".join(v)
            c.grounding_reasoning = ((c.grounding_reasoning or "") + " | " + tag).strip(" |")
            failing.append(c)
        else:
            passing.append(c)

    report = {
        "version": CHECK_VERSION,
        "n_evaluated": len(competencies),
        "n_passed": len(passing),
        "n_failed": len(failing),
        "per_competency_violations": per_comp_violations,
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.__dict__.items()},
    }
    return passing, failing, report
