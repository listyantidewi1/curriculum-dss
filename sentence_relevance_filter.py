"""
sentence_relevance_filter.py

Zero-shot LLM sentence relevance filter for the preprocessing stage.

Implements pipeline-redesign-v2 Phase 1.2 (Req 4): before extraction, classify
each sentence as `relevant` (skill/knowledge-bearing) or `irrelevant` (company
descriptions, benefits, logistics, etc.) and drop the latter.

Targets:
    precision >= 0.90 for the "relevant" class (don't discard true skill sentences)
    recall    >= 0.85 for the "relevant" class (remove most irrelevant content)

Uncertainty handling: when the LLM returns anything other than YES / NO for a
sentence, default to KEEP the sentence (Req 4.6 - if uncertain, retain). This
biases precision over recall on the irrelevant class, which is the safe choice
for an upstream filter feeding extraction.

Cost control:
    - Sentences are batched (default >= 30 per LLM call) to amortize prompt overhead.
    - Persistent on-disk cache keyed by SHA-256 of the (stripped) sentence text
      so re-runs over the same corpus cost zero LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI

DEFAULT_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_BATCH_SIZE = 30
DEFAULT_TEMPERATURE = 0.0
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2  # seconds

SYSTEM_PROMPT = (
    "For each numbered sentence, answer YES if it states a skill, knowledge, "
    "technology, or competency requirement (something a candidate must know or "
    "be able to do); otherwise answer NO. NO covers company descriptions, "
    "benefits, salary, location/logistics, application instructions, and "
    "boilerplate. Output exactly one YES or NO per line, in the same order as "
    "the input, with no numbering, no commentary, and no extra text."
)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Optional[Path]) -> dict:
    if cache_path is None or not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: Optional[Path], cache: dict) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# OpenRouter client
# --------------------------------------------------------------------------- #


def _load_openrouter_client() -> OpenAI:
    """Mirror generate_competencies.load_openrouter_client(): env var first,
    fallback to api_keys/jatevo.txt (the project's OpenRouter key file)."""
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        key_path = Path("api_keys") / "jatevo.txt"
        try:
            api_key = key_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OpenRouter API key not found. Set OPENROUTER_API_KEY "
                f"or create {key_path}."
            ) from exc
    if not api_key:
        raise RuntimeError("OpenRouter API key is empty.")
    return OpenAI(api_key=api_key, base_url=base_url)


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #


_VERDICT_LINE_RE = re.compile(r"^\s*(?:[\d]+[\.\)]\s*)?(YES|NO)\b", re.IGNORECASE)


def _classify_batch(
    client: OpenAI,
    sentences: List[str],
    *,
    model: str,
    temperature: float,
) -> List[bool]:
    """Classify a single batch via one LLM call. Returns a list of bools aligned
    with `sentences`. Items the LLM didn't classify clearly default to True (keep).
    """
    if not sentences:
        return []

    user_prompt = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

    last_error: Optional[str] = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2 * len(sentences) + 16,
            )
            content = (resp.choices[0].message.content or "").strip()
            return _parse_verdicts(content, len(sentences))
        except Exception as exc:  # network / rate limit / parse failure
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                print(
                    f"[WARN] sentence_relevance_filter: attempt {attempt + 1} "
                    f"failed ({last_error}); retrying in {wait}s..."
                )
                time.sleep(wait)

    # Hard failure: keep everything (Req 4.6 - retain when uncertain).
    print(
        f"[WARN] sentence_relevance_filter: all {1 + MAX_RETRIES} attempts "
        f"failed ({last_error}); defaulting to KEEP for batch of "
        f"{len(sentences)} sentences."
    )
    return [True] * len(sentences)


def _parse_verdicts(content: str, expected: int) -> List[bool]:
    """Parse YES/NO verdicts, one per non-empty line. Lines that don't match
    YES/NO default to True. If we got fewer lines than expected, pad with True.
    If more, truncate."""
    verdicts: List[bool] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _VERDICT_LINE_RE.match(line)
        if m:
            verdicts.append(m.group(1).upper() == "YES")
        else:
            # Couldn't parse a verdict on this line - retain (Req 4.6).
            verdicts.append(True)
        if len(verdicts) >= expected:
            break
    # Pad short responses with KEEP.
    while len(verdicts) < expected:
        verdicts.append(True)
    return verdicts[:expected]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def classify_sentences(
    sentences: Iterable[str],
    *,
    cache_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    client: Optional[OpenAI] = None,
) -> List[bool]:
    """Classify each sentence as relevant (True) or irrelevant (False).

    Parameters
    ----------
    sentences : iterable of str
        Sentence texts to classify, in order.
    cache_path : Path, optional
        On-disk SHA-256 cache so re-runs cost zero LLM calls. When None, no
        caching is performed.
    model : str
        OpenRouter model id.
    batch_size : int
        Number of sentences sent per LLM call. Plan minimum is 30; raise to
        amortize prompt overhead, lower if context windows complain.
    temperature : float
        LLM temperature; default 0.0 keeps verdicts reproducible.
    client : OpenAI, optional
        Reuse an existing OpenRouter client. When None, one is loaded lazily
        only if there are uncached sentences to classify.

    Returns
    -------
    list of bool aligned with the input sentences. True = keep (relevant),
    False = drop (irrelevant). Ambiguous LLM responses default to True
    (Req 4.6: retain when uncertain).
    """
    sentences = [str(s) for s in sentences]
    n = len(sentences)
    if n == 0:
        return []

    cache = _load_cache(cache_path)

    # Decide which sentences need an LLM call.
    digests: List[str] = [_sha256(s.strip()) for s in sentences]
    verdicts: List[Optional[bool]] = [
        cache.get(d) if isinstance(cache.get(d), bool) else None for d in digests
    ]
    pending_idx = [i for i, v in enumerate(verdicts) if v is None]

    if pending_idx:
        if client is None:
            client = _load_openrouter_client()
        # Batch the pending sentences; keep batches stable so the cache file
        # grows monotonically even if a later batch fails.
        for start in range(0, len(pending_idx), batch_size):
            batch_indices = pending_idx[start : start + batch_size]
            batch_sentences = [sentences[i] for i in batch_indices]
            batch_verdicts = _classify_batch(
                client, batch_sentences, model=model, temperature=temperature
            )
            for i, verdict in zip(batch_indices, batch_verdicts):
                verdicts[i] = verdict
                cache[digests[i]] = verdict
            _save_cache(cache_path, cache)

    # By construction every slot is now bool, but be defensive.
    return [bool(v) if v is not None else True for v in verdicts]


__all__ = ["classify_sentences", "DEFAULT_MODEL", "DEFAULT_BATCH_SIZE"]
