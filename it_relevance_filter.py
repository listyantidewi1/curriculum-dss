"""
it_relevance_filter.py

Zero-shot LLM filter that decides whether a job posting / sentence is
software-engineering / IT content. Used upstream of skill extraction so that
non-IT postings (and non-IT boilerplate within IT postings) never reach
Skill-LLM.

Two granularities — both follow the same yes/no batched pattern as
`sentence_relevance_filter.py`:

  - classify_jobs(descriptions, ...): one verdict per posting. Cheap upfront
    filter that culls obvious non-IT postings (retail, logistics, food
    service) before they fan out into sentences.

  - classify_it_sentences(sentences, ...): one verdict per sentence. Catches
    the case the v8.1 climbing competency exploited — a software-engineer
    posting that contains "Physical Requirements" boilerplate copy-pasted
    from a template. The job passes the job-level gate, but the climbing
    sentences fail the sentence-level gate.

Uncertainty handling: anything the LLM doesn't classify as a clean YES/NO
defaults to KEEP (same precedent as `sentence_relevance_filter`). The
downstream pipeline can re-filter; spuriously dropping a real IT skill is
costlier than letting one non-IT line through.

Cost control: batched LLM calls + persistent SHA-256 cache so re-runs are
free.
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
DEFAULT_JOB_BATCH_SIZE = 10           # job descriptions are longer; smaller batches
DEFAULT_SENTENCE_BATCH_SIZE = 30
DEFAULT_TEMPERATURE = 0.0
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2

# v9 sprint (2026-05-13): the climbing/pizza-assembly failure mode in v8.1
# motivated this filter. We classify with explicit positive examples (what
# IS IT/SE content) so the model doesn't drift into a vague "professional"
# definition that lets retail-with-IT-buzzwords slip through.

_JOB_SYSTEM_PROMPT = (
    "For each numbered JOB POSTING, answer YES if the posting is for a "
    "software-engineering, IT, data, security, or other digital/computing "
    "role (developer, engineer, analyst, architect, administrator, "
    "scientist, devops/sre, ml, qa, designer of digital products). "
    "Answer NO for retail, food service, hospitality, logistics, delivery, "
    "construction, manual labour, healthcare practitioner, sales/marketing "
    "(unless explicitly a software role), education non-tech, or other "
    "non-digital roles, even if the posting mentions 'using a computer' or "
    "'data entry'. Output exactly one YES or NO per line, in the same order, "
    "with no numbering, no commentary, and no extra text."
)


_SENTENCE_SYSTEM_PROMPT = (
    "For each numbered sentence taken from a job posting, answer YES if the "
    "sentence describes software-engineering, IT, data, security, or other "
    "digital/computing work, tools, methods, or required knowledge. Answer "
    "NO for physical-requirements boilerplate (climbing, lifting, standing, "
    "stooping, walking), benefits, EEO statements, location/logistics, "
    "company descriptions, application instructions, salary, non-tech "
    "industry-specific tasks (retail, food service, manual labour), and any "
    "sentence whose substance is not about IT work. Output exactly one YES "
    "or NO per line, in the same order, with no numbering, no commentary, "
    "and no extra text."
)


# --------------------------------------------------------------------------- #
# Cache (same shape as sentence_relevance_filter)
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
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# OpenRouter client (deepseek-chat is great at short classification tasks +
# cheap; ~$0.27/M input tokens at time of writing)
# --------------------------------------------------------------------------- #


def _load_openrouter_client() -> OpenAI:
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        key_path = Path("api_keys") / "OpenRouter.txt"
        try:
            api_key = key_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"OpenRouter API key not found. Set OPENROUTER_API_KEY or "
                f"create {key_path}."
            ) from exc
    if not api_key:
        raise RuntimeError("OpenRouter API key is empty.")
    return OpenAI(api_key=api_key, base_url=base_url)


# --------------------------------------------------------------------------- #
# Verdict parsing — identical to sentence_relevance_filter
# --------------------------------------------------------------------------- #


_VERDICT_LINE_RE = re.compile(r"^\s*(?:[\d]+[\.\)]\s*)?(YES|NO)\b", re.IGNORECASE)


def _parse_verdicts(content: str, expected: int) -> List[bool]:
    """One YES/NO per non-empty line. Unclear lines → KEEP. Short responses
    pad with KEEP; long responses truncate."""
    verdicts: List[bool] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _VERDICT_LINE_RE.match(line)
        verdicts.append(m.group(1).upper() == "YES" if m else True)
        if len(verdicts) >= expected:
            break
    while len(verdicts) < expected:
        verdicts.append(True)
    return verdicts[:expected]


def _classify_batch(
    client: OpenAI,
    texts: List[str],
    *,
    system_prompt: str,
    model: str,
    temperature: float,
) -> List[bool]:
    if not texts:
        return []
    user_prompt = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    last_error: Optional[str] = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2 * len(texts) + 16,
            )
            content = (resp.choices[0].message.content or "").strip()
            return _parse_verdicts(content, len(texts))
        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE ** (attempt + 1)
                print(
                    f"[WARN] it_relevance_filter: attempt {attempt + 1} failed "
                    f"({last_error}); retrying in {wait}s..."
                )
                time.sleep(wait)
    print(
        f"[WARN] it_relevance_filter: all {1 + MAX_RETRIES} attempts failed "
        f"({last_error}); defaulting to KEEP for batch of {len(texts)}."
    )
    return [True] * len(texts)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _truncate(text: str, max_chars: int) -> str:
    """Truncate a long job description so batches don't blow the context
    window. Job postings are bimodal: medium (1-2k chars) or huge (10k+
    chars of boilerplate). Truncating to ~2000 chars keeps the front matter
    (responsibilities, requirements) which is what matters for IT/non-IT
    classification."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " […]"


def classify_jobs(
    descriptions: Iterable[str],
    *,
    cache_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_JOB_BATCH_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    client: Optional[OpenAI] = None,
    max_chars_per_job: int = 2000,
) -> List[bool]:
    """Classify each job description as IT-relevant (True) or not (False).

    Truncates each description to `max_chars_per_job` before sending — the
    first ~2k chars usually contain enough role context for an unambiguous
    verdict, and keeps batches inside the model's context window.

    See module docstring for cost / cache behaviour.
    """
    return _classify(
        list(descriptions),
        system_prompt=_JOB_SYSTEM_PROMPT,
        cache_path=cache_path,
        model=model,
        batch_size=batch_size,
        temperature=temperature,
        client=client,
        max_chars=max_chars_per_job,
        cache_namespace="job",
    )


def classify_it_sentences(
    sentences: Iterable[str],
    *,
    cache_path: Optional[Path] = None,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_SENTENCE_BATCH_SIZE,
    temperature: float = DEFAULT_TEMPERATURE,
    client: Optional[OpenAI] = None,
) -> List[bool]:
    """Classify each sentence as IT-content (True) or not (False).

    Independent of the existing `sentence_relevance_filter.classify_sentences`
    — that one asks 'is this skill-bearing?', this one asks 'is this about IT
    work?'. The pipeline can run both (in either order) for tighter quality.
    """
    return _classify(
        list(sentences),
        system_prompt=_SENTENCE_SYSTEM_PROMPT,
        cache_path=cache_path,
        model=model,
        batch_size=batch_size,
        temperature=temperature,
        client=client,
        max_chars=None,
        cache_namespace="sent",
    )


def _classify(
    texts: List[str],
    *,
    system_prompt: str,
    cache_path: Optional[Path],
    model: str,
    batch_size: int,
    temperature: float,
    client: Optional[OpenAI],
    max_chars: Optional[int],
    cache_namespace: str,
) -> List[bool]:
    """Shared batching + caching driver for both job and sentence levels."""
    texts = [str(t) for t in texts]
    if not texts:
        return []

    # Optional truncation (job-level only).
    prepared = [_truncate(t, max_chars) if max_chars else t for t in texts]

    # Cache by SHA-256 of (namespace + prepared text). Namespace keeps job
    # verdicts separate from sentence verdicts even if texts collide.
    cache = _load_cache(cache_path)
    digests = [_sha256(f"{cache_namespace}\x00{t.strip()}") for t in prepared]
    verdicts: List[Optional[bool]] = [
        cache.get(d) if isinstance(cache.get(d), bool) else None for d in digests
    ]
    pending_idx = [i for i, v in enumerate(verdicts) if v is None]
    if not pending_idx:
        return [bool(v) if v is not None else True for v in verdicts]

    if client is None:
        client = _load_openrouter_client()
    for start in range(0, len(pending_idx), batch_size):
        batch_indices = pending_idx[start : start + batch_size]
        batch_texts = [prepared[i] for i in batch_indices]
        batch_verdicts = _classify_batch(
            client, batch_texts,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
        )
        for i, verdict in zip(batch_indices, batch_verdicts):
            verdicts[i] = verdict
            cache[digests[i]] = verdict
        _save_cache(cache_path, cache)

    return [bool(v) if v is not None else True for v in verdicts]


__all__ = [
    "classify_jobs",
    "classify_it_sentences",
    "DEFAULT_MODEL",
    "DEFAULT_JOB_BATCH_SIZE",
    "DEFAULT_SENTENCE_BATCH_SIZE",
]
