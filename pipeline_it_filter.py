"""
pipeline_it_filter.py — v9 sprint glue between the IT-relevance classifier
(`it_relevance_filter.py`) and the v2 pipeline's loaded item list.

Two passes:

  1. Job-level: for every unique source job_id, look up its description from
     `english_jobs.csv` and ask the LLM "is this an IT job?". Drop every
     item whose job_id failed.

  2. Sentence-level: for every unique source sentence_text in the surviving
     items, ask the LLM "is this sentence about IT work?". Drop every item
     whose sentence failed.

The two-pass design is cost-aware: most non-IT noise is in non-IT jobs that
the job-level pass kills cheaply (one LLM call per job, not per sentence).
The sentence-level pass then catches the residual case — an IT job that
contains boilerplate "Physical Requirements" or "EEO" paragraphs — at higher
fidelity.

Returns the filtered items list plus an audit dict suitable for serialising
to `it_relevance_audit.json` alongside the run's clusters.json.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_job_descriptions(raw_jobs_csv: Path) -> dict:
    """Build {job_id -> description} from the scraper output CSV."""
    if not raw_jobs_csv or not raw_jobs_csv.exists():
        return {}
    out: dict = {}
    with open(raw_jobs_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            jid = (row.get("id") or "").strip()
            desc = (row.get("description") or "").strip()
            if jid and desc:
                out[jid] = desc
    return out


def _job_id_from_sentence_id(sid: str) -> str:
    if not sid or "_" not in sid:
        return sid or ""
    head, _sep, tail = sid.rpartition("_")
    return head if tail.isdigit() else sid


def apply_it_relevance_filter(
    items: List,
    *,
    raw_jobs_csv: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[List, dict]:
    """Run the two-pass IT-relevance filter on a SkillItem/KnowledgeItem list.

    Args:
        items: list of SkillItem / KnowledgeItem (output of `load_real_items`).
        raw_jobs_csv: scraper output CSV with `id` + `description` columns.
            When missing, the job-level pass is skipped — only sentence-level
            classification runs.
        cache_dir: directory for the on-disk SHA-256 cache. When None, no
            caching is performed (re-runs cost LLM calls).

    Returns:
        (filtered_items, audit_dict). audit_dict has:
          "summary":          aggregate counts
          "dropped_jobs":     [{"job_id", "n_items_dropped", "description_excerpt"}, ...]
          "dropped_sentences":[{"sentence_id", "sentence_text"}, ...]
    """
    from it_relevance_filter import classify_jobs, classify_it_sentences

    n_in = len(items)
    summary: dict = {
        "n_items_in": n_in,
        "n_items_kept": n_in,
        "n_jobs_total": 0,
        "n_jobs_it": 0,
        "n_jobs_dropped": 0,
        "n_sentences_total": 0,
        "n_sentences_it": 0,
        "n_sentences_dropped": 0,
    }
    audit: dict = {"summary": summary, "dropped_jobs": [], "dropped_sentences": []}
    if not items:
        return items, audit

    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Pass 1 — job-level
    # ------------------------------------------------------------------ #
    job_descriptions = _load_job_descriptions(raw_jobs_csv) if raw_jobs_csv else {}
    unique_job_ids = {
        _job_id_from_sentence_id(getattr(it, "sentence_id", "") or "")
        for it in items
    }
    unique_job_ids.discard("")

    job_verdicts: dict = {}
    if job_descriptions and unique_job_ids:
        ordered_jobs = sorted(unique_job_ids)
        descs = [job_descriptions.get(jid, "") for jid in ordered_jobs]
        # Jobs with no description in english_jobs.csv default to KEEP so we
        # don't silently drop them — caller can spot in the audit if needed.
        verdicts = classify_jobs(
            descs,
            cache_path=cache_dir / "jobs.json" if cache_dir else None,
        )
        # If a job has empty description, the classifier still got "" and
        # may have voted NO; override to KEEP to be safe.
        for jid, d, v in zip(ordered_jobs, descs, verdicts):
            job_verdicts[jid] = v if d else True
        summary["n_jobs_total"] = len(ordered_jobs)
        summary["n_jobs_it"] = sum(1 for v in job_verdicts.values() if v)
        summary["n_jobs_dropped"] = summary["n_jobs_total"] - summary["n_jobs_it"]
        logger.info(
            "IT filter (job-level): %d / %d jobs classified as IT",
            summary["n_jobs_it"], summary["n_jobs_total"],
        )
    else:
        # Without descriptions, we can't classify jobs — keep them all and
        # let the sentence-level pass do the work.
        logger.info(
            "IT filter (job-level): SKIPPED (raw_jobs_csv missing or empty)"
        )
        for jid in unique_job_ids:
            job_verdicts[jid] = True

    # Record dropped-job audit entries
    items_per_job: dict = {}
    for it in items:
        jid = _job_id_from_sentence_id(getattr(it, "sentence_id", "") or "")
        items_per_job[jid] = items_per_job.get(jid, 0) + 1
    for jid, v in job_verdicts.items():
        if not v:
            desc = job_descriptions.get(jid, "")
            audit["dropped_jobs"].append({
                "job_id": jid,
                "n_items_dropped": items_per_job.get(jid, 0),
                "description_excerpt": desc[:300] + ("…" if len(desc) > 300 else ""),
            })

    # Apply job-level filter
    survivors_after_jobs = [
        it for it in items
        if job_verdicts.get(_job_id_from_sentence_id(getattr(it, "sentence_id", "") or ""), True)
    ]

    # ------------------------------------------------------------------ #
    # Pass 2 — sentence-level
    # ------------------------------------------------------------------ #
    unique_sentences: dict = {}  # sentence_text -> list of items
    for it in survivors_after_jobs:
        stxt = (getattr(it, "sentence_text", "") or "").strip()
        if stxt:
            unique_sentences.setdefault(stxt, []).append(it)

    sentence_verdicts: dict = {}
    if unique_sentences:
        ordered_sents = list(unique_sentences.keys())
        verdicts = classify_it_sentences(
            ordered_sents,
            cache_path=cache_dir / "sentences.json" if cache_dir else None,
        )
        sentence_verdicts = dict(zip(ordered_sents, verdicts))
        summary["n_sentences_total"] = len(ordered_sents)
        summary["n_sentences_it"] = sum(1 for v in verdicts if v)
        summary["n_sentences_dropped"] = summary["n_sentences_total"] - summary["n_sentences_it"]
        logger.info(
            "IT filter (sentence-level): %d / %d unique sentences classified as IT",
            summary["n_sentences_it"], summary["n_sentences_total"],
        )

    # Record dropped-sentence audit entries (cap so the audit file stays sane)
    DROPPED_SENTENCE_CAP = 1000
    for stxt, v in sentence_verdicts.items():
        if not v:
            sample_item = unique_sentences[stxt][0]
            audit["dropped_sentences"].append({
                "sentence_id": getattr(sample_item, "sentence_id", "") or "",
                "sentence_text": stxt[:500] + ("…" if len(stxt) > 500 else ""),
                "n_items_dropped": len(unique_sentences[stxt]),
            })
            if len(audit["dropped_sentences"]) >= DROPPED_SENTENCE_CAP:
                break

    # Apply sentence-level filter
    final = [
        it for it in survivors_after_jobs
        if sentence_verdicts.get(
            (getattr(it, "sentence_text", "") or "").strip(),
            True,
        )
    ]

    summary["n_items_kept"] = len(final)
    return final, audit


__all__ = ["apply_it_relevance_filter"]
