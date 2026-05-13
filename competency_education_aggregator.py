"""
competency_education_aggregator.py

Phase 2.4 — aggregate education-level requirements across each competency's
source job postings.

`preprocess_jobs_pipeline.py` already extracts a `min_education_kkni` (1-9)
and an `education_labels` string (e.g. "D3; S1/D4; S2/Magister") per job
posting and saves them into `jobs_metadata.csv`. This module joins on
`source_job_ids` to compute, per competency:

    education_levels_demanded:  Dict[stage, fraction]   e.g. {"SMK": 0.6, "D3": 0.3, "S1": 0.1}
    education_kkni_demanded:    Dict[kkni_level, fraction]
    n_jobs_with_education:      int

A curriculum designer reading the dashboard then sees, for each competency:
"60% of the job postings that demanded this skill set were targeting SMK
graduates; 30% targeting D3; 10% S1." That maps directly to "which programmes
should adopt this competency."

ISCED mapping note: the Indonesian KKNI maps cleanly to ISCED:
    KKNI 3 (SMK)   ≈ ISCED 3 (Upper secondary)
    KKNI 4 (D1)    ≈ ISCED 4 (Post-secondary non-tertiary)
    KKNI 5 (D3)    ≈ ISCED 5 (Short-cycle tertiary)
    KKNI 6 (S1/D4) ≈ ISCED 6 (Bachelor)
    KKNI 7 (Profesi)≈ ISCED 7 (Master, professional)
    KKNI 8 (S2)    ≈ ISCED 7 (Master)
    KKNI 9 (S3)    ≈ ISCED 8 (Doctoral)
Non-ID curriculum designers can use the ISCED view of the same distribution.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

from competency_v2_schema import CompetencyV2
from kkni import KKNI_LEVELS

logger = logging.getLogger(__name__)


AGGREGATOR_VERSION = "v2.4.0"


# KKNI → ISCED mapping (used in to_isced helper, see end of file)
KKNI_TO_ISCED = {
    1: 1,   # SD / SMP (lower)
    2: 2,   # SMA general
    3: 3,   # SMK / D1
    4: 4,   # D2
    5: 5,   # D3
    6: 6,   # S1 / D4
    7: 7,   # Profesi
    8: 7,   # S2 Magister
    9: 8,   # S3 Doktor
}


def _label_for_kkni(level: int) -> str:
    """Indonesian stage label that's most representative for a KKNI level.

    For levels with multiple stages (e.g. 3 = ['SMK', 'D1']) we pick the
    most curriculum-relevant one for vocational software-engineering
    (SMK is overwhelmingly the Indonesian SE-curriculum cohort for level 3).
    """
    stages = KKNI_LEVELS.get(level, {}).get("stages") or []
    if not stages:
        return f"Level {level}"
    # Prefer SMK over D1 at level 3 (per project context: SMK is the focus)
    preferred_order = ["SMK", "D3", "S1", "D4", "Profesi", "S2", "S3", "SMA"]
    for p in preferred_order:
        if p in stages:
            return p
    return stages[0]


def _load_jobs_metadata(path: Path) -> Dict[str, int]:
    """Return {job_id -> min_education_kkni (int)} from jobs_metadata.csv."""
    import csv as _csv

    out: Dict[str, int] = {}
    if not path.exists():
        logger.warning("jobs_metadata.csv not found at %s — education aggregator skipped", path)
        return out
    with open(path, encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            jid = (row.get("job_id") or "").strip()
            kkni_raw = (row.get("min_education_kkni") or "").strip()
            if not jid or not kkni_raw:
                continue
            try:
                # CSV often has "5.0" rather than "5"
                kkni = int(float(kkni_raw))
                if 1 <= kkni <= 9:
                    out[jid] = kkni
            except (TypeError, ValueError):
                continue
    return out


def aggregate_for_competency(
    competency: CompetencyV2,
    job_kkni_map: Dict[str, int],
) -> None:
    """Compute and attach education_levels_demanded / education_kkni_demanded.

    Mutates the competency in place. Safe to call even when `job_kkni_map`
    is empty (sets distribution to {} and n_jobs_with_education to 0).
    """
    kknis = []
    for jid in (competency.source_job_ids or []):
        k = job_kkni_map.get(jid)
        if k is not None:
            kknis.append(k)

    competency.n_jobs_with_education = len(kknis)
    if not kknis:
        competency.education_levels_demanded = {}
        competency.education_kkni_demanded = {}
        return

    counts = Counter(kknis)
    total = sum(counts.values())
    # KKNI-level distribution
    competency.education_kkni_demanded = {
        int(lvl): round(n / total, 4) for lvl, n in counts.items()
    }
    # Friendly-stage distribution (aggregate KKNI counts by their preferred label)
    stage_counts: Counter = Counter()
    for lvl, n in counts.items():
        stage_counts[_label_for_kkni(lvl)] += n
    competency.education_levels_demanded = {
        stage: round(n / total, 4) for stage, n in stage_counts.items()
    }


def aggregate_education_demand(
    competencies: Sequence[CompetencyV2],
    jobs_metadata_path: Path,
) -> dict:
    """Apply the aggregator across a batch. Returns an audit report.

    `jobs_metadata_path` should point at a jobs_metadata.csv emitted by
    `preprocess_jobs_pipeline.py` (it has `min_education_kkni` populated by
    `education_level_extractor.py`).
    """
    job_kkni_map = _load_jobs_metadata(Path(jobs_metadata_path))
    logger.info(
        "education aggregator: loaded %d job → KKNI mappings", len(job_kkni_map)
    )

    n_with_data = 0
    n_no_data = 0
    for c in competencies:
        aggregate_for_competency(c, job_kkni_map)
        if c.n_jobs_with_education > 0:
            n_with_data += 1
        else:
            n_no_data += 1

    return {
        "version": AGGREGATOR_VERSION,
        "n_evaluated": len(competencies),
        "n_with_education_data": n_with_data,
        "n_no_education_data": n_no_data,
        "n_jobs_in_lookup": len(job_kkni_map),
        "jobs_metadata_path": str(jobs_metadata_path),
    }


# --------------------------------------------------------------------------- #
# ISCED helper (for non-Indonesian curriculum designers)
# --------------------------------------------------------------------------- #


def to_isced_distribution(kkni_distribution: Dict[int, float]) -> Dict[int, float]:
    """Convert a KKNI-level distribution to ISCED-level distribution.

    Multiple KKNI levels may map to the same ISCED level (e.g. KKNI 7 + 8 → ISCED 7);
    the result aggregates their fractions.
    """
    out: Dict[int, float] = {}
    for kkni, frac in kkni_distribution.items():
        isced = KKNI_TO_ISCED.get(int(kkni))
        if isced is None:
            continue
        out[isced] = round(out.get(isced, 0.0) + float(frac), 4)
    return out
