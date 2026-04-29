"""
dashboard/publish.py
====================

Helpers for the "publish a canonical run" flow (Phase 2 of the 2026 reframe).

The admin runs the pipeline (run.bat) which writes ``results/`` at the project
root. When the admin is happy with that run, they call ``publish_results_dir``
which:

1. Captures aggregate stats (competency count, skill count, KKNI distribution).
2. Records a row in ``published_runs`` with ``is_active = 1`` (and clears the
   previous active row) and a snapshot of the directory path.
3. The public surface (``api_public.py``) always reads from
   ``get_active_published_run()``.

Note: we do not copy the directory; we record its path. The admin is expected
not to delete the directory. A future improvement could snapshot to
``data/published/{version}`` for full immutability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

import config
from . import db as ddb


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #

def _count_competencies(results_dir: Path) -> int:
    proposals = results_dir / "competency_proposals.json"
    if not proposals.exists():
        return 0
    try:
        data = json.loads(proposals.read_text(encoding="utf-8"))
        return len(data.get("competencies", []))
    except Exception:
        return 0


def _count_skills(results_dir: Path) -> int:
    for name in ("verified_skills.csv", "advanced_skills.csv"):
        p = results_dir / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if "skill" in df.columns:
                return int(df["skill"].nunique())
        except Exception:
            return 0
    return 0


def collect_publish_stats(results_dir: Path) -> Dict[str, Any]:
    return {
        "n_competencies": _count_competencies(results_dir),
        "n_skills": _count_skills(results_dir),
    }


# --------------------------------------------------------------------------- #
# Publish + retrieve
# --------------------------------------------------------------------------- #

def publish_results_dir(
    results_dir: Path | str,
    version_label: str,
    user_id: Optional[int] = None,
    spektrum_code: Optional[str] = None,
    vocational_field: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """Publish a results directory as the active canonical run.

    Returns the published_runs.id of the newly active row.

    Side effects:
        * Sets is_active = 0 on every prior row (only one active at a time).
        * Inserts the new row with is_active = 1.
    """
    results_path = Path(results_dir).resolve()
    if not results_path.exists() or not results_path.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_path}")

    competencies_path = results_path / "competency_proposals.json"
    if not competencies_path.exists():
        raise FileNotFoundError(
            f"Cannot publish: {competencies_path} is missing. "
            "Run the pipeline through generate_competencies.py first."
        )

    stats = collect_publish_stats(results_path)

    conn = ddb.get_conn()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE published_runs SET is_active = 0 WHERE is_active = 1")
        cur.execute(
            """
            INSERT INTO published_runs(
                version_label, results_dir, spektrum_code, vocational_field, notes,
                n_competencies, n_skills, published_by, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                version_label,
                str(results_path),
                spektrum_code,
                vocational_field,
                notes,
                stats["n_competencies"],
                stats["n_skills"],
                user_id,
            ),
        )
        conn.commit()
        new_id = int(cur.lastrowid or 0)
    finally:
        conn.close()

    return new_id


def get_active_published_run() -> Optional[Dict[str, Any]]:
    row = ddb.q_one(
        """
        SELECT id, version_label, results_dir, spektrum_code, vocational_field,
               notes, n_competencies, n_skills, published_at, is_active
        FROM published_runs
        WHERE is_active = 1
        ORDER BY id DESC
        LIMIT 1
        """
    )
    return dict(row) if row else None


def list_published_runs(limit: int = 50) -> list[Dict[str, Any]]:
    rows = ddb.q_all(
        """
        SELECT id, version_label, results_dir, spektrum_code, vocational_field,
               notes, n_competencies, n_skills, published_at, is_active
        FROM published_runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    return [dict(r) for r in rows]


def active_results_dir() -> Path:
    """Return the active published run's results directory.

    Falls back to ``config.OUTPUT_DIR`` if no run has been published yet, so
    the public surface can boot before the admin clicks publish.
    """
    run = get_active_published_run()
    if run and run.get("results_dir"):
        p = Path(run["results_dir"])
        if p.exists():
            return p
    return Path(config.OUTPUT_DIR)
