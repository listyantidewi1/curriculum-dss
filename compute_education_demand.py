"""
compute_education_demand.py
===========================

Aggregate per-skill education-level demand using the ``min_education_kkni``
column from ``jobs_metadata.csv`` (produced by preprocess_jobs_pipeline.py)
joined with the per-skill job_id list from ``advanced_skills_with_dates.csv``.

Output
------
``results/skill_education_demand.csv`` with one row per (skill, kkni_level):

    skill, kkni_level, n_jobs, share

plus a per-skill summary in ``results/skill_education_summary.csv``:

    skill, n_jobs_total, n_jobs_with_education, dominant_kkni, education_distribution_json

Usage
-----
    python compute_education_demand.py
    python compute_education_demand.py --output_dir results/hybrid

Resumable: if both output CSVs already exist and the input is older, the script
no-ops with a [SKIP] message. Pass --force to override.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

import config


def _load_jobs_metadata(meta_path: Path) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run preprocess_jobs_pipeline.py first so that "
            f"min_education_kkni is available in jobs_metadata.csv."
        )
    df = pd.read_csv(meta_path)
    if "job_id" not in df.columns:
        raise ValueError(f"{meta_path} must have a job_id column")
    if "min_education_kkni" not in df.columns:
        raise ValueError(
            f"{meta_path} is missing min_education_kkni. Re-run preprocess_jobs_pipeline.py "
            f"after the 2026 KKNI reframe."
        )
    return df[["job_id", "min_education_kkni"]].copy()


def _load_skills_with_dates(skills_path: Path) -> pd.DataFrame:
    if not skills_path.exists():
        raise FileNotFoundError(
            f"Missing {skills_path}. Run pipeline.py + enrich_with_dates.py first."
        )
    df = pd.read_csv(skills_path)
    needed = {"skill", "job_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{skills_path} is missing required columns: {missing}")
    return df[["skill", "job_id"]].copy()


def compute(meta_path: Path, skills_path: Path, out_dir: Path) -> Dict[str, Path]:
    meta = _load_jobs_metadata(meta_path)
    skills = _load_skills_with_dates(skills_path)

    # Coerce KKNI level to int where possible; keep NaN otherwise (= unknown)
    meta["min_education_kkni"] = pd.to_numeric(meta["min_education_kkni"], errors="coerce")

    # Join skill rows with their job's KKNI level. Inner join: skills whose
    # job has no KKNI signal contribute to "unknown" only via the summary file.
    joined = skills.merge(meta, on="job_id", how="left")

    # ---- per-skill x KKNI table ----
    rows = []
    for skill, grp in joined.groupby("skill", dropna=False):
        if not str(skill).strip():
            continue
        n_total = len(grp)
        n_known = int(grp["min_education_kkni"].notna().sum())
        # Count by KKNI level (skip NaN here)
        counts: Dict[int, int] = defaultdict(int)
        for v in grp["min_education_kkni"].dropna().tolist():
            try:
                counts[int(v)] += 1
            except (TypeError, ValueError):
                continue
        for lvl, n in sorted(counts.items()):
            share = n / n_known if n_known else 0.0
            rows.append({
                "skill": skill,
                "kkni_level": lvl,
                "n_jobs": n,
                "share": round(share, 4),
            })
        # Always emit at least one row so every skill is in the long-format
        # table for downstream joins.
        if not counts:
            rows.append({
                "skill": skill,
                "kkni_level": None,
                "n_jobs": 0,
                "share": 0.0,
            })

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "skill_education_demand.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    # ---- per-skill summary ----
    summary_rows = []
    for skill, grp in joined.groupby("skill", dropna=False):
        if not str(skill).strip():
            continue
        n_total = len(grp)
        n_known = int(grp["min_education_kkni"].notna().sum())
        dist = (
            grp["min_education_kkni"].dropna().astype(int).value_counts().sort_index().to_dict()
        )
        dominant = max(dist, key=dist.get) if dist else None
        summary_rows.append({
            "skill": skill,
            "n_jobs_total": n_total,
            "n_jobs_with_education": n_known,
            "dominant_kkni": dominant,
            "education_distribution_json": json.dumps({int(k): int(v) for k, v in dist.items()}),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "skill_education_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Wrote {long_path} ({len(long_df)} rows)")
    print(f"[INFO] Wrote {summary_path} ({len(summary_df)} skills)")
    if summary_df.empty:
        return {"long": long_path, "summary": summary_path}

    coverage = (
        summary_df["n_jobs_with_education"].sum() / max(1, summary_df["n_jobs_total"].sum())
    )
    print(f"[INFO] KKNI coverage: {coverage:.1%} of skill mentions have an education signal.")
    return {"long": long_path, "summary": summary_path}


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-skill education-level demand (KKNI).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing advanced_skills_with_dates.csv (default: config.OUTPUT_DIR).",
    )
    parser.add_argument(
        "--jobs_meta",
        type=str,
        default=None,
        help=(
            "Path to jobs_metadata.csv (default: "
            "config.PREPROCESS_OUTPUT_DIR/jobs_metadata.csv)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output files already exist.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.jobs_meta) if args.jobs_meta else (Path(config.PREPROCESS_OUTPUT_DIR) / "jobs_metadata.csv")
    skills_path = out_dir / "advanced_skills_with_dates.csv"

    long_path = out_dir / "skill_education_demand.csv"
    summary_path = out_dir / "skill_education_summary.csv"
    if not args.force and long_path.exists() and summary_path.exists():
        if long_path.stat().st_mtime > skills_path.stat().st_mtime:
            print(f"[SKIP] {long_path.name} and {summary_path.name} already exist and are newer than the input.")
            return

    compute(meta_path, skills_path, out_dir)


if __name__ == "__main__":
    main()
