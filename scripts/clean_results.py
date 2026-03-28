"""
clean_results.py - Clean results folder from jobs processed during OpenRouter credit outage.

When Phase 1 runs out of OpenRouter credit, some jobs may get saved with empty or
invalid extractions. This script helps remove those suspect results.

Usage:
  # List all jobs with mod time and skill/knowledge counts (to identify suspects)
  python scripts/clean_results.py --output_dir results --list

  # Remove jobs modified after a cutoff time (e.g. when you ran out of credit)
  python scripts/clean_results.py --output_dir results --since "2025-02-24 15:00"

  # Remove jobs with 0 skills AND 0 knowledge (likely API failures)
  python scripts/clean_results.py --output_dir results --empty-only

  # Remove jobs where LLM contributed nothing (gpt_skill_count=0); catches BERT-only
  # results when you ran out of OpenRouter credit but BERT still ran
  python scripts/clean_results.py --output_dir results --no-llm

  # Remove specific job IDs
  python scripts/clean_results.py --output_dir results --job-ids "in-abc123,in-def456"

  # Dry run: show what would be removed without deleting
  python scripts/clean_results.py --output_dir results --since "2025-02-24 15:00" --dry-run

After cleanup, re-run the pipeline to re-process the removed jobs:
  python run_with_job_scraping.py
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def _parse_dt(s: str) -> datetime:
    """Parse datetime string; support YYYY-MM-DD and YYYY-MM-DD HH:MM."""
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {s}")


def get_job_ids_from_results(output_dir: Path) -> list[tuple[str, float, int, int, int | None]]:
    """
    Scan job_*_analysis.json files.
    Returns list of (job_id, mtime_ts, n_skills, n_knowledge, gpt_skill_count).
    gpt_skill_count is from extraction_metrics; None if not present.
    """
    out = []
    for f in output_dir.glob("job_*_analysis.json"):
        name = f.stem  # e.g. job_go-xxx_analysis or job_in-xxx_analysis
        if not name.startswith("job_") or not name.endswith("_analysis"):
            continue
        job_id = name[4:-9]  # strip "job_" and "_analysis"
        mtime = f.stat().st_mtime
        n_skills = n_knowledge = 0
        gpt_skill_count = None
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
            n_skills = len(data.get("skills") or [])
            n_knowledge = len(data.get("knowledge") or [])
            em = data.get("extraction_metrics") or {}
            if "gpt_skill_count" in em:
                gpt_skill_count = int(em["gpt_skill_count"])
        except Exception:
            pass
        out.append((job_id, mtime, n_skills, n_knowledge, gpt_skill_count))
    return out


def select_jobs_to_remove(
    job_list: list[tuple[str, float, int, int, int | None]],
    since: datetime | None = None,
    empty_only: bool = False,
    no_llm: bool = False,
    job_ids: list[str] | None = None,
) -> set[str]:
    """Select job_ids to remove based on criteria."""
    to_remove = set()
    since_ts = since.timestamp() if since else None

    for jid, mtime, n_skills, n_knowledge, gpt_skill_count in job_list:
        if job_ids is not None and jid in job_ids:
            to_remove.add(jid)
            continue
        if since_ts is not None and mtime >= since_ts:
            to_remove.add(jid)
            continue
        if empty_only and n_skills == 0 and n_knowledge == 0:
            to_remove.add(jid)
            continue
        if no_llm and gpt_skill_count is not None and gpt_skill_count == 0:
            to_remove.add(jid)
    return to_remove


def remove_job_rows_from_csv(path: Path, job_ids: set[str], job_col: str = "job_id") -> int:
    """Remove rows where job_col is in job_ids. Returns rows removed."""
    if not path.exists():
        return 0
    df = pd.read_csv(path)
    if job_col not in df.columns:
        return 0
    n_before = len(df)
    df = df[~df[job_col].isin(job_ids)]
    n_removed = n_before - len(df)
    df.to_csv(path, index=False)
    return n_removed


def main():
    parser = argparse.ArgumentParser(
        description="Clean results folder from jobs processed during OpenRouter credit outage"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all jobs with mod time and counts (default if no other action)",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        metavar="DATETIME",
        help='Remove jobs modified after this time (e.g. "2025-02-24 15:00")',
    )
    parser.add_argument(
        "--empty-only",
        action="store_true",
        help="Remove only jobs with 0 skills and 0 knowledge",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Remove jobs where LLM contributed nothing (gpt_skill_count=0); includes BERT-only results",
    )
    parser.add_argument(
        "--job-ids",
        type=str,
        default=None,
        metavar="ID1,ID2,...",
        help="Comma-separated list of job IDs to remove",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without deleting",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[ERROR] Output dir not found: {output_dir}")
        return 1

    job_list = get_job_ids_from_results(output_dir)
    print(f"[INFO] Found {len(job_list)} job result files in {output_dir}")

    since_dt = None
    if args.since:
        try:
            since_dt = _parse_dt(args.since)
        except ValueError as e:
            print(f"[ERROR] {e}")
            return 1

    job_id_list = None
    if args.job_ids:
        job_id_list = [x.strip() for x in args.job_ids.split(",") if x.strip()]

    to_remove = select_jobs_to_remove(
        job_list,
        since=since_dt,
        empty_only=args.empty_only,
        no_llm=args.no_llm,
        job_ids=job_id_list,
    )

    if not to_remove:
        print("[INFO] No jobs match the removal criteria.")
        if args.list:
            for jid, mtime, ns, nk, gpt in sorted(job_list, key=lambda x: x[1]):
                dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                gpt_str = f"  gpt={gpt}" if gpt is not None else ""
                print(f"  {jid}  mtime={dt}  skills={ns}  knowledge={nk}{gpt_str}")
        return 0

    print(f"[INFO] Would remove {len(to_remove)} jobs")
    if args.dry_run:
        for jid in sorted(to_remove):
            info = next((x for x in job_list if x[0] == jid), None)
            if info:
                _, mtime, ns, nk, gpt = info
                dt = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                gpt_str = f"  gpt={gpt}" if gpt is not None else ""
                print(f"  {jid}  mtime={dt}  skills={ns}  knowledge={nk}{gpt_str}")
        print("\n[DRY RUN] No files deleted. Re-run without --dry-run to apply.")
        return 0

    # Delete JSON files
    deleted = 0
    for jid in to_remove:
        p = output_dir / f"job_{jid}_analysis.json"
        if p.exists():
            p.unlink()
            deleted += 1
    print(f"[INFO] Deleted {deleted} job_*_analysis.json files")

    # Remove rows from CSVs
    csv_files = [
        ("advanced_skills.csv", "job_id"),
        ("advanced_knowledge.csv", "job_id"),
        ("coverage_report.csv", "job_id"),
        ("comprehensive_analysis.csv", "job_id"),
        ("model_comparison.csv", "job_id"),
    ]
    for fname, col in csv_files:
        p = output_dir / fname
        n = remove_job_rows_from_csv(p, to_remove, col)
        if n > 0:
            print(f"[INFO] Removed {n} rows from {fname}")

    print("[DONE] Cleanup complete. Re-run the pipeline to re-process removed jobs.")
    return 0


if __name__ == "__main__":
    exit(main())
