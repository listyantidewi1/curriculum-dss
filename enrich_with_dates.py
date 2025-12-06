"""
enrich_with_dates.py

Attach job_date (and year/month) from jobs_metadata.csv to
existing pipeline outputs:

- advanced_skills.csv
- advanced_knowledge.csv
- coverage_report.csv
- comprehensive_analysis.csv

Outputs:
- advanced_skills_with_dates.csv
- advanced_knowledge_with_dates.csv
- coverage_report_with_dates.csv
- comprehensive_analysis_with_dates.csv
"""

import argparse
from pathlib import Path

import pandas as pd

import config  # uses config.OUTPUT_DIR


def add_dates(df: pd.DataFrame, jobs_meta: pd.DataFrame, kind: str) -> pd.DataFrame:
    if "job_id" not in df.columns:
        raise ValueError(f"{kind} must have a 'job_id' column.")

    merged = df.merge(
        jobs_meta[["job_id", "job_date"]],
        on="job_id",
        how="left",
        validate="m:1",
    )

    # Parse job_date to datetime if not already
    merged["job_date"] = pd.to_datetime(merged["job_date"], errors="coerce")

    # Convenience columns
    merged["job_year"] = merged["job_date"].dt.year
    merged["job_month"] = merged["job_date"].dt.to_period("M").astype(str)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Attach job dates to existing pipeline outputs."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory where pipeline outputs are stored.",
    )
    parser.add_argument(
        "--jobs_meta",
        type=str,
        default="jobs_metadata.csv",
        help="Path to jobs_metadata.csv (job_id + job_date).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    jobs_meta_path = Path(args.jobs_meta)

    if not jobs_meta_path.exists():
        raise FileNotFoundError(f"jobs_metadata.csv not found at {jobs_meta_path}")

    print(f"[INFO] Reading jobs metadata from {jobs_meta_path}")
    jobs_meta = pd.read_csv(jobs_meta_path)
    if "job_id" not in jobs_meta.columns:
        raise ValueError("jobs_metadata.csv must contain 'job_id' column.")

    # Ensure job_id types match typical pipeline outputs
    jobs_meta["job_id"] = jobs_meta["job_id"].astype(str)

    # Files to enrich if present
    targets = [
        ("advanced_skills.csv", "advanced_skills_with_dates.csv"),
        ("advanced_knowledge.csv", "advanced_knowledge_with_dates.csv"),
        ("coverage_report.csv", "coverage_report_with_dates.csv"),
        ("comprehensive_analysis.csv", "comprehensive_analysis_with_dates.csv"),
    ]

    for in_name, out_name in targets:
        in_path = out_dir / in_name
        if not in_path.exists():
            print(f"[WARN] {in_path} not found, skipping.")
            continue

        print(f"[INFO] Enriching {in_path} with job_date...")
        df = pd.read_csv(in_path)
        df["job_id"] = df["job_id"].astype(str)
        enriched = add_dates(df, jobs_meta, kind=in_name)

        out_path = out_dir / out_name
        enriched.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()
