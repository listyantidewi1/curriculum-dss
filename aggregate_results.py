"""
aggregate_results.py

Aggregate multiple experimental runs of the advanced pipeline into
a single set of CSVs for final plots, competencies, and review.

Assumes you have folders like:
    D:\Projects\skill-extraction\results_run1
    D:\Projects\skill-extraction\results_run2
    ...

Each folder contains:
    - advanced_skills.csv
    - advanced_knowledge.csv
    - coverage_report.csv
    - comprehensive_analysis.csv
    - model_comparison.csv
    - (optionally) future_skill_weights_dummy.csv
    - (optionally) verified_skills.csv

Output:
    - results_aggregated/advanced_skills.csv
    - results_aggregated/advanced_knowledge.csv
    - results_aggregated/coverage_report.csv
    - results_aggregated/comprehensive_analysis.csv
    - results_aggregated/model_comparison.csv
    - results_aggregated/future_skill_weights_dummy.csv  (if available)
    - results_aggregated/verified_skills.csv              (if available)
"""

import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd


FILES_TO_AGGREGATE = [
    "advanced_skills.csv",
    "advanced_knowledge.csv",
    "coverage_report.csv",
    "comprehensive_analysis.csv",
    "model_comparison.csv",
    "future_skill_weights_dummy.csv",  # from future_weight_mapping.py
    "verified_skills.csv",             # from verify_skills.py
]


def aggregate_single_file(
    file_name: str,
    run_dirs: List[Path],
    output_dir: Path,
) -> None:
    """Aggregate one file (e.g., advanced_skills.csv) across runs."""
    frames = []

    for run_dir in run_dirs:
        src = run_dir / file_name
        if not src.exists():
            print(f"[WARN] {src} not found, skipping for this run.")
            continue

        try:
            df = pd.read_csv(src)
        except Exception as e:
            print(f"[WARN] Failed to read {src}: {e}")
            continue

        # Add run_id column
        df["run_id"] = run_dir.name
        frames.append(df)

    if not frames:
        print(f"[WARN] No data found for {file_name} across runs. Skipping.")
        return

    combined = pd.concat(frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / file_name
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Aggregated {file_name} -> {out_path} ({len(combined)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multiple experiment runs into a single results directory."
    )
    parser.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of run directories to aggregate, "
            "e.g. --run_dirs results_run1 results_run2 results_run3"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_aggregated",
        help="Directory to write aggregated CSVs (default: results_aggregated)",
    )

    args = parser.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")

    output_dir = Path(args.output_dir)

    print("[INFO] Aggregating runs:")
    for d in run_dirs:
        print(f"       - {d}")

    for fname in FILES_TO_AGGREGATE:
        aggregate_single_file(fname, run_dirs, output_dir)

    print("[INFO] Aggregation complete.")


if __name__ == "__main__":
    main()
