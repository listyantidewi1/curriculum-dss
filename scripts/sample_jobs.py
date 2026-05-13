"""
sample_jobs.py

Sample N rows from a raw jobs CSV with a fixed seed for reproducibility.
Preserves the input schema so downstream preprocess_jobs_pipeline.py runs
unchanged.

Usage:
    python scripts/sample_jobs.py \
        --input job_scraping/output/english_jobs.csv \
        --output DATA/preprocessing/jobs_sample_n1k_seed42.csv \
        --n 1000 \
        --seed 42

Optional --stratify-by COLUMN performs stratified sampling on that column
(equal probability within each stratum, proportional sizes overall).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-by",
        default=None,
        help="Column name for stratified sampling (e.g. query_role).",
    )
    parser.add_argument(
        "--require-description",
        action="store_true",
        default=True,
        help="Drop rows with empty description before sampling (default: on).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"[ERROR] input not found: {in_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] reading {in_path}")
    df = pd.read_csv(in_path, low_memory=False)
    print(f"[INFO] loaded {len(df):,} rows, columns: {len(df.columns)}")

    # Drop rows with empty/missing description
    if args.require_description and "description" in df.columns:
        before = len(df)
        df = df.dropna(subset=["description"])
        df = df[df["description"].astype(str).str.strip() != ""]
        after = len(df)
        print(f"[INFO] dropped {before - after:,} rows with empty description")

    if len(df) <= args.n:
        print(f"[INFO] input has only {len(df)} rows <= n={args.n}; writing as-is")
        sample = df.copy()
    elif args.stratify_by and args.stratify_by in df.columns:
        # Stratified: proportional sampling per stratum
        counts = df[args.stratify_by].value_counts(dropna=False)
        total = counts.sum()
        rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=args.seed).tolist()  # noqa: F841
        parts = []
        for stratum, c in counts.items():
            sub = df[df[args.stratify_by] == stratum]
            target = max(1, round(args.n * c / total))
            parts.append(sub.sample(n=min(target, len(sub)), random_state=args.seed))
        sample = pd.concat(parts, ignore_index=True)
        # Trim to exactly n
        if len(sample) > args.n:
            sample = sample.sample(n=args.n, random_state=args.seed)
        print(f"[INFO] stratified sample by '{args.stratify_by}': {len(sample)} rows")
    else:
        sample = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)

    # Stats on the sample
    print(f"[INFO] sample size: {len(sample):,}")
    for col in ("query_role", "query_location_label", "site", "is_english"):
        if col in sample.columns:
            top = sample[col].value_counts().head(5).to_dict()
            print(f"[INFO] {col} top-5 in sample: {top}")

    sample.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
