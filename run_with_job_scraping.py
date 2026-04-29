"""
run_with_job_scraping.py

Runs the full pipeline using job_scraping/output/english_jobs.csv as the data source:

  1. preprocess_jobs_pipeline.py --input job_scraping/output/english_jobs.csv
  2. log_run_metadata.py (for reproducibility)
  3. pipeline.py --input_csv DATA/preprocessing/data_prepared/jobs_sentences.csv

Usage:
  python run_with_job_scraping.py [--sample_size N] [--output_dir PATH] [--seed N]
  python run_with_job_scraping.py --bert-knowledge         # hybrid mode
  python run_with_job_scraping.py --include-indonesian     # Item 5: merge Indonesian jobs
  python run_with_job_scraping.py --dedupe                 # Item 7: dedup at job level + sentence level
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config


def _merge_indonesian(english_csv: Path, indonesian_csv: Path, out_path: Path) -> Path:
    """Merge english_jobs.csv and indonesian_jobs.csv into a single merged CSV.
    Returns the path to the merged file."""
    import pandas as pd
    dfs = [pd.read_csv(english_csv)]
    if indonesian_csv.exists():
        id_df = pd.read_csv(indonesian_csv)
        id_df["_source_lang"] = "id"
        dfs[0]["_source_lang"] = "en"
        dfs.append(id_df)
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Merged {len(dfs[0])} English + {len(id_df)} Indonesian jobs → {out_path}")
    else:
        print(f"[WARN] Indonesian jobs CSV not found at {indonesian_csv}. Using English only.")
        out_path = english_csv
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocess + pipeline using job_scraping output"
    )
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llm_concurrency",
        type=int,
        default=None,
        help="Parallel LLM workers for pipeline (default: 3); use 1 for sequential",
    )
    parser.add_argument("--translate", action="store_true", help="Translate non-English to English")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate at job level + sentence level")
    parser.add_argument("--bert-knowledge", action="store_true", help="Fuse BERT knowledge (hybrid mode)")
    parser.add_argument(
        "--include-indonesian", action="store_true",
        help="Item 5: Merge job_scraping/output/indonesian_jobs.csv before preprocessing (auto-enables --translate --dedupe)",
    )
    parser.add_argument(
        "--extraction-mode", choices=["hybrid", "llm_only", "bert_only"], default="llm_only",
        help="Extraction mode for pipeline.py (default: llm_only — primary mode after 2026 reframe; hybrid retained as RQ1 ablation)",
    )
    args = parser.parse_args()

    # --include-indonesian implies --translate and --dedupe
    if args.include_indonesian:
        args.translate = True
        args.dedupe = True

    project_root = Path(config.PROJECT_ROOT)
    jobs_csv = config.JOBS_SCRAPING_CSV
    preprocess_out = config.PREPROCESS_OUTPUT_DIR
    pipeline_input = config.PIPELINE_INPUT_CSV

    if not jobs_csv.exists():
        print(f"[ERROR] {jobs_csv} not found. Run job_scraping/scrape_english_jobs.py first.")
        sys.exit(1)

    preprocess_out.mkdir(parents=True, exist_ok=True)

    # Item 5: Optionally merge Indonesian jobs before preprocessing
    input_csv = jobs_csv
    if args.include_indonesian:
        indonesian_csv = getattr(config, "JOBS_SCRAPING_CSV_ID",
                                 project_root / "job_scraping" / "output" / "indonesian_jobs.csv")
        merged_path = preprocess_out / "merged_jobs_input.csv"
        input_csv = _merge_indonesian(jobs_csv, Path(indonesian_csv), merged_path)

    # 1. Preprocess
    preprocess_cmd = [
        sys.executable,
        str(project_root / "preprocess_jobs_pipeline.py"),
        "--input",
        str(input_csv),
        "--output_dir",
        str(preprocess_out),
    ]
    if args.translate:
        preprocess_cmd.append("--translate")
    if args.dedupe:
        preprocess_cmd.append("--dedupe")

    print(f"[1/3] Preprocessing: {jobs_csv} -> {preprocess_out}")
    r = subprocess.run(preprocess_cmd, cwd=str(project_root))
    if r.returncode != 0:
        sys.exit(r.returncode)

    # 2. Log run metadata (for reproducibility; needs jobs_sentences.csv to exist)
    log_cmd = [
        sys.executable,
        str(project_root / "log_run_metadata.py"),
        "--output_dir", args.output_dir,
        "--seed", str(args.seed),
    ]
    print(f"[2/3] Logging run metadata...")
    r = subprocess.run(log_cmd, cwd=str(project_root))
    if r.returncode != 0:
        sys.exit(r.returncode)

    # 3. Pipeline
    pipeline_cmd = [
        sys.executable,
        str(project_root / "pipeline.py"),
        "--input_csv",
        str(pipeline_input),
        "--output_dir",
        args.output_dir,
        "--sample_size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
    ]
    if args.bert_knowledge:
        pipeline_cmd.append("--bert-knowledge")
    if args.llm_concurrency is not None:
        pipeline_cmd.extend(["--llm_concurrency", str(args.llm_concurrency)])
    pipeline_cmd.extend(["--extraction-mode", args.extraction_mode])

    print(f"[3/3] Pipeline: {pipeline_input} -> {args.output_dir}")
    r = subprocess.run(pipeline_cmd, cwd=str(project_root))
    sys.exit(r.returncode if r.returncode else 0)


if __name__ == "__main__":
    main()
