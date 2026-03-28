"""
run_with_job_scraping.py

Runs the full pipeline using job_scraping/output/english_jobs.csv as the data source:

  1. preprocess_jobs_pipeline.py --input job_scraping/output/english_jobs.csv
  2. log_run_metadata.py (for reproducibility)
  3. pipeline.py --input_csv DATA/preprocessing/data_prepared/jobs_sentences.csv

Usage:
  python run_with_job_scraping.py [--sample_size N] [--output_dir PATH] [--seed N]
  python run_with_job_scraping.py --bert-knowledge  # hybrid mode (fuse BERT knowledge)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import config


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocess + pipeline using job_scraping/output/english_jobs.csv"
    )
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llm_concurrency",
        type=int,
        default=None,
        help="Parallel LLM workers for pipeline (default: 3); use 1 for sequential",
    )
    parser.add_argument("--translate", action="store_true", help="Translate non-English to English")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate sentences")
    parser.add_argument("--bert-knowledge", action="store_true", help="Fuse BERT knowledge (hybrid mode)")
    args = parser.parse_args()

    project_root = Path(config.PROJECT_ROOT)
    jobs_csv = config.JOBS_SCRAPING_CSV
    preprocess_out = config.PREPROCESS_OUTPUT_DIR
    pipeline_input = config.PIPELINE_INPUT_CSV

    if not jobs_csv.exists():
        print(f"[ERROR] {jobs_csv} not found. Run job_scraping/scrape_english_jobs.py first.")
        sys.exit(1)

    preprocess_out.mkdir(parents=True, exist_ok=True)

    # 1. Preprocess
    preprocess_cmd = [
        sys.executable,
        str(project_root / "preprocess_jobs_pipeline.py"),
        "--input",
        str(jobs_csv),
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

    print(f"[3/3] Pipeline: {pipeline_input} -> {args.output_dir}")
    r = subprocess.run(pipeline_cmd, cwd=str(project_root))
    sys.exit(r.returncode if r.returncode else 0)


if __name__ == "__main__":
    main()
