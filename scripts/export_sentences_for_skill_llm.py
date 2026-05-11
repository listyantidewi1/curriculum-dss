"""
export_sentences_for_skill_llm.py

Convert preprocessed jobs_sentences.csv into a JSONL ready for upload to
Kaggle, where baseline_versions/skill_llm/kaggle/run_inference_on_kaggle.py
runs the Skill-LLM 8B LoRA on each sentence and produces a matching JSONL
of extractions.

Workflow (see baseline_versions/skill_llm/INTEGRATION.md for the full
end-to-end picture):

    1. python preprocess_jobs_pipeline.py        # makes jobs_sentences.csv
    2. python scripts/export_sentences_for_skill_llm.py
                                                 # makes skill_llm_input.jsonl
    3. (Manual) Upload skill_llm_input.jsonl to Kaggle as a Dataset.
    4. (Manual) Open the Kaggle notebook running run_inference_on_kaggle.py,
       attach (a) the LoRA adapter as a Dataset, (b) the input JSONL as a
       Dataset. Save Version -> Save & Run All (Commit).
    5. (Manual) Download the output JSONL to results/skill_llm_extractions.jsonl.
    6. python pipeline.py --extraction-mode skill_llm_offline

Output JSONL format (one record per non-empty sentence):
    {"sentence_id": "job123_0001", "sentence_text": "Strong Python skills required"}

Usage:
    python scripts/export_sentences_for_skill_llm.py
        --input  DATA/preprocessing/data_prepared/jobs_sentences.csv
        --output skill_llm_input.jsonl
        [--max-sentences 0]   # 0 = no limit; helpful for smoke tests
        [--dedupe]            # drop duplicate sentence_text (default: keep all)

Idempotent: re-running overwrites the output file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default="DATA/preprocessing/data_prepared/jobs_sentences.csv",
        help="Path to jobs_sentences.csv from preprocess_jobs_pipeline.py",
    )
    parser.add_argument(
        "--output",
        default="skill_llm_input.jsonl",
        help="Path to write the Kaggle-ready JSONL (will be uploaded as a Kaggle Dataset)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=0,
        help="Cap output to first N sentences (0 = no cap). Use for smoke tests.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate sentence_text rows. Keeps the first sentence_id per text.",
    )
    parser.add_argument(
        "--sentence-id-col",
        default="sentence_id",
        help="Column name for sentence_id in the input CSV.",
    )
    parser.add_argument(
        "--sentence-text-col",
        default="sentence",
        help="Column name for sentence_text in the input CSV.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"[ERROR] input not found: {in_path}", file=sys.stderr)
        print(
            "        Run `python preprocess_jobs_pipeline.py` first to produce jobs_sentences.csv.",
            file=sys.stderr,
        )
        return 2

    df = pd.read_csv(in_path)
    missing = [c for c in (args.sentence_id_col, args.sentence_text_col) if c not in df.columns]
    if missing:
        print(
            f"[ERROR] input CSV at {in_path} is missing required columns: {missing}\n"
            f"        Available columns: {list(df.columns)}\n"
            f"        Override the column names with --sentence-id-col / --sentence-text-col.",
            file=sys.stderr,
        )
        return 3

    # Drop empty sentences
    df = df.copy()
    df[args.sentence_text_col] = df[args.sentence_text_col].astype(str).str.strip()
    df = df[df[args.sentence_text_col].str.len() > 0]

    n_total = len(df)

    if args.dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=[args.sentence_text_col], keep="first")
        print(f"[INFO] deduplicated {before - len(df)} duplicate sentence_text rows")

    if args.max_sentences and args.max_sentences > 0:
        df = df.head(args.max_sentences)
        print(f"[INFO] capped output to first {args.max_sentences} sentences (smoke-test mode)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            sid = str(row[args.sentence_id_col]).strip()
            stext = str(row[args.sentence_text_col]).strip()
            if not stext:
                continue
            f.write(json.dumps({"sentence_id": sid, "sentence_text": stext}, ensure_ascii=False))
            f.write("\n")

    print(
        f"[INFO] wrote {len(df)} sentences to {out_path} "
        f"(from {n_total} non-empty input sentences)"
    )
    print(
        f"[INFO] next: upload {out_path.name} to Kaggle as a Dataset, then run "
        f"baseline_versions/skill_llm/kaggle/run_inference_on_kaggle.py via Save Version."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
