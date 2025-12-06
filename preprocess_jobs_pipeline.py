"""
preprocess_jobs_pipeline.py

Unified preprocessing pipeline:

- Reads raw job postings from DATA/preprocessing/jobs_rpl_id_raw_2.csv
- Splits descriptions into sentences (with job_id + job_date)
- Cleans markdown/noise
- Detects language + (optional) translates to English
- Removes duplicate sentences (optional)
- Saves:
    DATA/preprocessing/data_prepared/jobs_metadata.csv
    DATA/preprocessing/data_prepared/job_sentences_full.csv
    DATA/preprocessing/data_prepared/job_sentences_for_pipeline.csv
"""

import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd
from langdetect import detect, LangDetectException
from transformers import pipeline as hf_pipeline


# -------------------------------------------------------
# Sentence splitting
# -------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.replace("\r", " ").replace("\n", " ")
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return sents


# -------------------------------------------------------
# Cleaning helpers
# -------------------------------------------------------

BULLET_RE = re.compile(r"^[\s\-\*\u2022\u00b7•·]+")  # -, *, •, etc.


def clean_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text

    s = s.replace("\r", " ").replace("\n", " ").replace("\xa0", " ")
    s = s.replace("*", "").replace("`", "")

    # collapse repeated slashes
    s = re.sub(r"[\\/]{2,}", " ", s)

    # remove weird /- combinations
    s = re.sub(r"\s*[/\\]\s*-\s*[/\\]?\s*", " ", s)

    # remove bullet prefixes
    s = BULLET_RE.sub("", s)

    # collapse multi punctuation
    s = re.sub(r"([!?.,;:])\1+", r"\1", s)

    # clean empty parentheses
    s = re.sub(r"\(\s*\)", " ", s)

    s = re.sub(r"\s{2,}", " ", s)

    return s.strip(" \"'")


# -------------------------------------------------------
# Language + Translation
# -------------------------------------------------------

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"


class Translator:
    def __init__(self, model="Helsinki-NLP/opus-mt-mul-en"):
        print(f"[INFO] Loading translation model: {model}")
        self.pipe = hf_pipeline("translation", model=model)

    def translate_batch(self, texts: List[str]):
        out = self.pipe(texts, truncation=True, max_length=512)
        return [d["translation_text"] for d in out]


# -------------------------------------------------------
# Main pipeline
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full preprocessing for raw jobs → pipeline-ready sentences")
    parser.add_argument(
        "--input",
        type=str,
        default=r"DATA/preprocessing/jobs_rpl_id_raw_2.csv",
        help="Raw scraped job postings CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"DATA/preprocessing/data_prepared",
        help="Where to store metadata + cleaned sentence CSVs"
    )
    parser.add_argument("--translate", action="store_true", help="Translate non-English sentences → English")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate cleaned sentences")
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--translation_model", type=str, default="Helsinki-NLP/opus-mt-mul-en")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[INFO] Reading raw data: {input_path}")
    df = pd.read_csv(input_path)

    if "description" not in df.columns:
        raise ValueError("Raw CSV must contain a 'description' column.")

    # -----------------------------
    # Normalize job_id + job_date
    # -----------------------------
    if "id" in df.columns:
        df["job_id"] = df["id"].astype(str)
    else:
        df = df.reset_index()
        df["job_id"] = df["index"].astype(str)

    df["job_date"] = pd.to_datetime(df.get("date_posted", pd.NaT))

    # -----------------------------
    # Save job metadata
    # -----------------------------
    metadata_cols = [c for c in ["job_id", "title", "company", "location", "site", "job_url", "job_date"] if c in df.columns]
    metadata = df[metadata_cols].drop_duplicates()
    metadata.to_csv(out_dir / "jobs_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved jobs_metadata.csv")

    # -----------------------------
    # Sentence splitting
    # -----------------------------
    print("[INFO] Splitting into sentences...")
    rows = []

    for _, row in df.iterrows():
        job_id = row["job_id"]
        job_date = row.get("job_date", pd.NaT)
        text = row["description"]

        sents = split_sentences(text)
        sid = 0
        for sent in sents:
            if len(sent) < args.min_len or len(sent) > args.max_len:
                continue
            sid += 1
            rows.append({
                "job_id": job_id,
                "job_date": job_date,
                "sentence_raw": sent
            })

    sents_df = pd.DataFrame(rows)
    print(f"[INFO] Total split sentences: {len(sents_df)}")

    # -----------------------------
    # Cleaning
    # -----------------------------
    print("[INFO] Cleaning sentences...")
    sents_df["sentence_clean"] = sents_df["sentence_raw"].apply(clean_sentence)

    # -----------------------------
    # Language detection
    # -----------------------------
    print("[INFO] Detecting language...")
    sents_df["lang"] = sents_df["sentence_clean"].apply(detect_language)

    # -----------------------------
    # Translation (optional)
    # -----------------------------
    if args.translate:
        translator = Translator(args.translation_model)
        sents_df["sentence_en"] = sents_df["sentence_clean"]

        mask = (sents_df["lang"] != "en") & (sents_df["lang"] != "unknown")
        idxs = sents_df.index[mask]

        print(f"[INFO] Translating {len(idxs)} non-English sentences...")
        batch = 64
        for i in range(0, len(idxs), batch):
            b = idxs[i:i+batch]
            text_batch = sents_df.loc[b, "sentence_clean"].tolist()
            translated = translator.translate_batch(text_batch)
            sents_df.loc[b, "sentence_en"] = translated
    else:
        print("[INFO] Translation disabled. Using cleaned text only.")
        sents_df["sentence_en"] = sents_df["sentence_clean"]

    # -----------------------------
    # Dedupe (optional)
    # -----------------------------
    if args.dedupe:
        before = len(sents_df)
        sents_df = sents_df.drop_duplicates(subset=["sentence_en"])
        after = len(sents_df)
        print(f"[INFO] Removed duplicates: {before - after}")

    # -----------------------------
    # Save full + pipeline-ready
    # -----------------------------
    full_path = out_dir / "job_sentences_full.csv"
    sents_df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {full_path}")

    pipe_df = sents_df[["job_id", "job_date", "sentence_en"]].rename(columns={"sentence_en": "sentence"})
    pipe_path = out_dir / "job_sentences_for_pipeline.csv"
    pipe_df.to_csv(pipe_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved pipeline-ready file: {pipe_path}")

    print("[INFO] Preprocessing completed successfully.")
