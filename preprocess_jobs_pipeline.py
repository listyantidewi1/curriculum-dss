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

try:
    import config
    _JOBS_SCRAPING_CSV = config.JOBS_SCRAPING_CSV
    _PREPROCESS_OUTPUT_DIR = config.PREPROCESS_OUTPUT_DIR
except ImportError:
    _JOBS_SCRAPING_CSV = Path("job_scraping/output/english_jobs.csv")
    _PREPROCESS_OUTPUT_DIR = Path("DATA/preprocessing/data_prepared")


# -------------------------------------------------------
# Sentence splitting
# -------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Split on newline+bullet or paragraph breaks (keeps segments under ~128 BERT tokens)
BULLET_OR_PARAGRAPH_RE = re.compile(r"\n\s*[\-\*\u2022\u00b7•·]\s+|\n\n+")


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    # First split on bullet boundaries and paragraph breaks
    parts = BULLET_OR_PARAGRAPH_RE.split(text)
    result = []
    for part in parts:
        part = part.replace("\r", " ").replace("\n", " ")
        sents = [s.strip() for s in SENT_SPLIT_RE.split(part) if s.strip()]
        result.extend(sents)
    return result


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
        from transformers import pipeline as hf_pipeline
        print(f"[INFO] Loading translation model: {model}", flush=True)
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
        default=str(_JOBS_SCRAPING_CSV),
        help="Raw scraped job postings CSV (default: job_scraping/output/english_jobs.csv for 12mo data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_PREPROCESS_OUTPUT_DIR),
        help="Where to store metadata + cleaned sentence CSVs",
    )
    parser.add_argument("--translate", action="store_true", help="Translate non-English sentences → English")
    parser.add_argument("--dedupe", action="store_true", help="Remove duplicate cleaned sentences")
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=600,
        help="Max sentence length in chars (default 600 for JobBERT 128-token limit)")
    parser.add_argument("--translation_model", type=str, default="Helsinki-NLP/opus-mt-mul-en")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"[INFO] Reading raw data: {input_path}", flush=True)
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

    # --------------------------------------------------
    # Item 7: Job-level deduplication (before splitting)
    # --------------------------------------------------
    # Fingerprint each job by (title, company, normalised-description hash).
    # This removes re-posted jobs that inflate demand counts and trend slopes.
    if args.dedupe:
        import hashlib, re as _re
        def _job_fingerprint(row) -> str:
            title = str(row.get("title", "")).strip().lower()
            company = str(row.get("company", "")).strip().lower()
            desc_norm = _re.sub(r"\s+", " ", str(row.get("description", "")).lower().strip())[:500]
            raw = f"{title}||{company}||{desc_norm}"
            return hashlib.md5(raw.encode("utf-8")).hexdigest()

        before_jobs = len(df)
        df["_fp"] = df.apply(_job_fingerprint, axis=1)
        df = df.drop_duplicates(subset=["_fp"]).drop(columns=["_fp"])
        after_jobs = len(df)
        print(f"[INFO] Job-level dedup: {before_jobs} → {after_jobs} jobs (removed {before_jobs - after_jobs} duplicates)")

    # -----------------------------
    # Education-level extraction (KKNI)
    # Adds min_education_kkni and education_labels per job before saving metadata.
    # See education_level_extractor.py and kkni.py.
    # -----------------------------
    try:
        from education_level_extractor import annotate_dataframe as _edu_annotate
        text_cols = [c for c in ("description", "requirements") if c in df.columns]
        if text_cols:
            _edu_annotate(df, text_columns=text_cols)
            n_known = int(df["min_education_kkni"].notna().sum())
            print(f"[INFO] Education-level extraction: {n_known}/{len(df)} jobs tagged with KKNI level")
        else:
            df["min_education_kkni"] = None
            df["education_labels"] = ""
    except ImportError:
        print("[WARN] education_level_extractor not available; skipping KKNI annotation")
        df["min_education_kkni"] = None
        df["education_labels"] = ""

    # -----------------------------
    # Save job metadata
    # -----------------------------
    metadata_cols = [
        c for c in [
            "job_id", "title", "company", "location", "site", "job_url",
            "job_date", "min_education_kkni", "education_labels",
        ] if c in df.columns
    ]
    metadata = df[metadata_cols].drop_duplicates(subset=["job_id"] if "job_id" in metadata_cols else None)
    metadata.to_csv(out_dir / "jobs_metadata.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved jobs_metadata.csv", flush=True)

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
            # sentence_id is zero-padded {job_id}_{sid:04d} so it sorts naturally
            # and stays unique across the whole corpus. Used for provenance traces
            # back from extracted skills/knowledge to source sentences.
            rows.append({
                "job_id": job_id,
                "job_date": job_date,
                "sentence_id": f"{job_id}_{sid:04d}",
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
    # Language detection & Translation (optional)
    # -----------------------------
    if args.translate:
        print("[INFO] Detecting language...")
        sents_df["lang"] = sents_df["sentence_clean"].apply(detect_language)

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

    # Pipeline expects: job_id, sentence_id, sentence_text (and optionally job_date).
    # sentence_id is the provenance handle that downstream stages thread back to
    # advanced_skills.csv / advanced_knowledge.csv per pipeline-redesign-v2 Phase 1.1.
    pipe_df = sents_df[["job_id", "job_date", "sentence_id", "sentence_en"]].rename(
        columns={"sentence_en": "sentence_text"}
    )
    pipe_path = out_dir / "job_sentences_for_pipeline.csv"
    pipe_df.to_csv(pipe_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved pipeline-ready file: {pipe_path}")

    # Also save as jobs_sentences.csv for pipeline compatibility (matches INPUT_CSV)
    jobs_sent_path = out_dir / "jobs_sentences.csv"
    pipe_df.to_csv(jobs_sent_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {jobs_sent_path} (pipeline input)")

    print("[INFO] Preprocessing completed successfully.", flush=True)


if __name__ == "__main__":
    main()
