"""
convert_skill_llm_jsonl_to_phase1_csv.py

Convert a Skill-LLM batch output (skill_llm_extractions.jsonl from Kaggle)
into the legacy Phase-1 CSV format that the v2 pipeline consumes:

    <output_dir>/advanced_skills.csv      (job_id, skill, type, confidence_score, source, ...)
    <output_dir>/advanced_knowledge.csv   (job_id, knowledge, confidence_score, source)

This bypasses Layer 2 (the full-posting LLM extractor) and uses only the
Skill-LLM extractions — which is the cheapest way to bring the bigger
Kaggle-batched corpus into the v2 dashboard.

Bonus: also writes a job_sentences.csv mapping that lets the retroactive
sentence-text loader actually recover the real source sentences (no need
for the english_jobs.csv fallback that we used for results.old2).

Usage:
    python scripts/convert_skill_llm_jsonl_to_phase1_csv.py
        --input results/skill_llm_extractions.jsonl
        --output DATA/preprocessing/phase1_n1k
        [--skill-llm-source-tag skill_llm_8b_lora_v1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Default confidence for Skill-LLM extractions. The model doesn't emit per-span
# confidence; we use the constant tag the existing offline loader applies
# (extractors/skill_llm_offline.py::SKILL_LLM_CONFIDENCE = 0.90).
DEFAULT_CONFIDENCE = 0.90
DEFAULT_SOURCE_TAG = "skill_llm_8b_lora_v1"


def _job_id_from_sentence_id(sid: str) -> str:
    """Convention: sentence_id = <job_id>_<NNNN>; recover job_id."""
    if not sid or "_" not in sid:
        return sid or ""
    head, _sep, tail = sid.rpartition("_")
    return head if tail.isdigit() else sid


def _confidence_tier_from_score(score: float) -> str:
    if score >= 0.9:  return "Very High"
    if score >= 0.8:  return "High"
    if score >= 0.7:  return "Medium High"
    if score >= 0.6:  return "Medium"
    if score >= 0.5:  return "Medium Low"
    if score >= 0.4:  return "Low"
    return "Very Low"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/skill_llm_extractions.jsonl")
    parser.add_argument(
        "--output",
        default="DATA/preprocessing/phase1_from_skill_llm",
        help="output directory (will be created)",
    )
    parser.add_argument(
        "--skill-llm-source-tag", default=DEFAULT_SOURCE_TAG,
        help="value to write in the 'source' column (used by downstream filters)",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="per-span confidence (Skill-LLM doesn't emit one)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[ERROR] input not found: {in_path}")
        return 1

    skills_rows: List[Dict] = []
    knowledge_rows: List[Dict] = []
    sentence_rows: List[Dict] = []
    n_records = 0
    n_with_extraction = 0
    n_parse_fail = 0

    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_records += 1

            sid = rec.get("sentence_id", "")
            stext = rec.get("sentence_text", "")
            jid = _job_id_from_sentence_id(sid)
            if rec.get("raw_output"):
                n_parse_fail += 1
            had = False

            # Sentence mapping (useful for the v2 dashboard's provenance)
            if sid and stext:
                sentence_rows.append({
                    "sentence_id": sid,
                    "job_id": jid,
                    "sentence_text": stext,
                })

            for s in (rec.get("SKILL") or []):
                text = (s.get("skill_span") or "").strip()
                if not text:
                    continue
                had = True
                skills_rows.append({
                    "job_id": jid,
                    "date_posted": "",
                    "skill": text,
                    "type": "Hard",
                    "bloom": "",
                    "confidence_score": args.confidence,
                    "confidence_tier": _confidence_tier_from_score(args.confidence),
                    "source": args.skill_llm_source_tag,
                    "semantic_density": 1.0,
                    "context_agreement": 1.0,
                    "sentence_id": sid,
                    "sentence_text": stext,
                })
            for k in (rec.get("KNOWLEDGE") or []):
                text = (k.get("skill_span") or "").strip()
                if not text:
                    continue
                had = True
                knowledge_rows.append({
                    "job_id": jid,
                    "date_posted": "",
                    "knowledge": text,
                    "confidence_score": args.confidence,
                    "confidence_tier": _confidence_tier_from_score(args.confidence),
                    "source": args.skill_llm_source_tag,
                    "sentence_id": sid,
                    "sentence_text": stext,
                })
            if had:
                n_with_extraction += 1

    # Write CSVs
    skills_cols = [
        "job_id", "date_posted", "skill", "type", "bloom",
        "confidence_score", "confidence_tier", "source",
        "semantic_density", "context_agreement",
        "sentence_id", "sentence_text",
    ]
    knowledge_cols = [
        "job_id", "date_posted", "knowledge",
        "confidence_score", "confidence_tier", "source",
        "sentence_id", "sentence_text",
    ]
    sentences_cols = ["sentence_id", "job_id", "sentence_text"]

    sp = out_dir / "advanced_skills.csv"
    kp = out_dir / "advanced_knowledge.csv"
    snp = out_dir / "job_sentences_mapping.csv"

    with open(sp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=skills_cols)
        w.writeheader()
        w.writerows(skills_rows)
    with open(kp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=knowledge_cols)
        w.writeheader()
        w.writerows(knowledge_rows)
    with open(snp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sentences_cols)
        w.writeheader()
        w.writerows(sentence_rows)

    n_unique_jobs = len({r["job_id"] for r in skills_rows} | {r["job_id"] for r in knowledge_rows})
    print(f"[INFO] read {n_records} JSONL records ({n_with_extraction} with at least one extraction, "
          f"{n_parse_fail} parse failures)")
    print(f"[INFO] wrote:")
    print(f"  {sp}  ({len(skills_rows)} skill rows)")
    print(f"  {kp}  ({len(knowledge_rows)} knowledge rows)")
    print(f"  {snp}  ({len(sentence_rows)} sentence mappings)")
    print(f"[INFO] {n_unique_jobs} unique job IDs covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
