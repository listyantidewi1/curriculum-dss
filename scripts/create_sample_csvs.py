"""
create_sample_csvs.py

Creates sample job postings and curriculum CSVs for dashboard simulation.
Extracts from actual pipeline data:
  - Jobs: DATA/train.json (SkillSpan format)
  - Curriculum: config.CURRICULUM_COMPONENTS

Output:
  - DATA/samples/jobs_sample.csv
  - DATA/samples/curriculum_sample.csv
"""

import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_JSON = PROJECT_ROOT / "DATA" / "train.json"
OUTPUT_DIR = PROJECT_ROOT / "DATA" / "samples"
JOBS_SAMPLE = OUTPUT_DIR / "jobs_sample.csv"
CURRICULUM_SAMPLE = OUTPUT_DIR / "curriculum_sample.csv"


def extract_jobs_from_skillspan(train_path: Path, max_jobs: int = 50) -> pd.DataFrame:
    """Extract job postings from SkillSpan train.json. Each idx = one job."""
    rows = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    jobs = {}
    date_re = re.compile(r"(\d{4}-\d{2}-\d{2})")

    for r in rows:
        idx = r["idx"]
        tokens = r.get("tokens", [])
        text = " ".join(tokens)

        if idx not in jobs:
            jobs[idx] = {"id": str(idx), "description": "", "date_posted": "2021-01-01"}

        # Extract date from "Date posted: YYYY-MM-DD"
        if "Date" in text and "posted" in text.lower():
            m = date_re.search(text)
            if m:
                jobs[idx]["date_posted"] = m.group(1)

        # Skip metadata lines (short, non-descriptive)
        skip_prefixes = ("Date posted:", "Likes:", "Job type:", "Role:", "Industry:", "Company size:", "Company type:", "Technologies", "Job description:", "Job description", "Location options:", "Job benefits:", "Company description:")
        first_words = " ".join(tokens[:4]) if len(tokens) >= 4 else text
        if any(first_words.startswith(p) for p in skip_prefixes) and len(tokens) < 8:
            continue

        # Append to description (join paragraphs with newlines)
        if jobs[idx]["description"]:
            jobs[idx]["description"] += "\n\n" + text
        else:
            jobs[idx]["description"] = text

    # Filter: keep only jobs with substantial description (>100 chars)
    out = []
    for idx, j in sorted(jobs.items())[:max_jobs]:
        desc = j["description"].strip()
        if len(desc) >= 100:
            out.append({
                "id": j["id"],
                "description": desc,
                "date_posted": j["date_posted"],
            })

    return pd.DataFrame(out)


def extract_curriculum_from_config() -> pd.DataFrame:
    """Extract curriculum CSV from config.CURRICULUM_COMPONENTS.

    Aggregates phrase examples across all Bloom-keyed sub-lists in the legacy
    config structure into a single row per component (Bloom levels were
    removed from the curriculum schema in pipeline-redesign-v2 / Req 1).
    """
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from config import CURRICULUM_COMPONENTS
    except ImportError:
        # Fallback: minimal sample
        return pd.DataFrame([
            {"component_id": "computational_thinking", "component_name": "Computational Thinking", "phrases": "pseudocode, flowcharting, algorithm execution"},
            {"component_id": "digital_literacy", "component_name": "Digital Literacy", "phrases": "search engine mechanisms, information ecosystem, digital citizenship"},
            {"component_id": "programming", "component_name": "Programming Fundamentals", "phrases": "variables, control structures, functions, debugging"},
        ])

    rows = []
    for comp_id, data in CURRICULUM_COMPONENTS.items():
        if not isinstance(data, dict):
            continue
        name = comp_id.replace("_", " ").title()
        phrases: list[str] = []
        for key, phrases_list in data.items():
            if key in ("understand", "apply", "analyze", "create", "remember", "evaluate") and isinstance(phrases_list, list):
                phrases.extend(str(p).strip() for p in phrases_list if p)
        # Dedupe while preserving order, then cap for sample readability
        seen: set[str] = set()
        deduped: list[str] = []
        for p in phrases:
            if p and p not in seen:
                seen.add(p)
                deduped.append(p)
        phrases_str = ", ".join(deduped[:15])
        if phrases_str:
            rows.append({
                "component_id": comp_id,
                "component_name": name,
                "phrases": phrases_str,
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Extracting jobs from SkillSpan train.json...")
    jobs_df = extract_jobs_from_skillspan(TRAIN_JSON, max_jobs=50)
    jobs_df.to_csv(JOBS_SAMPLE, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(jobs_df)} jobs to {JOBS_SAMPLE}")

    print("[INFO] Extracting curriculum from config.CURRICULUM_COMPONENTS...")
    curr_df = extract_curriculum_from_config()
    curr_df.to_csv(CURRICULUM_SAMPLE, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(curr_df)} curriculum rows to {CURRICULUM_SAMPLE}")

    print("[INFO] Sample CSVs created. Use for dashboard upload simulation:")
    print(f"  - Jobs: {JOBS_SAMPLE}")
    print(f"  - Curriculum: {CURRICULUM_SAMPLE}")


if __name__ == "__main__":
    main()
