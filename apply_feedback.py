"""
apply_feedback.py

Applies human feedback from feedback_store/ to pipeline outputs:
- Type overrides: use human_type from type_corrections.json
- Human-verified filter: keep only skills/knowledge with human_valid=valid

Usage:
  python apply_feedback.py --output_dir results_run1
  # Creates advanced_skills_human_filtered.csv, advanced_knowledge_human_filtered.csv
  # with type overrides applied and optional filtering to human-verified only
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import config

DEFAULT_FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"


def load_type_corrections(feedback_dir: Path) -> dict:
    """Load skill_text -> correct_type from type_corrections.json.
    Valid types: Hard, Soft, Both (hybrid)."""
    path = feedback_dir / "type_corrections.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_type_overrides(df: pd.DataFrame, corrections: dict) -> pd.DataFrame:
    """Override type column for skills that have human corrections."""
    if not corrections or "skill" not in df.columns or "type" not in df.columns:
        return df
    df = df.copy()
    corrections_lower = {k.strip().lower(): v for k, v in corrections.items()}
    df_skill_lower = df["skill"].astype(str).str.strip().str.lower()
    for skill_lower, correct_type in corrections_lower.items():
        mask = df_skill_lower == skill_lower
        if mask.any():
            df.loc[mask, "type"] = correct_type
    return df


def load_human_verified_skills(feedback_dir: Path) -> pd.DataFrame:
    """Load human-verified skills (human_valid=valid)."""
    path = feedback_dir / "human_verified_skills.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_human_verified_knowledge(feedback_dir: Path) -> pd.DataFrame:
    """Load human-verified knowledge."""
    path = feedback_dir / "human_verified_knowledge.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def filter_to_human_verified_skills(df: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where (job_id, skill) exists in human_verified_skills.
    Uses case-insensitive skill matching."""
    if verified.empty or "skill" not in df.columns:
        return df
    verified_pairs = set(
        zip(
            verified["job_id"].astype(str),
            verified["skill"].astype(str).str.strip().str.lower(),
        )
    )
    mask = df.apply(
        lambda r: (
            str(r.get("job_id", "")),
            str(r.get("skill", "")).strip().lower(),
        ) in verified_pairs,
        axis=1,
    )
    return df[mask].copy()


def filter_to_human_verified_knowledge(df: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    """Keep only knowledge items that are human-verified.
    Uses case-insensitive matching."""
    if verified.empty or "knowledge" not in df.columns:
        return df
    verified_know = set(verified["knowledge"].astype(str).str.strip().str.lower())
    mask = df["knowledge"].astype(str).str.strip().str.lower().isin(verified_know)
    return df[mask].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Apply human feedback to pipeline outputs (type overrides, human-verified filter)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing advanced_skills.csv, advanced_knowledge.csv",
    )
    parser.add_argument(
        "--filter_only",
        action="store_true",
        help="If set, output only human-verified items. Otherwise only apply type overrides.",
    )
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default=None,
        help="Feedback store directory (default: PROJECT_ROOT/feedback_store)",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    feedback_dir = Path(args.feedback_dir) if args.feedback_dir else DEFAULT_FEEDBACK_DIR
    feedback_dir.mkdir(parents=True, exist_ok=True)

    type_corrections = load_type_corrections(feedback_dir)
    verified_skills = load_human_verified_skills(feedback_dir)
    verified_knowledge = load_human_verified_knowledge(feedback_dir)

    # Skills
    skills_path = out_dir / "advanced_skills.csv"
    if skills_path.exists():
        df = pd.read_csv(skills_path)
        df = apply_type_overrides(df, type_corrections)
        if args.filter_only and not verified_skills.empty:
            df = filter_to_human_verified_skills(df, verified_skills)
            print(f"[INFO] Filtered to {len(df)} human-verified skills")
        out_path = out_dir / "advanced_skills_human_filtered.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved skills to {out_path}")
    else:
        print(f"[WARN] {skills_path} not found")

    # Knowledge
    know_path = out_dir / "advanced_knowledge.csv"
    if know_path.exists():
        df = pd.read_csv(know_path)
        if args.filter_only and not verified_knowledge.empty:
            df = filter_to_human_verified_knowledge(df, verified_knowledge)
            print(f"[INFO] Filtered to {len(df)} human-verified knowledge items")
        out_path = out_dir / "advanced_knowledge_human_filtered.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved knowledge to {out_path}")
    else:
        print(f"[WARN] {know_path} not found")

    print("[INFO] Done. Use advanced_*_human_filtered.csv for downstream steps when --filter_only.")


if __name__ == "__main__":
    main()
