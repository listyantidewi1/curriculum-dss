"""
merge_gold_labels.py

Merges multi-reviewer gold labels from gold_labeling_ui with gold set templates.
Applies majority vote to produce merged files for evaluate_extraction.py and
evaluate_future_mapping.py.

Inputs:
    DATA/labels/gold_skills.csv         (template)
    DATA/labels/gold_knowledge.csv      (template)
    DATA/labels/gold_future_domain.csv  (template)
    DATA/labels/gold_labels/skill_labels.csv
    DATA/labels/gold_labels/knowledge_labels.csv
    DATA/labels/gold_labels/domain_labels.csv

Outputs:
    DATA/labels/gold_skills_merged.csv
    DATA/labels/gold_knowledge_merged.csv
    DATA/labels/gold_future_domain_merged.csv

Majority vote: For is_correct, yes wins if count(yes) > count(no); else no.
If tie, use conservative (no for is_correct). For type_label, true_domain_id:
use mode (most frequent).
"""

import argparse
from pathlib import Path

import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"
GOLD_LABELS_DIR = LABELS_DIR / "gold_labels"


def _majority_is_correct(series: pd.Series) -> str:
    """Majority vote for is_correct. Tie → no (conservative)."""
    yes = (series.str.lower().str.strip() == "yes").sum()
    no = (series.str.lower().str.strip() == "no").sum()
    return "yes" if yes > no else "no"


def _mode_or_empty(series: pd.Series) -> str:
    """Mode of non-empty values, else empty string."""
    clean = series.astype(str).str.strip()
    clean = clean[clean != ""]
    if clean.empty:
        return ""
    return clean.mode().iloc[0] if not clean.mode().empty else ""


def merge_skill_labels(template: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Merge template with skill labels using majority vote."""
    if labels.empty or "gold_id" not in labels.columns:
        return template
    agg = labels.groupby("gold_id").agg(
        is_correct=("is_correct", _majority_is_correct),
        type_label=("type_label", _mode_or_empty),
        labeler_id=("labeler_id", lambda x: ",".join(sorted(x.astype(str).unique()))),
    ).reset_index()
    out = template.merge(agg[["gold_id", "is_correct", "type_label", "labeler_id"]],
                         on="gold_id", how="left", suffixes=("_tpl", ""))
    for c in ["is_correct", "type_label", "labeler_id"]:
        tpl_col = f"{c}_tpl"
        if tpl_col in out.columns:
            out[c] = out[c].combine_first(out[tpl_col]).fillna("").astype(str)
            out = out.drop(columns=[tpl_col], errors="ignore")
    return out


def merge_knowledge_labels(template: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Merge template with knowledge labels using majority vote."""
    if labels.empty or "gold_id" not in labels.columns:
        return template
    agg = labels.groupby("gold_id").agg(
        is_correct=("is_correct", _majority_is_correct),
        labeler_id=("labeler_id", lambda x: ",".join(sorted(x.astype(str).unique()))),
    ).reset_index()
    out = template.merge(agg, on="gold_id", how="left", suffixes=("_tpl", ""))
    for c in ["is_correct", "labeler_id"]:
        tpl_col = f"{c}_tpl"
        if tpl_col in out.columns:
            out[c] = out[c].combine_first(out[tpl_col]).fillna("").astype(str)
            out = out.drop(columns=[tpl_col], errors="ignore")
    return out


def merge_domain_labels(template: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Merge template with domain labels using mode."""
    if labels.empty or "gold_id" not in labels.columns:
        return template
    agg = labels.groupby("gold_id").agg(
        true_domain_id=("true_domain_id", _mode_or_empty),
        labeler_id=("labeler_id", lambda x: ",".join(sorted(x.astype(str).unique()))),
    ).reset_index()
    out = template.merge(agg, on="gold_id", how="left", suffixes=("_tpl", ""))
    for c in ["true_domain_id", "labeler_id"]:
        tpl_col = f"{c}_tpl"
        if tpl_col in out.columns:
            out[c] = out[c].combine_first(out[tpl_col]).fillna("").astype(str)
            out = out.drop(columns=[tpl_col], errors="ignore")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Merge multi-reviewer gold labels with majority vote."
    )
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR))
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    gold_labels_dir = labels_dir / "gold_labels"

    if not gold_labels_dir.exists():
        print("[WARN] gold_labels/ not found. Run gold_labeling_ui and label first.")
        return

    # Skills
    skills_tpl = labels_dir / "gold_skills.csv"
    skills_lbl = gold_labels_dir / "skill_labels.csv"
    if skills_tpl.exists() and skills_lbl.exists():
        tpl = pd.read_csv(skills_tpl)
        lbl = pd.read_csv(skills_lbl)
        merged = merge_skill_labels(tpl, lbl)
        out = labels_dir / "gold_skills_merged.csv"
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        n_valid = merged["is_correct"].isin(["yes", "no"]).sum()
        print(f"[INFO] Merged {len(merged)} skills → {out} ({n_valid} with is_correct)")
    else:
        print("[WARN] gold_skills.csv or skill_labels.csv not found")

    # Knowledge
    know_tpl = labels_dir / "gold_knowledge.csv"
    know_lbl = gold_labels_dir / "knowledge_labels.csv"
    if know_tpl.exists() and know_lbl.exists():
        tpl = pd.read_csv(know_tpl)
        lbl = pd.read_csv(know_lbl)
        merged = merge_knowledge_labels(tpl, lbl)
        out = labels_dir / "gold_knowledge_merged.csv"
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        n_valid = merged["is_correct"].isin(["yes", "no"]).sum()
        print(f"[INFO] Merged {len(merged)} knowledge → {out} ({n_valid} with is_correct)")
    else:
        print("[WARN] gold_knowledge.csv or knowledge_labels.csv not found")

    # Domain
    dom_tpl = labels_dir / "gold_future_domain.csv"
    dom_lbl = gold_labels_dir / "domain_labels.csv"
    if dom_tpl.exists() and dom_lbl.exists():
        tpl = pd.read_csv(dom_tpl)
        lbl = pd.read_csv(dom_lbl)
        merged = merge_domain_labels(tpl, lbl)
        out = labels_dir / "gold_future_domain_merged.csv"
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        n_valid = merged["true_domain_id"].astype(str).str.strip().str.len().gt(0).sum()
        print(f"[INFO] Merged {len(merged)} domain items → {out} ({n_valid} with true_domain_id)")
    else:
        print("[WARN] gold_future_domain.csv or domain_labels.csv not found")

    print("[INFO] Merge complete. Run evaluate_extraction.py and evaluate_future_mapping.py.")


if __name__ == "__main__":
    main()
