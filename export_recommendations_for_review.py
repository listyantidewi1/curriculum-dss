"""
export_recommendations_for_review.py

Exports top-N curriculum recommendations for expert priority labeling.
Supports multi-reviewer IRR: first --overlap_n items are marked is_overlap=True
and should be labeled by ALL reviewers independently.

Workflow:
    1. Run recommendations.py first to generate recommendations.csv
    2. Run this script to export recommendations_for_review.csv
    3. Each expert fills in 'priority_label' column (yes / no / partial)
       and 'reviewer_id' column with their name/email
    4. Collect all filled CSVs into DATA/labels/
    5. Run import_feedback.py to merge labels and compute IRR (Cohen's Kappa)
    6. Re-run recommendations.py --evaluate --baseline for P@20/NDCG@20

Outputs:
    DATA/labels/recommendations_for_review.csv  (top-N with empty priority_label)
    DATA/labels/recommendations_overlap.csv     (first overlap_n items for IRR)
"""

import argparse
from pathlib import Path

import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def main():
    parser = argparse.ArgumentParser(
        description="Export top-N curriculum recommendations for expert priority labeling."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR),
                        help="Directory containing recommendations.csv")
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR),
                        help="Output directory for review CSV (default: DATA/labels/)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top recommendations to export (default: 20)")
    parser.add_argument("--overlap_n", type=int, default=10,
                        help="First N items marked is_overlap=True for IRR (default: 10)")
    args = parser.parse_args()

    rec_path = Path(args.output_dir) / "recommendations.csv"
    if not rec_path.exists():
        print(f"[ERROR] {rec_path} not found. Run recommendations.py first.")
        return

    recs = pd.read_csv(rec_path)
    keep_cols = [c for c in [
        "rank", "skill", "priority_score", "demand_freq",
        "trend_label", "trend_q_value", "future_weight",
        "future_domain", "skill_type", "bloom_level",
    ] if c in recs.columns]
    top = recs.head(args.top_n)[keep_cols].copy()

    # Annotation columns
    top["priority_label"] = ""   # yes / no / partial
    top["reviewer_id"] = ""      # reviewer name or email
    top["notes"] = ""
    top["is_overlap"] = top["rank"] <= args.overlap_n

    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    out_path = labels_dir / "recommendations_for_review.csv"
    top.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Exported {len(top)} recommendations to {out_path}")
    print(f"[INFO] First {args.overlap_n} items marked is_overlap=True for IRR computation")

    overlap = top[top["is_overlap"]].copy()
    overlap_path = labels_dir / "recommendations_overlap.csv"
    overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Exported {len(overlap)} overlap items to {overlap_path}")

    print(f"\n  Instructions for reviewers:")
    print(f"    1. Fill 'priority_label' column: yes / no / partial")
    print(f"    2. Fill 'reviewer_id' with your name or email")
    print(f"    3. All reviewers MUST label items where is_overlap=True")
    print(f"    4. Return filled CSV to the research team")
    print(f"    5. Run: python import_feedback.py")
    print(f"    6. Run: python recommendations.py --evaluate --baseline")


if __name__ == "__main__":
    main()
