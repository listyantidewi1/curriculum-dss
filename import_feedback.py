"""
import_feedback.py

Reads feedback from feedback_store/ (multi-reviewer format) or expert_review_*.csv (legacy).
Merges multi-reviewer feedback using majority vote. Validates human_valid, human_bloom, etc.

Scientific methods (see SCIENTIFIC_METHODOLOGY.md):
    - Majority vote for multi-reviewer merge (most frequent non-empty value)
    - Cohen's Kappa (2 raters) or Fleiss' Kappa (3+ raters) for IRR

Outputs to feedback_store/:
    - human_verified_skills.csv
    - human_verified_knowledge.csv
    - competency_assessments.json
    - bloom_corrections.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config

VALID_HUMAN_VALID = {"valid", "invalid", ""}
VALID_HUMAN_BLOOM = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "N/A", ""}
VALID_HUMAN_QUALITY = {"1", "2", "3", "4", "5", "poor", "fair", "good", "excellent", ""}
VALID_HUMAN_RELEVANT = {"yes", "no", "partial", ""}


def _normalize(val: str) -> str:
    """Strip and lowercase for comparison."""
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def _majority(votes: list) -> str:
    """Return majority vote, or first non-empty if tie."""
    votes = [str(v).strip() for v in votes if v and str(v).strip()]
    if not votes:
        return ""
    c = Counter(votes)
    best = c.most_common(1)[0]
    return best[0]


def _load_skill_feedback(output_dir: Path, feedback_dir: Path) -> pd.DataFrame:
    """Load skill feedback: prefer feedback_store/skill_feedback.csv, else legacy expert_review_skills.csv.
    Migrates legacy data to feedback_store on first import."""
    fb_path = feedback_dir / "skill_feedback.csv"
    template_path = output_dir / "expert_review_skills.csv"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        if not fb.empty and "review_id" in fb.columns and "reviewer_id" in fb.columns:
            if template_path.exists():
                tpl = pd.read_csv(template_path)
                if "skill" in tpl.columns and "job_id" in tpl.columns:
                    merged = fb.merge(tpl[["review_id", "skill", "job_id"]], on="review_id", how="left")
                    return merged
            return fb
    # Legacy: read from expert_review_skills.csv and migrate to feedback_store
    if template_path.exists():
        df = pd.read_csv(template_path)
        if "human_valid" in df.columns:
            hv = df["human_valid"].astype(str).str.strip()
            if hv.isin(["valid", "invalid"]).any() or (df["human_bloom"].astype(str).str.strip() != "").any():
                # Migrate to feedback_store for future UI loads
                mig = df[["review_id", "human_valid", "human_type", "human_bloom", "human_notes"]].copy()
                mig["reviewer_id"] = "default"
                mig.to_csv(fb_path, index=False, encoding="utf-8-sig")
                print(f"[INFO] Migrated legacy skill feedback to {fb_path}")
            df["reviewer_id"] = "default"
            return df
    return pd.DataFrame()


def import_skills_feedback(output_dir: Path, feedback_dir: Path) -> None:
    """Import skill feedback from feedback_store or legacy expert_review_skills.csv."""
    filled = _load_skill_feedback(output_dir, feedback_dir)
    if filled.empty:
        print("[INFO] No skill feedback found.")
        return

    # Aggregate by review_id (multi-reviewer: majority vote)
    valid_bloom = {v for v in VALID_HUMAN_BLOOM if v}
    valid_types = {"Hard", "Soft", "Both"}

    bloom_corrections = {}
    type_corrections = {}
    verified_rows = []

    for review_id, grp in filled.groupby("review_id"):
        hv_list = grp["human_valid"].astype(str).str.strip().str.lower().tolist()
        hb_list = grp["human_bloom"].astype(str).str.strip().tolist()
        ht_list = grp["human_type"].astype(str).str.strip().tolist()
        hv = _majority([v for v in hv_list if v in ("valid", "invalid")])
        hb = _majority([v for v in hb_list if v in valid_bloom])
        ht = _majority([v for v in ht_list if v in valid_types])

        if not hv and not hb and not ht:
            continue

        row = grp.iloc[0].to_dict()
        row["human_valid"] = hv
        row["human_bloom"] = hb
        row["human_type"] = ht
        if "skill" in row and hb:
            bloom_corrections[str(row["skill"]).strip()] = hb
        if "skill" in row and ht:
            type_corrections[str(row["skill"]).strip()] = ht
        if hv == "valid":
            verified_rows.append(row)

    if not verified_rows and not bloom_corrections and not type_corrections:
        print("[INFO] No skill feedback found.")
        return

    # Build human_verified_skills.csv
    if verified_rows:
        verified = pd.DataFrame(verified_rows)
        out_cols = [c for c in ["review_id", "job_id", "skill", "type", "bloom"] if c in verified.columns]
        if out_cols:
            verified[out_cols].to_csv(feedback_dir / "human_verified_skills.csv", index=False, encoding="utf-8-sig")
            print(f"[INFO] Saved {len(verified)} verified skills to human_verified_skills.csv")

    if bloom_corrections:
        (feedback_dir / "bloom_corrections.json").write_text(
            json.dumps(bloom_corrections, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INFO] Saved {len(bloom_corrections)} Bloom corrections")
    if type_corrections:
        (feedback_dir / "type_corrections.json").write_text(
            json.dumps(type_corrections, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INFO] Saved {len(type_corrections)} type corrections")


def _load_knowledge_feedback(output_dir: Path, feedback_dir: Path) -> pd.DataFrame:
    """Load knowledge feedback: prefer feedback_store, else legacy. Migrates legacy on first import."""
    fb_path = feedback_dir / "knowledge_feedback.csv"
    template_path = output_dir / "expert_review_knowledge.csv"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        if not fb.empty:
            if template_path.exists():
                tpl = pd.read_csv(template_path)
                if "knowledge" in tpl.columns:
                    return fb.merge(tpl[["review_id", "knowledge"]], on="review_id", how="left")
            return fb
    if template_path.exists():
        df = pd.read_csv(template_path)
        if "human_valid" in df.columns and df["human_valid"].astype(str).str.strip().isin(["valid", "invalid"]).any():
            mig = df[["review_id", "human_valid", "human_notes"]].copy()
            mig["reviewer_id"] = "default"
            mig.to_csv(fb_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] Migrated legacy knowledge feedback to {fb_path}")
        if "human_valid" in df.columns:
            df["reviewer_id"] = "default"
            return df
    return pd.DataFrame()


def import_knowledge_feedback(output_dir: Path, feedback_dir: Path) -> None:
    """Import knowledge feedback from feedback_store or legacy."""
    filled = _load_knowledge_feedback(output_dir, feedback_dir)
    if filled.empty:
        print("[INFO] No knowledge feedback found.")
        return

    verified_rows = []
    for review_id, grp in filled.groupby("review_id"):
        hv_list = grp["human_valid"].astype(str).str.strip().str.lower().tolist()
        hv = _majority([v for v in hv_list if v in ("valid", "invalid")])
        if hv == "valid":
            verified_rows.append(grp.iloc[0].to_dict())

    if verified_rows:
        verified = pd.DataFrame(verified_rows)
        out_cols = [c for c in ["review_id", "knowledge"] if c in verified.columns]
        if out_cols:
            verified[out_cols].to_csv(feedback_dir / "human_verified_knowledge.csv", index=False, encoding="utf-8-sig")
            print(f"[INFO] Saved {len(verified)} verified knowledge items")


def _load_competency_feedback(output_dir: Path, feedback_dir: Path) -> pd.DataFrame:
    """Load competency feedback: prefer feedback_store, else legacy. Migrates legacy on first import."""
    fb_path = feedback_dir / "competency_feedback.csv"
    path = output_dir / "expert_review_competencies.csv"
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        if not fb.empty:
            return fb
    if path.exists():
        df = pd.read_csv(path)
        has_feedback = (
            ("human_quality" in df.columns and df["human_quality"].astype(str).str.strip().ne("").any())
            or ("human_relevant" in df.columns and df["human_relevant"].astype(str).str.strip().ne("").any())
        )
        if ("human_quality" in df.columns or "human_relevant" in df.columns) and has_feedback:
            cols = ["competency_id", "human_quality", "human_relevant", "human_notes"]
            if "human_skill_focus" in df.columns:
                cols.insert(3, "human_skill_focus")
            mig = df[[c for c in cols if c in df.columns]].copy()
            mig["reviewer_id"] = "default"
            mig.to_csv(fb_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] Migrated legacy competency feedback to {fb_path}")
        if "human_quality" in df.columns or "human_relevant" in df.columns:
            df["reviewer_id"] = "default"
            return df
    return pd.DataFrame()


def import_competency_feedback(output_dir: Path, feedback_dir: Path) -> None:
    """Import competency assessment from feedback_store or legacy."""
    filled = _load_competency_feedback(output_dir, feedback_dir)
    if filled.empty:
        print("[INFO] No competency feedback found.")
        return

    assessments = {}
    for cid, grp in filled.groupby("competency_id"):
        cid = str(cid).strip()
        if not cid:
            continue
        hq_list = grp["human_quality"].astype(str).str.strip().tolist()
        hr_list = grp["human_relevant"].astype(str).str.strip().tolist()
        hn_list = grp["human_notes"].astype(str).str.strip().tolist()
        hq = _majority([v for v in hq_list if v])
        hr = _majority([v for v in hr_list if v])
        hn = _majority([v for v in hn_list if v])
        hsf = ""
        if "human_skill_focus" in grp.columns:
            hsf_list = grp["human_skill_focus"].astype(str).str.strip().tolist()
            hsf = _majority([v for v in hsf_list if v])
        if hq or hr or hsf or hn:
            assessments[cid] = {"quality": hq, "relevant": hr, "skill_focus": hsf, "notes": hn}

    if assessments:
        (feedback_dir / "competency_assessments.json").write_text(
            json.dumps(assessments, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INFO] Saved {len(assessments)} competency assessments")


def _fleiss_kappa(ratings_matrix: list) -> float:
    """
    Compute Fleiss' Kappa for 3+ raters.
    ratings_matrix: list of lists; rows=items, cols=raters; cells=category labels (str or int).
    Returns kappa in [-1, 1]; 0=chance, 1=perfect agreement.
    """
    n = len(ratings_matrix)  # subjects
    if n == 0:
        return 0.0
    N = len(ratings_matrix[0])  # raters
    if N < 3:
        return 0.0

    all_cats = sorted(set(cell for row in ratings_matrix for cell in row if cell is not None and str(cell).strip()))
    if not all_cats:
        return 0.0
    k = len(all_cats)
    cat2idx = {c: i for i, c in enumerate(all_cats)}

    # n_ij = count of raters who assigned subject i to category j
    n_ij = []
    for row in ratings_matrix:
        counts = [0] * k
        for cell in row:
            if cell is not None and str(cell).strip():
                c = str(cell).strip().lower()
                if c in cat2idx:
                    counts[cat2idx[c]] += 1
        n_ij.append(counts)

    n_ij = np.array(n_ij)
    # P_i = (1/(N*(N-1))) * sum_j(n_ij*(n_ij-1))
    P_i = (1.0 / (N * (N - 1))) * np.sum(n_ij * (n_ij - 1), axis=1)
    P_bar = np.mean(P_i)

    # p_j = proportion of all assignments to category j
    p_j = np.sum(n_ij, axis=0) / (n * N)
    P_e = np.sum(p_j ** 2)

    if P_e >= 1.0:
        return 1.0
    return float((P_bar - P_e) / (1.0 - P_e))


def _cohens_kappa(labels1, labels2) -> float:
    """Compute Cohen's Kappa for two lists of labels."""
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0
    all_labels = sorted(set(labels1) | set(labels2))
    n = len(labels1)
    if n == 0:
        return 0.0
    observed_agree = sum(1 for a, b in zip(labels1, labels2) if a == b) / n
    freq1 = {l: labels1.count(l) / n for l in all_labels}
    freq2 = {l: labels2.count(l) / n for l in all_labels}
    expected_agree = sum(freq1.get(l, 0) * freq2.get(l, 0) for l in all_labels)
    if expected_agree >= 1.0:
        return 1.0
    return (observed_agree - expected_agree) / (1.0 - expected_agree)


def compute_inter_rater_reliability(output_dir: Path, feedback_dir: Path) -> None:
    """Compute IRR (Cohen's Kappa) when multiple reviewers assessed the same items."""
    report = {}

    for fb_name, id_col, label_col in [
        ("skill_feedback.csv", "review_id", "human_valid"),
        ("knowledge_feedback.csv", "review_id", "human_valid"),
        ("competency_feedback.csv", "competency_id", "human_quality"),
    ]:
        path = feedback_dir / fb_name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "reviewer_id" not in df.columns or label_col not in df.columns:
            continue

        df["reviewer_id"] = df["reviewer_id"].astype(str).str.strip()
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()
        df = df[(df["reviewer_id"] != "") & (df[label_col] != "")].copy()

        reviewers = sorted(df["reviewer_id"].unique())
        if len(reviewers) < 2:
            continue

        reviewer_dfs = {r: df[df["reviewer_id"] == r] for r in reviewers}
        shared = set(reviewer_dfs[reviewers[0]][id_col])
        for r in reviewers[1:]:
            shared &= set(reviewer_dfs[r][id_col])
        if len(shared) < 5:
            continue

        shared_ids = sorted(shared)
        r1_map = dict(zip(reviewer_dfs[reviewers[0]][id_col], reviewer_dfs[reviewers[0]][label_col]))
        r2_map = dict(zip(reviewer_dfs[reviewers[1]][id_col], reviewer_dfs[reviewers[1]][label_col]))
        labels1 = [r1_map[sid] for sid in shared_ids]
        labels2 = [r2_map[sid] for sid in shared_ids]
        kappa = _cohens_kappa(labels1, labels2)
        pct_agree = sum(1 for a, b in zip(labels1, labels2) if a == b) / len(labels1) * 100
        disagreed = [sid for sid, a, b in zip(shared_ids, labels1, labels2) if a != b]

        entry = {
            "reviewers": reviewers,
            "shared_items": len(shared_ids),
            "cohens_kappa": round(kappa, 4),
            "pct_agreement": round(pct_agree, 1),
            "disagreed_ids": disagreed[:20],
        }
        if len(reviewers) >= 3:
            ratings_matrix = []
            for sid in shared_ids:
                row = []
                for r in reviewers:
                    rdf = reviewer_dfs[r]
                    match = rdf[rdf[id_col] == sid]
                    val = match[label_col].iloc[0] if len(match) > 0 else ""
                    row.append(str(val).strip() if pd.notna(val) else "")
                ratings_matrix.append(row)
            entry["fleiss_kappa"] = round(_fleiss_kappa(ratings_matrix), 4)
        report[fb_name] = entry

    if report:
        out_path = feedback_dir / "inter_rater_report.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Inter-rater reliability report saved to {out_path}")
        for fb_name, r in report.items():
            msg = f"  {fb_name}: cohens_kappa={r['cohens_kappa']:.3f}, agree={r['pct_agreement']:.1f}% ({r['shared_items']} shared items)"
            if "fleiss_kappa" in r:
                msg += f", fleiss_kappa={r['fleiss_kappa']:.3f}"
            print(msg)
    else:
        print("[INFO] No multi-reviewer data found; skipping inter-rater report.")


def import_recommendation_feedback(labels_dir: Path) -> None:
    """Import expert priority labels for recommendations; compute IRR on overlap items.

    Reads DATA/labels/recommendations_for_review.csv (with reviewer_id and priority_label
    filled in by experts). Merges by rank using majority vote. Computes Cohen's Kappa
    on items where is_overlap=True (labeled by 2+ reviewers independently).

    Outputs:
        DATA/labels/recommendations_reviewed.csv  (merged labels)
        DATA/labels/recommendations_irr.json       (Cohen's Kappa + agreement)
    """
    review_path = labels_dir / "recommendations_for_review.csv"
    if not review_path.exists():
        print("[INFO] No recommendations_for_review.csv found; skipping recommendation IRR.")
        return

    df = pd.read_csv(review_path)
    required = {"priority_label", "reviewer_id", "rank"}
    if not required.issubset(set(df.columns)):
        print(f"[WARN] recommendations_for_review.csv missing columns: "
              f"{required - set(df.columns)}")
        return

    df["priority_label"] = df["priority_label"].astype(str).str.strip().str.lower()
    df["reviewer_id"] = df["reviewer_id"].astype(str).str.strip()
    df = df[df["priority_label"].isin(["yes", "no", "partial"]) & (df["reviewer_id"] != "")].copy()
    if df.empty:
        print("[INFO] No filled recommendation priority labels found.")
        return

    # Majority vote by rank
    merged_rows = []
    for rank_val, grp in df.groupby("rank"):
        votes = grp["priority_label"].tolist()
        label = _majority(votes)
        row = grp.iloc[0].to_dict()
        row["priority_label"] = label
        merged_rows.append(row)
    merged = pd.DataFrame(merged_rows)

    out_path = labels_dir / "recommendations_reviewed.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved merged recommendation labels ({len(merged)} items) to {out_path}")

    # IRR on overlap items (is_overlap == True)
    if "is_overlap" not in df.columns:
        print("[INFO] No is_overlap column; skipping recommendation IRR.")
        return

    overlap_df = df[df["is_overlap"].astype(str).str.lower().isin(["true", "1"])].copy()
    reviewers = [r for r in overlap_df["reviewer_id"].unique() if r]
    if len(reviewers) < 2:
        print("[INFO] Need ≥2 reviewers on overlap items for recommendation IRR.")
        return

    pivot = overlap_df.pivot_table(
        index="rank", columns="reviewer_id",
        values="priority_label", aggfunc="first"
    ).dropna()
    valid_reviewers = [r for r in reviewers if r in pivot.columns]
    if len(valid_reviewers) < 2 or len(pivot) < 5:
        print(f"[INFO] Insufficient overlap for recommendation IRR "
              f"(n={len(pivot)}, reviewers={len(valid_reviewers)}).")
        return

    r1, r2 = valid_reviewers[0], valid_reviewers[1]
    labels1 = pivot[r1].tolist()
    labels2 = pivot[r2].tolist()
    kappa = _cohens_kappa(labels1, labels2)
    po = sum(1 for a, b in zip(labels1, labels2) if a == b) / len(labels1)
    irr_result = {
        "status": "ok",
        "reviewers": [r1, r2],
        "overlap_items": len(pivot),
        "observed_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
        "kappa_interpretation": (
            "almost perfect" if kappa > 0.80 else
            "substantial" if kappa > 0.60 else
            "moderate" if kappa > 0.40 else
            "fair" if kappa > 0.20 else "slight or worse"
        ),
    }
    irr_path = labels_dir / "recommendations_irr.json"
    irr_path.write_text(json.dumps(irr_result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Recommendation IRR: kappa={kappa:.4f} ({irr_result['kappa_interpretation']}), "
          f"agreement={po:.1%} (n={len(pivot)})")


def main():
    parser = argparse.ArgumentParser(
        description="Import human feedback from expert_review CSVs into feedback_store."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing expert_review_*.csv (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default=None,
        help="Feedback store directory (default: PROJECT_ROOT/feedback_store)",
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    feedback_dir = Path(args.feedback_dir) if args.feedback_dir else Path(config.PROJECT_ROOT) / "feedback_store"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading from {out_dir}, writing to {feedback_dir}")

    import_skills_feedback(out_dir, feedback_dir)
    import_knowledge_feedback(out_dir, feedback_dir)
    import_competency_feedback(out_dir, feedback_dir)
    compute_inter_rater_reliability(out_dir, feedback_dir)

    labels_dir_path = Path(config.PROJECT_ROOT) / "DATA" / "labels"
    import_recommendation_feedback(labels_dir_path)

    print("[INFO] Import complete.")


if __name__ == "__main__":
    main()
