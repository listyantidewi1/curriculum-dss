"""
validate_parameters.py

Validates pipeline scoring parameters against human review data.

Metrics:
    - AUC-ROC per scoring column
    - Brier score (calibration quality)
    - Calibration curve (10-bin reliability diagram data)
    - Youden's J optimal threshold
    - 5-fold cross-validated AUC
    - Confusion matrix at optimal threshold
    - Calibrated threshold exported for verify_skills.py

Inputs:
    - DATA/labels/gold_skills.csv (optional, preferred if available)
    - config.OUTPUT_DIR/expert_review_skills.csv (scores, review_id)
    - feedback_store/skill_feedback.csv (human_valid merged by review_id)

Outputs:
    - parameter_validation_report.json
    - feedback_store/calibrated_threshold.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config

DEFAULT_FEEDBACK_DIR = Path(config.PROJECT_ROOT) / "feedback_store"
LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def _majority_vote(votes: list) -> str:
    """Return majority vote for human_valid, or first non-empty if tie."""
    votes = [str(v).strip().lower() for v in votes if v and str(v).strip() and str(v).strip().lower() in ("valid", "invalid")]
    if not votes:
        return ""
    c = Counter(votes)
    return c.most_common(1)[0][0]


def load_reviewed_skills(output_dir: Path, feedback_dir: Path) -> pd.DataFrame:
    """Load labeled skills: prefer gold set, then expert_review_skills merged with skill_feedback."""
    gold_path = LABELS_DIR / "gold_skills.csv"
    if gold_path.exists():
        df = pd.read_csv(gold_path)
        if "is_correct" in df.columns and "confidence_score" in df.columns:
            df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
            df = df[df["is_correct"].isin(["yes", "no"])].copy()
            if not df.empty:
                df["is_valid"] = (df["is_correct"] == "yes").astype(int)
                df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
                df = df.dropna(subset=["confidence_score"])
                if not df.empty:
                    print(f"[INFO] Using gold set ({len(df)} items)")
                    return df

    # Load expert_review_skills.csv (has scores) and merge with skill_feedback.csv (has human_valid)
    tpl_path = output_dir / "expert_review_skills.csv"
    fb_path = feedback_dir / "skill_feedback.csv"
    if not tpl_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(tpl_path)
    if "review_id" not in df.columns or "confidence_score" not in df.columns:
        return pd.DataFrame()

    # Merge with feedback_store/skill_feedback.csv to get human_valid
    if fb_path.exists():
        fb = pd.read_csv(fb_path)
        if not fb.empty and "review_id" in fb.columns and "human_valid" in fb.columns:
            # Aggregate by review_id: majority vote for human_valid
            hv_agg = fb.groupby("review_id")["human_valid"].apply(
                lambda x: _majority_vote(x.dropna().astype(str).tolist())
            ).reset_index()
            hv_agg.columns = ["review_id", "human_valid"]
            df = df.merge(hv_agg, on="review_id", how="inner", suffixes=("", "_fb"))
            if "human_valid_fb" in df.columns:
                df["human_valid"] = df["human_valid_fb"].fillna(df.get("human_valid", ""))
                df = df.drop(columns=["human_valid_fb"], errors="ignore")
        else:
            # Fallback: use human_valid from template if present (legacy)
            if "human_valid" not in df.columns:
                return pd.DataFrame()
    else:
        # Legacy: human_valid in expert_review_skills.csv
        if "human_valid" not in df.columns:
            return pd.DataFrame()

    if "human_valid" not in df.columns or df.empty:
        return pd.DataFrame()
    df["human_valid"] = df["human_valid"].astype(str).str.strip().str.lower()
    df = df[df["human_valid"].isin(["valid", "invalid"])].copy()
    if df.empty:
        return df
    df["is_valid"] = (df["human_valid"] == "valid").astype(int)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
    df = df.dropna(subset=["confidence_score"])
    return df


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray):
    """Compute AUC-ROC without sklearn dependency using trapezoidal rule."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true = y_true[desc_idx]
    y_score = y_score[desc_idx]

    distinct_idxs = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate([distinct_idxs, [len(y_true) - 1]])

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]

    total_pos = y_true.sum()
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5, [], [], []

    tpr = np.concatenate([[0], tps / total_pos])
    fpr = np.concatenate([[0], fps / total_neg])
    thresholds = y_score[threshold_idxs]

    # NumPy 2.x removed the alias `trapz`; prefer `trapz` if available
    # but fall back to `trapezoid` for newer NumPy versions.
    if hasattr(np, "trapz"):
        auc = float(np.trapz(tpr, fpr))
    else:
        auc = float(np.trapezoid(tpr, fpr))
    return auc, fpr.tolist(), tpr.tolist(), thresholds.tolist()


def youden_j_optimal(fpr_list, tpr_list, thresholds):
    """Find the threshold that maximizes Youden's J = TPR - FPR."""
    if not thresholds:
        return None
    fpr_arr = np.array(fpr_list[1:])
    tpr_arr = np.array(tpr_list[1:])
    j = tpr_arr - fpr_arr
    best_idx = int(np.argmax(j))
    return {
        "threshold": float(thresholds[best_idx]),
        "tpr": float(tpr_arr[best_idx]),
        "fpr": float(fpr_arr[best_idx]),
        "youden_j": float(j[best_idx]),
    }


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10) -> dict:
    """Compute calibration curve data (reliability diagram)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    observed_freq = []
    predicted_mean = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        count = mask.sum()
        if count > 0:
            bin_centers.append(round((bins[i] + bins[i + 1]) / 2, 3))
            observed_freq.append(round(float(y_true[mask].mean()), 4))
            predicted_mean.append(round(float(y_prob[mask].mean()), 4))
            bin_counts.append(int(count))

    max_cal_error = 0.0
    for obs, pred in zip(observed_freq, predicted_mean):
        max_cal_error = max(max_cal_error, abs(obs - pred))

    return {
        "n_bins": n_bins,
        "bin_centers": bin_centers,
        "observed_freq": observed_freq,
        "predicted_mean": predicted_mean,
        "bin_counts": bin_counts,
        "max_calibration_error": round(max_cal_error, 4),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray,
                             threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": round(threshold, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def cross_validated_auc(y_true: np.ndarray, y_score: np.ndarray,
                        k: int = 5, seed: int = 42) -> dict:
    """K-fold cross-validated AUC (shuffle-split)."""
    n = len(y_true)
    if n < k * 5:
        return {"status": "insufficient_data", "n": n}

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // k
    aucs = []

    for i in range(k):
        test_idx = indices[i * fold_size:(i + 1) * fold_size]
        y_t = y_true[test_idx]
        y_s = y_score[test_idx]
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        auc_val, _, _, _ = compute_auc_roc(y_t, y_s)
        aucs.append(auc_val)

    if not aucs:
        return {"status": "degenerate_folds"}

    return {
        "status": "ok",
        "k": k,
        "fold_aucs": [round(a, 4) for a in aucs],
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc": round(float(np.std(aucs)), 4),
    }


def cross_validated_threshold(y_true: np.ndarray, y_score: np.ndarray,
                              k: int = 5, seed: int = 42) -> dict:
    """Select Youden-optimal threshold via k-fold CV to avoid circularity.

    Each fold: fit threshold on (k-1) training folds, evaluate on held-out
    fold.  The exported threshold is the *mean* of per-fold optimal
    thresholds, not the full-data optimum.
    """
    n = len(y_true)
    if n < k * 5:
        return {"status": "insufficient_data", "n": n}

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // k

    fold_thresholds = []
    fold_test_metrics = []

    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        y_train, s_train = y_true[train_idx], y_score[train_idx]
        y_test, s_test = y_true[test_idx], y_score[test_idx]

        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue
        if y_test.sum() == 0 or y_test.sum() == len(y_test):
            continue

        _, fpr_tr, tpr_tr, thr_tr = compute_auc_roc(y_train, s_train)
        opt_train = youden_j_optimal(fpr_tr, tpr_tr, thr_tr)
        if opt_train is None:
            continue

        thr = opt_train["threshold"]
        fold_thresholds.append(thr)

        cm = compute_confusion_matrix(y_test, s_test, thr)
        fold_test_metrics.append(cm)

    if not fold_thresholds:
        return {"status": "degenerate_folds"}

    mean_thr = float(np.mean(fold_thresholds))
    std_thr = float(np.std(fold_thresholds))
    mean_prec = float(np.mean([m["precision"] for m in fold_test_metrics]))
    mean_recall = float(np.mean([m["recall"] for m in fold_test_metrics]))

    return {
        "status": "ok",
        "k": k,
        "fold_thresholds": [round(t, 4) for t in fold_thresholds],
        "cv_threshold": round(mean_thr, 4),
        "cv_threshold_std": round(std_thr, 4),
        "mean_test_precision": round(mean_prec, 4),
        "mean_test_recall": round(mean_recall, 4),
    }


def validate_column(df: pd.DataFrame, col: str) -> dict:
    """Full validation for a single score column: AUC, Brier, calibration, CV."""
    if col not in df.columns:
        return {"column": col, "status": "missing"}
    subset = df.dropna(subset=[col])
    if len(subset) < 10:
        return {"column": col, "status": "insufficient_data", "n": len(subset)}

    y_true = subset["is_valid"].values
    y_score = pd.to_numeric(subset[col], errors="coerce").fillna(0).values

    auc, fpr, tpr, thresholds = compute_auc_roc(y_true, y_score)
    optimal = youden_j_optimal(fpr, tpr, thresholds)
    corr = float(np.corrcoef(y_score, y_true)[0, 1]) if y_true.std() > 0 else 0.0
    brier = compute_brier_score(y_true, y_score)
    cal = compute_calibration_curve(y_true, y_score)
    cv = cross_validated_auc(y_true, y_score)
    cv_thr = cross_validated_threshold(y_true, y_score)

    confusion = {}
    if optimal:
        confusion = compute_confusion_matrix(y_true, y_score, optimal["threshold"])

    cv_confusion = {}
    if cv_thr.get("status") == "ok":
        cv_confusion = compute_confusion_matrix(y_true, y_score, cv_thr["cv_threshold"])

    return {
        "column": col,
        "n": len(subset),
        "n_valid": int(y_true.sum()),
        "n_invalid": int(len(y_true) - y_true.sum()),
        "auc_roc": round(auc, 4),
        "pearson_corr": round(corr, 4),
        "brier_score": round(brier, 4),
        "calibration": cal,
        "cross_validated_auc": cv,
        "optimal_threshold_full_data": optimal,
        "cv_threshold": cv_thr,
        "confusion_at_optimal": confusion,
        "confusion_at_cv_threshold": cv_confusion,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate pipeline scoring parameters against human review data."
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(config.OUTPUT_DIR),
        help="Directory containing expert_review_skills.csv",
    )
    parser.add_argument(
        "--output", type=str, default="parameter_validation_report.json",
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
    df = load_reviewed_skills(out_dir, feedback_dir)

    if df.empty:
        print("[WARN] No reviewed skills with human_valid found. "
              "Run the review workflow first.")
        report = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(df)} reviewed skills "
              f"({df['is_valid'].sum()} valid, "
              f"{(~df['is_valid'].astype(bool)).sum()} invalid)")

        score_cols = ["confidence_score", "semantic_density", "context_agreement"]
        results = [validate_column(df, c) for c in score_cols]

        report = {
            "status": "ok",
            "total_reviewed": len(df),
            "validations": results,
        }

        for r in results:
            if "auc_roc" in r:
                print(f"  {r['column']}:")
                print(f"    AUC-ROC={r['auc_roc']:.4f}, "
                      f"Brier={r['brier_score']:.4f}, "
                      f"corr={r['pearson_corr']:.4f}")
                print(f"    Max calibration error={r['calibration']['max_calibration_error']:.4f}")
                if r.get("cross_validated_auc", {}).get("status") == "ok":
                    cv = r["cross_validated_auc"]
                    print(f"    CV AUC={cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")
                if r.get("optimal_threshold_full_data"):
                    opt = r["optimal_threshold_full_data"]
                    print(f"    Full-data threshold={opt['threshold']:.3f} "
                          f"(TPR={opt['tpr']:.3f}, FPR={opt['fpr']:.3f})")
                cv_thr = r.get("cv_threshold", {})
                if cv_thr.get("status") == "ok":
                    print(f"    CV threshold={cv_thr['cv_threshold']:.3f} "
                          f"+/- {cv_thr['cv_threshold_std']:.3f} "
                          f"(test P={cv_thr['mean_test_precision']:.3f}, "
                          f"R={cv_thr['mean_test_recall']:.3f})")
                if r.get("confusion_at_cv_threshold"):
                    cm = r["confusion_at_cv_threshold"]
                    print(f"    At CV threshold: P={cm['precision']:.3f}, "
                          f"R={cm['recall']:.3f}, F1={cm['f1']:.3f}")

        # Export calibrated threshold for verify_skills.py (CV-based to avoid circularity)
        conf_result = next((r for r in results if r["column"] == "confidence_score"), None)
        if conf_result:
            cv_thr = conf_result.get("cv_threshold", {})
            if cv_thr.get("status") == "ok":
                threshold_val = cv_thr["cv_threshold"]
                threshold_source = "cv_threshold"
            elif conf_result.get("optimal_threshold_full_data"):
                threshold_val = conf_result["optimal_threshold_full_data"]["threshold"]
                threshold_source = "full_data_fallback"
            else:
                threshold_val = None
                threshold_source = None

            if threshold_val is not None:
                cal_threshold = {
                    "source": "validate_parameters.py",
                    "threshold_selection": threshold_source,
                    "confidence_threshold": threshold_val,
                    "auc_roc": conf_result["auc_roc"],
                    "brier_score": conf_result["brier_score"],
                    "cv_details": cv_thr if cv_thr.get("status") == "ok" else {},
                    "confusion_at_threshold": conf_result.get("confusion_at_cv_threshold", {}),
                }
                cal_path = feedback_dir / "calibrated_threshold.json"
                cal_path.write_text(json.dumps(cal_threshold, indent=2), encoding="utf-8")
                print(f"[INFO] Exported calibrated threshold ({threshold_source}) to {cal_path}")

    report_path = out_dir / args.output
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Saved report to {report_path}")


if __name__ == "__main__":
    main()
