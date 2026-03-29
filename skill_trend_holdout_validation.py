"""
skill_trend_holdout_validation.py  —  Item 4: Longitudinal validation for RQ3

Retrospective held-out validation of the trend detection pipeline.

Method:
  1. Split job postings into TRAIN (months 1..N-test_months) and
     TEST (last test_months months).
  2. Compute linear-regression trend signals on TRAIN with FDR correction.
  3. Measure whether predicted emerging / declining skills actually
     grew / shrank in the TEST window.

Metrics reported:
  - emerging_direction_accuracy  : fraction of Emerging predictions where
      mean TEST frequency > last TRAIN frequency  (target > 0.60)
  - declining_direction_accuracy : same for Declining predictions
  - slope_correlation            : Pearson r between TRAIN slope and
      TEST frequency delta  (target > 0.30)
  - precision_emerging_significant : fraction of Emerging predictions that
      had a significant positive slope on TEST data (p < 0.10)

Output:
  results/trend_holdout_validation_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
MIN_MONTHS_TRAIN = 6    # Need at least 6 months for a meaningful trend
MIN_MONTHS_TEST = 2     # Need at least 2 held-out months
MIN_DATA_POINTS = 3     # Minimum (skill, month) observations per split
MIN_SLOPE = 0.05        # Practical significance threshold (same as main analysis)
FDR_ALPHA = 0.05


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_skills_with_dates(results_dir: Path) -> pd.DataFrame:
    for name in [
        "advanced_skills_with_dates.csv",
        "advanced_skills_normalized.csv",
        "advanced_skills.csv",
    ]:
        path = results_dir / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"[INFO] Loaded {name} ({len(df):,} rows)")
            return df
    return pd.DataFrame()


def _build_monthly_freq(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Return (monthly_freq_df, total_month_count).

    monthly_freq_df columns: skill, month_idx, freq
    """
    date_col = next(
        (c for c in ["job_date", "date_posted", "date"] if c in df.columns), None
    )
    if date_col is None:
        return pd.DataFrame(), 0

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["_month"] = df[date_col].dt.to_period("M")

    months_sorted = sorted(df["_month"].unique())
    if not months_sorted:
        return pd.DataFrame(), 0
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    df["month_idx"] = df["_month"].map(month_to_idx)

    skill_col = "canonical_skill" if "canonical_skill" in df.columns else "skill"
    agg = (
        df.groupby([skill_col, "month_idx"])
        .size()
        .reset_index(name="freq")
        .rename(columns={skill_col: "skill"})
    )
    return agg, len(months_sorted)


def _linreg(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, p_value, r_squared)."""
    if len(x) < MIN_DATA_POINTS:
        return 0.0, 1.0, 0.0
    res = sp_stats.linregress(x, y)
    return float(res.slope), float(res.pvalue), float(res.rvalue ** 2)


def _apply_fdr(p_values: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p_values)
    q = np.empty(n)
    q[sorted_idx[-1]] = p_values[sorted_idx[-1]]
    for i in range(n - 2, -1, -1):
        rank = i + 1
        q[sorted_idx[i]] = min(p_values[sorted_idx[i]] * n / rank, q[sorted_idx[i + 1]])
    return np.clip(q, 0.0, 1.0)


# ------------------------------------------------------------------
# Main validation routine
# ------------------------------------------------------------------

def run_holdout_validation(
    results_dir: Path,
    test_months: int = 3,
) -> Dict:
    df = _load_skills_with_dates(results_dir)
    if df.empty:
        return {
            "status": "no_data",
            "message": "No skills-with-dates file found in results dir.",
        }

    monthly, total_months = _build_monthly_freq(df)
    if monthly.empty or total_months == 0:
        return {
            "status": "no_dates",
            "message": "Skills file has no usable date column.",
        }

    if total_months < MIN_MONTHS_TRAIN + MIN_MONTHS_TEST:
        return {
            "status": "insufficient_data",
            "total_months": total_months,
            "message": (
                f"Need ≥ {MIN_MONTHS_TRAIN + MIN_MONTHS_TEST} months of data, "
                f"got {total_months}."
            ),
        }

    split_idx = total_months - test_months
    train = monthly[monthly["month_idx"] < split_idx].copy()
    test = monthly[monthly["month_idx"] >= split_idx].copy()

    # ---- Compute trends on TRAIN ----
    records: List[Dict] = []
    for skill, grp in train.groupby("skill"):
        if len(grp) < MIN_DATA_POINTS:
            continue
        slope, p_val, r2 = _linreg(grp["month_idx"].values, grp["freq"].values)
        records.append({
            "skill": skill,
            "train_slope": slope,
            "train_p": p_val,
            "train_r2": r2,
        })

    if not records:
        return {
            "status": "no_trends",
            "message": "No skills had sufficient TRAIN data for regression.",
        }

    train_df = pd.DataFrame(records)
    train_df["train_q"] = _apply_fdr(train_df["train_p"].values)

    # Classify predictions
    train_df["train_label"] = "Stable"
    train_df.loc[
        (train_df["train_q"] < FDR_ALPHA) & (train_df["train_slope"] >= MIN_SLOPE),
        "train_label",
    ] = "Emerging"
    train_df.loc[
        (train_df["train_q"] < FDR_ALPHA) & (train_df["train_slope"] <= -MIN_SLOPE),
        "train_label",
    ] = "Declining"

    # ---- Compute TEST deltas ----
    last_train_freq = (
        train[train["month_idx"] == split_idx - 1]
        .set_index("skill")["freq"]
    )
    test_mean_freq = test.groupby("skill")["freq"].mean()
    test_slope_map: Dict[str, Tuple[float, float]] = {}
    for skill, grp in test.groupby("skill"):
        if len(grp) >= MIN_DATA_POINTS:
            s, p, _ = _linreg(grp["month_idx"].values, grp["freq"].values)
            test_slope_map[skill] = (s, p)

    train_df["last_train_freq"] = train_df["skill"].map(last_train_freq).fillna(0.0)
    train_df["test_mean_freq"] = train_df["skill"].map(test_mean_freq).fillna(0.0)
    train_df["test_delta"] = train_df["test_mean_freq"] - train_df["last_train_freq"]
    train_df["test_slope"] = train_df["skill"].map(
        lambda s: test_slope_map.get(s, (float("nan"), float("nan")))[0]
    )
    train_df["test_p"] = train_df["skill"].map(
        lambda s: test_slope_map.get(s, (float("nan"), float("nan")))[1]
    )

    # ---- Metrics ----
    emerging = train_df[train_df["train_label"] == "Emerging"]
    declining = train_df[train_df["train_label"] == "Declining"]

    def _dir_acc(subset: pd.DataFrame, sign: int) -> Optional[float]:
        if subset.empty:
            return None
        return float((subset["test_delta"] * sign > 0).mean())

    emerging_acc = _dir_acc(emerging, +1)
    declining_acc = _dir_acc(declining, -1)

    # Fraction of Emerging predictions with a significant positive TEST slope
    prec_emerging: Optional[float] = None
    if not emerging.empty:
        sig_pos = (
            (emerging["test_slope"] > 0) &
            (emerging["test_p"].fillna(1.0) < 0.10)
        ).sum()
        prec_emerging = float(sig_pos / len(emerging))

    # Pearson r between train slope and test delta
    slope_corr: Optional[float] = None
    slope_corr_p: Optional[float] = None
    valid = train_df.dropna(subset=["train_slope", "test_delta"])
    if len(valid) >= 5:
        r, p = sp_stats.pearsonr(valid["train_slope"], valid["test_delta"])
        slope_corr, slope_corr_p = round(float(r), 4), round(float(p), 4)

    # ---- Assemble report ----
    target_met = (
        emerging_acc is not None and emerging_acc > 0.60
        and slope_corr is not None and slope_corr > 0.30
    )

    def _top20(subset: pd.DataFrame) -> List[Dict]:
        cols = ["skill", "train_slope", "train_q", "test_delta", "test_slope"]
        cols = [c for c in cols if c in subset.columns]
        return (
            subset.sort_values("train_slope", ascending=False)
            .head(20)[cols]
            .round(4)
            .to_dict(orient="records")
        )

    report = {
        "status": "ok",
        "rq3_holdout_target_met": target_met,
        "total_months": int(total_months),
        "train_months": int(split_idx),
        "test_months": int(test_months),
        "n_skills_evaluated": int(len(train_df)),
        "n_emerging_predicted": int(len(emerging)),
        "n_declining_predicted": int(len(declining)),
        "emerging_direction_accuracy": round(emerging_acc, 4) if emerging_acc is not None else None,
        "declining_direction_accuracy": round(declining_acc, 4) if declining_acc is not None else None,
        "precision_emerging_significant_test": round(prec_emerging, 4) if prec_emerging is not None else None,
        "train_test_slope_correlation": slope_corr,
        "train_test_slope_correlation_p": slope_corr_p,
        "thresholds": {
            "fdr_alpha": FDR_ALPHA,
            "min_slope": MIN_SLOPE,
            "test_months_held_out": int(test_months),
        },
        "interpretation": {
            "emerging_direction_accuracy": (
                "Fraction of skills predicted Emerging (FDR q<0.05, slope≥0.05) "
                "where mean TEST frequency > last TRAIN month frequency. Target: > 0.60"
            ),
            "slope_correlation": (
                "Pearson r between TRAIN regression slope and actual TEST frequency "
                "delta. Measures whether the pipeline's growth estimate is calibrated. "
                "Target: > 0.30"
            ),
            "precision_emerging_significant_test": (
                "Fraction of Emerging predictions with a statistically significant "
                "positive slope in the TEST window (p < 0.10)."
            ),
        },
        "top_validated_emerging": _top20(emerging),
        "top_validated_declining": _top20(
            declining.sort_values("train_slope")
        ),
    }
    return report


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Longitudinal holdout validation for trend analysis (RQ3)"
    )
    parser.add_argument(
        "--output_dir", default=str(config.OUTPUT_DIR),
        help="Directory containing skill data and where report is saved"
    )
    parser.add_argument(
        "--test_months", type=int, default=3,
        help="Number of most-recent months to hold out (default 3)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running trend holdout validation (test_months={args.test_months}) …")
    report = run_holdout_validation(out_dir, test_months=args.test_months)

    out_path = out_dir / "trend_holdout_validation_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[INFO] Saved → {out_path}")

    if report.get("status") == "ok":
        ea = report.get("emerging_direction_accuracy")
        sc = report.get("train_test_slope_correlation")
        print(f"  Emerging direction accuracy : {ea}")
        print(f"  Train-test slope correlation: {sc}")
        print(f"  RQ3 holdout target met      : {'YES ✓' if report.get('rq3_holdout_target_met') else 'NO — investigate'}")
    else:
        print(f"  Status: {report.get('status')} — {report.get('message', '')}")


if __name__ == "__main__":
    main()
