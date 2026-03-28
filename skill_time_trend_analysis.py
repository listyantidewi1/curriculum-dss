"""
skill_time_trend_analysis.py

Compute statistically rigorous time trends for skills using
advanced_skills_with_dates.csv.  Uses scipy.stats.linregress for
slope, p-value, and R-squared, with Benjamini-Hochberg FDR correction
to control false discoveries across many skills.

Scientific methods (see SCIENTIFIC_METHODOLOGY.md §8):
    - Linear regression per skill (freq vs month_idx)
    - Benjamini-Hochberg q-values; trend_label: Emerging/Declining/Stable (q < 0.05)
    - Minimum-slope filter to require practical significance alongside statistical significance
    - Durbin-Watson test to flag potential autocorrelation in residuals

Limitations:
    - Linear model assumes monotonic trend; non-linear patterns (U-shaped, seasonal)
      are not captured.  Residual plots should be inspected for high-value skills.
    - Autocorrelation in monthly counts is plausible (e.g., hiring seasonality).
      The Durbin-Watson statistic is reported but no correction is applied;
      p-values may be anti-conservative when DW << 2 or DW >> 2.

Outputs:
    skill_time_trends.csv  (with slope, p_value, q_value, r_squared, trend_label, durbin_watson)
    trend_stability_report.json  (optional, with --stability flag)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

import config


MIN_DATA_POINTS = 5


def compute_trend_group(grp: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Return (slope, p_value, r_squared, durbin_watson) from linear regression.

    Durbin-Watson tests for serial autocorrelation in residuals (2.0 = none,
    <1.5 = positive autocorrelation, >2.5 = negative autocorrelation).
    """
    x = grp["month_idx"].values.astype(float)
    y = grp["freq"].values.astype(float)

    if len(x) < MIN_DATA_POINTS:
        return 0.0, 1.0, 0.0, float("nan")

    result = sp_stats.linregress(x, y)
    residuals = y - (result.slope * x + result.intercept)
    diffs = np.diff(residuals)
    dw = float(np.sum(diffs ** 2) / np.sum(residuals ** 2)) if np.sum(residuals ** 2) > 0 else float("nan")
    return float(result.slope), float(result.pvalue), float(result.rvalue ** 2), dw


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return q-values (FDR-adjusted p-values) using Benjamini-Hochberg."""
    n = len(p_values)
    if n == 0:
        return np.array([])

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    q_values = np.empty(n)
    q_values[sorted_idx[-1]] = sorted_p[-1]

    for i in range(n - 2, -1, -1):
        rank = i + 1
        adjusted = sorted_p[i] * n / rank
        q_values[sorted_idx[i]] = min(adjusted, q_values[sorted_idx[i + 1]])

    return np.clip(q_values, 0, 1)


def label_trend_fdr(slope: float, q_value: float, alpha: float = 0.05,
                    min_slope: float = 0.0) -> str:
    """Label trend using FDR-corrected q-value and minimum effect-size filter.

    A skill must pass *both* statistical significance (q < alpha) and
    practical significance (|slope| >= min_slope) to be labeled non-Stable.
    """
    if q_value > alpha:
        return "Stable"
    if abs(slope) < min_slope:
        return "Stable"
    return "Emerging" if slope > 0 else "Declining"


def compute_trends_for_df(df: pd.DataFrame, min_jobs: int, fdr_alpha: float = 0.05,
                          min_slope: float = 0.0) -> pd.DataFrame:
    """Core trend computation (reusable for stability analysis)."""
    df["year_month"] = df["job_date"].dt.to_period("M").astype(str)
    months_sorted = sorted(df["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}
    df["month_idx"] = df["year_month"].map(month_to_idx)

    grp = (
        df.groupby(["skill", "year_month", "month_idx"])
        .agg(freq=("job_id", "nunique"))
        .reset_index()
    )

    total_counts = grp.groupby("skill")["freq"].sum().reset_index()
    valid_skills = total_counts[total_counts["freq"] >= min_jobs]["skill"]
    grp = grp[grp["skill"].isin(valid_skills)].copy()

    if grp.empty:
        return pd.DataFrame()

    trend_rows = []
    for skill, g in grp.groupby("skill"):
        g_sorted = g.sort_values("month_idx")
        slope, p_value, r_squared, dw = compute_trend_group(g_sorted)
        trend_rows.append({
            "skill": skill,
            "total_freq": g_sorted["freq"].sum(),
            "n_months": g_sorted["year_month"].nunique(),
            "slope": slope,
            "p_value": p_value,
            "r_squared": r_squared,
            "durbin_watson": round(dw, 4) if not np.isnan(dw) else None,
        })

    trends = pd.DataFrame(trend_rows)

    p_vals = trends["p_value"].values
    trends["q_value"] = benjamini_hochberg(p_vals)

    trends["trend_label"] = trends.apply(
        lambda r: label_trend_fdr(r["slope"], r["q_value"], alpha=fdr_alpha,
                                  min_slope=min_slope),
        axis=1,
    )

    # Flag skills with potential autocorrelation (DW < 1.5 = positive, DW > 2.5 = negative).
    # Flagged skills have anti-conservative p-values; exclude from high-confidence trend claims.
    trends["dw_flagged"] = trends["durbin_watson"].apply(
        lambda dw: bool(dw is not None and not (isinstance(dw, float) and dw != dw)
                        and (dw < 1.5 or dw > 2.5))
    )

    return trends


def stability_analysis(df: pd.DataFrame, min_jobs_values: List[int],
                       top_n: int = 20, n_seeds: int = 3,
                       base_seed: int | None = None) -> dict:
    """Measure stability of top-N emerging skills across min_jobs and seeds."""
    if base_seed is None:
        base_seed = config.RANDOM_SEED
    results = []

    for mj in min_jobs_values:
        seed_sets = []
        seed_sets_clean = []
        for seed_offset in range(n_seeds):
            df_sampled = df.sample(frac=0.9, random_state=base_seed + seed_offset).copy()
            trends = compute_trends_for_df(df_sampled, mj)
            if trends.empty:
                continue
            emerging = set(
                trends[trends["trend_label"] == "Emerging"]
                .nlargest(top_n, "slope")["skill"]
                .tolist()
            )
            seed_sets.append(emerging)
            # Clean set: exclude DW-flagged skills (potential autocorrelation)
            if "dw_flagged" in trends.columns:
                clean_trends = trends[~trends["dw_flagged"]]
            else:
                clean_trends = trends
            emerging_clean = set(
                clean_trends[clean_trends["trend_label"] == "Emerging"]
                .nlargest(top_n, "slope")["skill"]
                .tolist()
            )
            seed_sets_clean.append(emerging_clean)

        if len(seed_sets) < 2:
            results.append({
                "min_jobs": mj, "n_seeds": len(seed_sets),
                "mean_jaccard": 0.0, "top_n": top_n,
            })
            continue

        jaccards = []
        for i in range(len(seed_sets)):
            for j in range(i + 1, len(seed_sets)):
                a, b = seed_sets[i], seed_sets[j]
                if a or b:
                    jacc = len(a & b) / len(a | b) if (a | b) else 0.0
                    jaccards.append(jacc)

        jaccards_clean = []
        for i in range(len(seed_sets_clean)):
            for j in range(i + 1, len(seed_sets_clean)):
                a, b = seed_sets_clean[i], seed_sets_clean[j]
                if a or b:
                    jacc = len(a & b) / len(a | b) if (a | b) else 0.0
                    jaccards_clean.append(jacc)

        results.append({
            "min_jobs": mj,
            "n_seeds": len(seed_sets),
            "mean_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
            "std_jaccard": round(float(np.std(jaccards)), 4) if jaccards else 0.0,
            "mean_jaccard_clean": round(float(np.mean(jaccards_clean)), 4) if jaccards_clean else 0.0,
            "top_n": top_n,
        })

    return {"stability_by_min_jobs": results}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze time trends of skills with FDR-controlled significance."
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(config.OUTPUT_DIR),
        help="Directory where advanced_skills_with_dates.csv is stored.",
    )
    parser.add_argument(
        "--input", type=str, default="advanced_skills_with_dates.csv",
        help="Input CSV (with job_date).",
    )
    parser.add_argument(
        "--output", type=str, default="skill_time_trends.csv",
        help="Output CSV file for skill trends.",
    )
    parser.add_argument(
        "--min_jobs", type=int, default=10,
        help="Minimum number of jobs mentioning a skill to include in trends.",
    )
    parser.add_argument(
        "--only_hard", action="store_true",
        help="If set, only analyze skills where type == 'hard'.",
    )
    parser.add_argument(
        "--fdr_alpha", type=float, default=0.05,
        help="FDR significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--min_slope", type=float, default=0.0,
        help="Minimum absolute slope to label a skill as Emerging/Declining. "
             "Ensures practical significance alongside statistical significance "
             "(e.g., 0.5 = at least 0.5 additional jobs/month).",
    )
    parser.add_argument(
        "--stability", action="store_true",
        help="Run stability analysis across min_jobs settings and seeds.",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    in_path = out_dir / args.input

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found")

    print(f"[INFO] Reading {in_path}")
    df = pd.read_csv(in_path)

    if "job_date" not in df.columns:
        raise ValueError("Input file must contain 'job_date' column. "
                         "Run enrich_with_dates.py first.")

    df["job_date"] = pd.to_datetime(df["job_date"], errors="coerce")
    df = df[df["job_date"].notna()].copy()

    if args.only_hard and "type" in df.columns:
        df = df[df["type"].astype(str).str.lower().isin(["hard", "both"])].copy()

    if args.min_slope > 0:
        print(f"[INFO] Minimum slope filter: |slope| >= {args.min_slope}")

    trends = compute_trends_for_df(df, args.min_jobs, fdr_alpha=args.fdr_alpha,
                                   min_slope=args.min_slope)

    if trends.empty:
        print("[WARN] No skills passed the min_jobs filter; nothing to analyze.")
        return

    n_emerging = (trends["trend_label"] == "Emerging").sum()
    n_declining = (trends["trend_label"] == "Declining").sum()
    n_stable = (trends["trend_label"] == "Stable").sum()
    print(f"[INFO] Trend summary (FDR alpha={args.fdr_alpha}): "
          f"Emerging={n_emerging}, Declining={n_declining}, Stable={n_stable}")
    print(f"[INFO] q-value stats: mean={trends['q_value'].mean():.4f}, "
          f"median={trends['q_value'].median():.4f}")

    dw_summary = {}
    if "durbin_watson" in trends.columns:
        dw_valid = trends["durbin_watson"].dropna()
        if not dw_valid.empty:
            n_pos_ac = int((dw_valid < 1.5).sum())
            n_neg_ac = int((dw_valid > 2.5).sum())
            n_ok = int(((dw_valid >= 1.5) & (dw_valid <= 2.5)).sum())
            dw_summary = {
                "n_skills_with_dw": len(dw_valid),
                "n_positive_autocorr_dw_lt_1_5": n_pos_ac,
                "n_acceptable_1_5_to_2_5": n_ok,
                "n_negative_autocorr_dw_gt_2_5": n_neg_ac,
                "pct_flagged": round((n_pos_ac + n_neg_ac) / len(dw_valid), 4),
                "mean_dw": round(float(dw_valid.mean()), 4),
                "median_dw": round(float(dw_valid.median()), 4),
            }
            print(f"[INFO] Durbin-Watson distribution: "
                  f"DW<1.5 (pos autocorr)={n_pos_ac}, "
                  f"1.5-2.5 (acceptable)={n_ok}, "
                  f"DW>2.5 (neg autocorr)={n_neg_ac} "
                  f"({dw_summary['pct_flagged']:.1%} flagged)")
            if "dw_flagged" in trends.columns:
                n_flagged_emerging = int(
                    trends[(trends["trend_label"] == "Emerging") & trends["dw_flagged"]].shape[0]
                )
                print(f"[INFO] DW-flagged among Emerging skills: {n_flagged_emerging} "
                      f"(treat with caution — potential autocorrelation)")

    out_path = out_dir / args.output
    trends.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved skill trends to {out_path}")

    if args.stability:
        print("[INFO] Running stability analysis...")
        min_jobs_values = [5, 10, 15, 20]
        stab = stability_analysis(df, min_jobs_values, top_n=20, n_seeds=3)

        # Add DW distribution to the stability report
        stab["dw_distribution"] = dw_summary

        # Add clean top-20 emerging (DW-filtered) and its Jaccard vs full top-20
        if "dw_flagged" in trends.columns:
            top20_all = set(
                trends[trends["trend_label"] == "Emerging"]
                .nlargest(20, "slope")["skill"].tolist()
            )
            top20_clean = set(
                trends[(trends["trend_label"] == "Emerging") & ~trends["dw_flagged"]]
                .nlargest(20, "slope")["skill"].tolist()
            )
            union_dw = top20_all | top20_clean
            inter_dw = top20_all & top20_clean
            stab["clean_top20_emerging"] = sorted(top20_clean)
            stab["clean_top20_jaccard_vs_all"] = round(
                len(inter_dw) / len(union_dw), 4) if union_dw else 1.0
            print(f"[INFO] Clean top-20 Jaccard (DW-filtered vs all): "
                  f"{stab['clean_top20_jaccard_vs_all']:.4f}")

        stab_path = out_dir / "trend_stability_report.json"
        stab_path.write_text(json.dumps(stab, indent=2), encoding="utf-8")
        print(f"[INFO] Saved stability report to {stab_path}")
        for entry in stab["stability_by_min_jobs"]:
            clean_j = entry.get('mean_jaccard_clean', entry['mean_jaccard'])
            print(f"  min_jobs={entry['min_jobs']}: "
                  f"Jaccard={entry['mean_jaccard']:.3f} +/- {entry.get('std_jaccard', 0):.3f} "
                  f"(clean={clean_j:.3f})")

    try:
        top_emerging = trends[trends["trend_label"] == "Emerging"].nlargest(10, "slope")
        top_declining = trends[trends["trend_label"] == "Declining"].nsmallest(10, "slope")

        if not top_emerging.empty or not top_declining.empty:
            plt.figure(figsize=(10, 6))
            if not top_emerging.empty:
                sns.barplot(data=top_emerging, x="slope", y="skill",
                            label="Emerging", orient="h")
            if not top_declining.empty:
                sns.barplot(data=top_declining, x="slope", y="skill",
                            color="red", label="Declining", orient="h")
            plt.xlabel("Slope of monthly job count")
            plt.ylabel("Skill")
            plt.title("Top Emerging vs Declining Skills (FDR-controlled)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "skill_trend_barplot.png", dpi=300)
            plt.close()
            print("[INFO] Saved skill_trend_barplot.png")
    except Exception as e:
        print(f"[WARN] Could not create trend plot: {e}")


if __name__ == "__main__":
    main()
