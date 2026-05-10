"""
recommendations.py

Generates ranked curriculum recommendations with evidence traces.

Scientific methods (see SCIENTIFIC_METHODOLOGY.md §7, 9):
    - priority_score = 0.4×demand + 0.3×trend + 0.3×future (coverage for insights only)
    - Weight sensitivity: Jaccard vs baseline top-20 for alternative weight configs

Each recommendation is a skill/knowledge gap with:
    - priority_score (composite of demand, trend, future_weight; coverage is for insights only)
    - evidence trace (supporting job_ids, trend stats, domain info)
    - curriculum gap status

Supports ablation: signals can be enabled/disabled individually.

Outputs:
    results/recommendations.csv
    results/recommendations_report.json

Evaluation (when gold labels are available):
    results/recommendation_evaluation_report.json
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


# ---------------------------------------------------------------------------
# Signal loaders
# ---------------------------------------------------------------------------

def load_skills(output_dir: Path) -> pd.DataFrame:
    for name in ["advanced_skills_human_filtered.csv", "verified_skills.csv",
                  "advanced_skills.csv"]:
        path = output_dir / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"[INFO] Loaded skills from {name} ({len(df)} rows)")
            return df
    return pd.DataFrame()


def load_trends(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "skill_time_trends.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_future_weights(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "future_skill_weights.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_coverage(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "coverage_report.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Build recommendation table
# ---------------------------------------------------------------------------

def _normalize_skill_for_grouping(text: str) -> str:
    """Normalize skill for grouping (matches future_weight_mapping)."""
    if not text or not isinstance(text, str):
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_vague_hard_skill(skill: str, skill_type: str) -> bool:
    """True if Hard skill with fewer than 2 words (e.g. 'design', 'develop', 'manage').
    Such skills are too generic for curriculum priorities—'design' could mean
    design software, design UI, design systems, etc."""
    if not skill or not isinstance(skill, str):
        return False
    st = str(skill_type).strip().lower()
    if st not in ("hard", "hard skill"):
        return False
    words = str(skill).strip().split()
    return len(words) < 2


def build_skill_demand(skills_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-skill demand statistics. Groups equivalent skills (case/punctuation variants)."""
    if skills_df.empty or "skill" not in skills_df.columns:
        return pd.DataFrame()

    skills_df = skills_df.copy()
    skills_df["_group_key"] = skills_df["skill"].apply(_normalize_skill_for_grouping)
    skills_df = skills_df[skills_df["_group_key"] != ""]

    agg_dict = {
        "demand_freq": ("_group_key", "count"),
        "mean_confidence": ("confidence_score", "mean") if "confidence_score" in skills_df.columns else ("_group_key", "count"),
    }
    grp = skills_df.groupby("_group_key").agg(**agg_dict).reset_index()

    canonical = skills_df.groupby("_group_key")["skill"].apply(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    canonical.columns = ["_group_key", "skill"]
    grp = grp.merge(canonical, on="_group_key")

    if "type" in skills_df.columns:
        type_map = skills_df.groupby("_group_key")["type"].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
        ).reset_index()
        type_map.columns = ["_group_key", "skill_type"]
        grp = grp.merge(type_map, on="_group_key", how="left")

    # bloom column dropped in pipeline-redesign-v2 (Req 1); bloom_level no longer
    # joined onto recommendations.
    if "job_id" in skills_df.columns:
        job_ids = skills_df.groupby("_group_key")["job_id"].apply(
            lambda x: list(x.unique()[:5])
        ).reset_index()
        job_ids.columns = ["_group_key", "example_job_ids"]
        grp = grp.merge(job_ids, on="_group_key", how="left")

    grp = grp.drop(columns=["_group_key"], errors="ignore")
    return grp


def compute_priority_scores(
    demand: pd.DataFrame,
    trends: pd.DataFrame,
    future_weights: pd.DataFrame,
    coverage: pd.DataFrame,
    use_trend: bool = True,
    use_future: bool = True,
    use_coverage: bool = False,
    use_validity: bool = True,
    w_demand: float = 0.40,
    w_trend: float = 0.30,
    w_future: float = 0.30,
    w_coverage: float = 0.0,
) -> pd.DataFrame:
    """Compute priority score per skill. Signals can be ablated."""
    if demand.empty:
        return pd.DataFrame()

    rec = demand.copy()
    skill_col = "skill"

    max_freq = rec["demand_freq"].max()
    rec["demand_norm"] = rec["demand_freq"] / max_freq if max_freq > 0 else 0.0

    # Trend signal
    rec["trend_score_norm"] = 0.0
    rec["trend_label"] = "Unknown"
    rec["trend_q_value"] = 1.0
    if use_trend and not trends.empty and skill_col in trends.columns:
        trend_map = trends.set_index(skill_col)
        # Build normalized index for fallback lookup (canonical form may differ from CSV)
        norm_to_exact_trend = {}
        for idx_val in trend_map.index:
            norm = _normalize_skill_for_grouping(str(idx_val))
            if norm and norm not in norm_to_exact_trend:
                norm_to_exact_trend[norm] = idx_val
        for idx, row in rec.iterrows():
            s = row[skill_col]
            lookup_key = s if s in trend_map.index else norm_to_exact_trend.get(_normalize_skill_for_grouping(s))
            if lookup_key is not None:
                t = trend_map.loc[lookup_key]
                if isinstance(t, pd.DataFrame):
                    t = t.iloc[0]
                label = str(t.get("trend_label", "Stable"))
                q = float(t.get("q_value", t.get("p_value", 1.0)))
                slope = float(t.get("slope", 0))
                rec.at[idx, "trend_label"] = label
                rec.at[idx, "trend_q_value"] = q
                if label == "Emerging":
                    rec.at[idx, "trend_score_norm"] = min(1.0, abs(slope) / 0.5)
                elif label == "Declining":
                    rec.at[idx, "trend_score_norm"] = max(-1.0, -abs(slope) / 0.5)

    # Future weight signal
    rec["future_weight"] = 0.0
    rec["future_domain"] = ""
    rec["mapping_margin"] = 0.0
    if use_future and not future_weights.empty and skill_col in future_weights.columns:
        fw_map = future_weights.drop_duplicates(subset=[skill_col]).set_index(skill_col)
        # Build normalized index for fallback lookup (canonical form may differ from CSV)
        norm_to_exact_fw = {}
        for idx_val in fw_map.index:
            norm = _normalize_skill_for_grouping(str(idx_val))
            if norm and norm not in norm_to_exact_fw:
                norm_to_exact_fw[norm] = idx_val
        for idx, row in rec.iterrows():
            s = row[skill_col]
            lookup_key = s if s in fw_map.index else norm_to_exact_fw.get(_normalize_skill_for_grouping(s))
            if lookup_key is not None:
                fw = fw_map.loc[lookup_key]
                if isinstance(fw, pd.DataFrame):
                    fw = fw.iloc[0]
                rec.at[idx, "future_weight"] = float(fw.get("future_weight", 0))
                rec.at[idx, "future_domain"] = str(fw.get("best_future_domain", ""))
                rec.at[idx, "mapping_margin"] = float(fw.get("mapping_margin", 0))

    fw_max = rec["future_weight"].abs().max()
    rec["future_norm"] = rec["future_weight"] / fw_max if fw_max > 0 else 0.0

    # Coverage gap signal (1.0 = not covered at all, 0.0 = fully covered)
    rec["coverage_gap"] = 1.0
    if use_coverage and not coverage.empty:
        if "coverage_percentage" in coverage.columns:
            mean_cov = coverage["coverage_percentage"].mean() / 100.0
            rec["coverage_gap"] = 1.0 - mean_cov

    # Validity signal (use mean_confidence as proxy)
    rec["validity_score"] = 0.5
    if use_validity and "mean_confidence" in rec.columns:
        rec["validity_score"] = rec["mean_confidence"].fillna(0.5)

    # Composite priority score
    rec["priority_score"] = (
        w_demand * rec["demand_norm"]
        + (w_trend * rec["trend_score_norm"].clip(-1, 1) if use_trend else 0)
        + (w_future * rec["future_norm"] if use_future else 0)
        + (w_coverage * rec["coverage_gap"] if use_coverage else 0)
    )

    rec = rec.sort_values("priority_score", ascending=False).reset_index(drop=True)
    rec.insert(0, "rank", range(1, len(rec) + 1))

    return rec


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def dcg(scores: List[float], k: int) -> float:
    result = 0.0
    for i, s in enumerate(scores[:k]):
        result += s / math.log2(i + 2)
    return result


def ndcg_at_k(relevance: List[float], k: int) -> float:
    actual = dcg(relevance, k)
    ideal = dcg(sorted(relevance, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_recommendations(recs: pd.DataFrame, top_n: int = 20) -> dict:
    """Evaluate top-N recommendations if expert labels are available.

    Expects a column 'expert_priority' in recs (yes/no/partial) for top-N items.
    If not present, returns instructions for labeling.
    """
    if "expert_priority" not in recs.columns:
        return {
            "status": "no_expert_labels",
            "instructions": (
                "Add 'expert_priority' column to the top-N rows of "
                "recommendations.csv with values: yes, no, partial. "
                "Then re-run this script."
            ),
        }

    top = recs.head(top_n).copy()
    top["expert_priority"] = top["expert_priority"].astype(str).str.strip().str.lower()
    labeled = top[top["expert_priority"].isin(["yes", "no", "partial"])]

    if labeled.empty:
        return {"status": "no_labeled_items"}

    n = len(labeled)
    n_yes = (labeled["expert_priority"] == "yes").sum()
    n_partial = (labeled["expert_priority"] == "partial").sum()
    n_no = (labeled["expert_priority"] == "no").sum()

    precision = (n_yes + 0.5 * n_partial) / n if n > 0 else 0.0

    relevance = []
    for _, r in labeled.iterrows():
        ep = r["expert_priority"]
        relevance.append(1.0 if ep == "yes" else (0.5 if ep == "partial" else 0.0))

    ndcg = ndcg_at_k(relevance, top_n)

    result = {
        "status": "ok",
        "top_n": top_n,
        "labeled_items": n,
        "n_yes": int(n_yes),
        "n_partial": int(n_partial),
        "n_no": int(n_no),
        "precision_at_n": round(precision, 4),
        "ndcg_at_n": round(ndcg, 4),
    }
    # Load IRR if available (from import_recommendation_feedback in import_feedback.py)
    irr_path = LABELS_DIR / "recommendations_irr.json"
    if irr_path.exists():
        try:
            import json as _json
            result["irr"] = _json.loads(irr_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return result


def run_coverage_ablation(demand, trends, future_weights, coverage, top_n: int = 20) -> dict:
    """Item 6: Coverage ablation — compare rankings with vs without coverage weight.

    Tests whether including curriculum coverage in the priority formula
    materially changes the top-N recommendations.  If Jaccard overlap is high
    (> 0.80) the 0.0 coverage weight design choice is empirically supported.
    """
    configs = {
        "no_coverage":   {"w_demand": 0.40, "w_trend": 0.30, "w_future": 0.30, "w_coverage": 0.00, "use_coverage": False},
        "coverage_0.10": {"w_demand": 0.36, "w_trend": 0.27, "w_future": 0.27, "w_coverage": 0.10, "use_coverage": True},
        "coverage_0.20": {"w_demand": 0.32, "w_trend": 0.24, "w_future": 0.24, "w_coverage": 0.20, "use_coverage": True},
        "coverage_0.30": {"w_demand": 0.28, "w_trend": 0.21, "w_future": 0.21, "w_coverage": 0.30, "use_coverage": True},
    }
    results: dict = {}
    for name, cfg in configs.items():
        recs = compute_priority_scores(demand, trends, future_weights, coverage, **cfg)
        top_skills = recs.head(top_n)["skill"].tolist() if not recs.empty else []
        results[name] = {"top_skills": top_skills, "n_total": len(recs)}

    baseline_set = set(results["no_coverage"]["top_skills"])
    for name in results:
        variant_set = set(results[name]["top_skills"])
        union = baseline_set | variant_set
        inter = baseline_set & variant_set
        results[name]["jaccard_vs_no_coverage"] = round(len(inter) / len(union), 4) if union else 1.0

    # High Jaccard (>0.80) validates the 0.0 coverage weight design choice.
    j_010 = results["coverage_0.10"]["jaccard_vs_no_coverage"]
    results["_interpretation"] = {
        "conclusion": (
            "Coverage weight design justified (Jaccard>0.80 at 0.10 weight)"
            if j_010 > 0.80 else
            "Coverage weight changes recommendations — consider including"
        ),
        "jaccard_at_coverage_010": j_010,
        "note": "Weights in coverage configs are redistributed proportionally from baseline",
    }
    return results


def run_ablation(demand, trends, future_weights, coverage, top_n: int = 20) -> dict:
    """Run ablation study: remove one signal at a time."""
    variants = {
        "full": {"use_trend": True, "use_future": True, "use_coverage": False},
        "no_trend": {"use_trend": False, "use_future": True, "use_coverage": False},
        "no_future": {"use_trend": True, "use_future": False, "use_coverage": False},
        "with_coverage": {"use_trend": True, "use_future": True, "use_coverage": True},
        "demand_only": {"use_trend": False, "use_future": False, "use_coverage": False},
    }

    results = {}
    for name, flags in variants.items():
        recs = compute_priority_scores(demand, trends, future_weights, coverage, **flags)
        top_skills = recs.head(top_n)["skill"].tolist() if not recs.empty else []
        results[name] = {
            "top_skills": top_skills,
            "n_total": len(recs),
        }

    # Compute Jaccard overlaps between full and ablated
    full_set = set(results["full"]["top_skills"])
    for name in results:
        variant_set = set(results[name]["top_skills"])
        union = full_set | variant_set
        inter = full_set & variant_set
        results[name]["jaccard_vs_full"] = round(len(inter) / len(union), 4) if union else 1.0

    return results


def run_weight_sensitivity(
    demand: pd.DataFrame,
    trends: pd.DataFrame,
    future_weights: pd.DataFrame,
    coverage: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """
    Sweep weight configurations; compare top-20 to baseline via Jaccard.
    Baseline: w_demand=0.4, w_trend=0.3, w_future=0.3.
    """
    baseline_weights = (0.40, 0.30, 0.30)
    recs_baseline = compute_priority_scores(
        demand, trends, future_weights, coverage,
        w_demand=baseline_weights[0], w_trend=baseline_weights[1], w_future=baseline_weights[2],
    )
    baseline_top = set(recs_baseline.head(top_n)["skill"].tolist()) if not recs_baseline.empty else set()

    configs = [
        (0.3, 0.35, 0.35),
        (0.4, 0.30, 0.30),  # baseline
        (0.5, 0.25, 0.25),
        (0.3, 0.2, 0.5),
        (0.4, 0.2, 0.4),
        (0.5, 0.2, 0.3),
    ]

    results = {}
    for w_d, w_t, w_f in configs:
        key = f"d{w_d}_t{w_t}_f{w_f}"
        recs = compute_priority_scores(
            demand, trends, future_weights, coverage,
            w_demand=w_d, w_trend=w_t, w_future=w_f,
        )
        top_skills = recs.head(top_n)["skill"].tolist() if not recs.empty else []
        top_set = set(top_skills)
        inter = len(baseline_top & top_set)
        union = len(baseline_top | top_set)
        jaccard = round(inter / union, 4) if union else 1.0
        results[key] = {
            "weights": {"demand": w_d, "trend": w_t, "future": w_f},
            "jaccard_vs_baseline": jaccard,
            "top_skills": top_skills[:20],
        }
    return {
        "baseline_weights": {"demand": 0.4, "trend": 0.3, "future": 0.3},
        "configs": results,
    }


def compute_demand_only_ranking(demand: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Demand-only frequency baseline ranking for P@20 floor comparison.

    Uses only demand_freq (normalized) with weight=1.0. This is the simplest
    possible baseline: rank skills by how often they appear in job postings.
    A full formula P@20 that barely beats this baseline would indicate the
    trend and future signals add little value.
    """
    if demand.empty:
        return pd.DataFrame()
    rec = demand.copy()
    max_freq = rec["demand_freq"].max()
    rec["demand_norm"] = rec["demand_freq"] / max_freq if max_freq > 0 else 0.0
    rec["priority_score_baseline"] = rec["demand_norm"]
    rec = rec.sort_values("priority_score_baseline", ascending=False).reset_index(drop=True)
    rec.insert(0, "baseline_rank", range(1, len(rec) + 1))
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ranked curriculum recommendations with evidence traces."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--top_n", type=int, default=20,
                        help="Top-N recommendations to highlight (default: 20)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (remove signals one at a time)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run weight sensitivity analysis (sweep demand/trend/future weights)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate against expert labels in recommendations.csv")
    parser.add_argument("--baseline", action="store_true",
                        help="Generate demand-only frequency baseline ranking and compare "
                             "P@20/NDCG@20 against full formula (floor comparison for RQ5)")
    parser.add_argument("--coverage-ablation", action="store_true",
                        help="Item 6: Run coverage weight ablation (0, 0.10, 0.20, 0.30) and "
                             "report Jaccard overlap to empirically validate the 0.0 default")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skills_df = load_skills(out_dir)
    trends = load_trends(out_dir)
    future_weights = load_future_weights(out_dir)
    coverage = load_coverage(out_dir)

    if skills_df.empty:
        print("[ERROR] No skills data found. Run the pipeline first.")
        return

    demand = build_skill_demand(skills_df)
    print(f"[INFO] Built demand table: {len(demand)} unique skills")

    # Exclude vague single-word hard skills (e.g. "design", "develop") — too ambiguous for curriculum
    vague_mask = demand.apply(
        lambda r: _is_vague_hard_skill(str(r.get("skill", "")), str(r.get("skill_type", ""))),
        axis=1,
    )
    if vague_mask.any():
        n_vague = vague_mask.sum()
        demand = demand[~vague_mask].reset_index(drop=True)
        print(f"[INFO] Excluded {n_vague} vague single-word hard skills from curriculum priorities")

    if not trends.empty:
        print(f"[INFO] Loaded {len(trends)} trend records")
    if not future_weights.empty:
        print(f"[INFO] Loaded {len(future_weights)} future weight records")
    if not coverage.empty:
        print(f"[INFO] Loaded {len(coverage)} coverage records")

    recs = compute_priority_scores(demand, trends, future_weights, coverage)
    recs["expert_priority"] = ""

    rec_path = out_dir / "recommendations.csv"
    recs.to_csv(rec_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved {len(recs)} recommendations to {rec_path}")

    top = recs.head(args.top_n)
    print(f"\n[INFO] Top-{args.top_n} curriculum priorities:")
    for _, r in top.iterrows():
        print(f"  #{int(r['rank']):3d}  {r['skill'][:50]:50s}  "
              f"priority={r['priority_score']:.3f}  "
              f"demand={int(r['demand_freq'])}  "
              f"trend={r['trend_label']}  "
              f"future={r['future_weight']:.2f}")

    report = {
        "total_skills": len(recs),
        "top_n": args.top_n,
        "top_recommendations": top[[
            "rank", "skill", "priority_score", "demand_freq",
            "trend_label", "trend_q_value", "future_weight",
            "future_domain", "mapping_margin", "coverage_gap",
        ]].to_dict(orient="records"),
    }

    if args.ablation:
        print("\n[INFO] Running ablation study...")
        ablation = run_ablation(demand, trends, future_weights, coverage,
                                top_n=args.top_n)
        report["ablation"] = ablation
        for name, data in ablation.items():
            print(f"  {name}: Jaccard vs full = {data['jaccard_vs_full']:.3f}")

    if args.sensitivity:
        print("\n[INFO] Running weight sensitivity analysis...")
        sensitivity = run_weight_sensitivity(demand, trends, future_weights, coverage, top_n=args.top_n)
        report["weight_sensitivity"] = sensitivity
        sens_path = out_dir / "weight_sensitivity_report.json"
        sens_path.write_text(json.dumps(sensitivity, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Saved weight sensitivity report to {sens_path}")
        for key, data in sensitivity["configs"].items():
            print(f"  {key}: Jaccard vs baseline = {data['jaccard_vs_baseline']:.3f}")

    if args.evaluate:
        existing = out_dir / "recommendations.csv"
        if existing.exists():
            eval_df = pd.read_csv(existing)
            eval_result = evaluate_recommendations(eval_df, top_n=args.top_n)
            report["evaluation"] = eval_result
            if eval_result.get("status") == "ok":
                print(f"\n[INFO] Evaluation: P@{args.top_n}={eval_result['precision_at_n']:.3f}, "
                      f"NDCG@{args.top_n}={eval_result['ndcg_at_n']:.3f}")
                if "irr" in eval_result:
                    irr = eval_result["irr"]
                    print(f"  Recommendation IRR: kappa={irr.get('cohens_kappa', 'N/A')}")
            else:
                print(f"[INFO] Evaluation: {eval_result.get('status', 'unknown')}")

    if args.baseline:
        print("\n[INFO] Computing demand-only frequency baseline...")
        baseline_recs = compute_demand_only_ranking(demand, top_n=args.top_n)
        full_set = set(recs.head(args.top_n)["skill"].tolist())
        base_set = set(baseline_recs.head(args.top_n)["skill"].tolist()) if not baseline_recs.empty else set()
        union_b = full_set | base_set
        inter_b = full_set & base_set
        jaccard_b = round(len(inter_b) / len(union_b), 4) if union_b else 1.0
        baseline_report = {
            "top_skills_baseline": sorted(base_set),
            "jaccard_full_vs_baseline": jaccard_b,
            "baseline_p_at_n": None,
            "baseline_ndcg_at_n": None,
            "full_p_at_n": report.get("evaluation", {}).get("precision_at_n"),
            "full_ndcg_at_n": report.get("evaluation", {}).get("ndcg_at_n"),
            "interpretation": (
                f"Jaccard={jaccard_b:.3f} between full-formula and demand-only top-{args.top_n}. "
                "Low Jaccard means trend+future signals meaningfully change the ranking."
            ),
        }
        if args.evaluate and not baseline_recs.empty:
            existing = out_dir / "recommendations.csv"
            if existing.exists():
                full_df = pd.read_csv(existing)
                if "expert_priority" in full_df.columns:
                    label_map = (
                        full_df[["skill", "expert_priority"]]
                        .drop_duplicates("skill")
                        .set_index("skill")["expert_priority"]
                        .to_dict()
                    )
                    baseline_top = baseline_recs.head(args.top_n).copy()
                    baseline_top["expert_priority"] = baseline_top["skill"].map(label_map).fillna("")
                    base_eval = evaluate_recommendations(baseline_top, top_n=args.top_n)
                    if base_eval.get("status") == "ok":
                        baseline_report["baseline_p_at_n"] = base_eval["precision_at_n"]
                        baseline_report["baseline_ndcg_at_n"] = base_eval["ndcg_at_n"]
                        print(f"  Baseline P@{args.top_n}={base_eval['precision_at_n']:.3f} "
                              f"vs Full P@{args.top_n}="
                              f"{baseline_report.get('full_p_at_n', 'N/A')}")
        report["baseline_comparison"] = baseline_report
        if not baseline_recs.empty:
            baseline_path = out_dir / "recommendations_baseline.csv"
            baseline_recs.head(args.top_n).to_csv(baseline_path, index=False, encoding="utf-8-sig")
            print(f"[INFO] Saved baseline ranking to {baseline_path}")
        print(f"[INFO] Full vs baseline Jaccard: {jaccard_b:.3f}")

    # Item 6: Coverage ablation
    if args.coverage_ablation:
        print("\n[INFO] Running coverage weight ablation (Item 6)…")
        cov_result = run_coverage_ablation(demand, trends, future_weights, coverage, top_n=args.top_n)
        report["coverage_ablation"] = cov_result
        j = cov_result.get("coverage_0.10", {}).get("jaccard_vs_no_coverage", "N/A")
        print(f"  Jaccard(no_coverage vs coverage@0.10): {j}")
        print(f"  Conclusion: {cov_result.get('_interpretation', {}).get('conclusion', '')}")
        ablation_path = out_dir / "coverage_ablation_report.json"
        ablation_path.write_text(
            json.dumps(cov_result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Saved → {ablation_path}")

    report_path = out_dir / "recommendations_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[INFO] Saved report to {report_path}")


if __name__ == "__main__":
    main()
