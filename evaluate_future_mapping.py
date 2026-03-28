"""
evaluate_future_mapping.py

Validates future-domain mapping accuracy against expert-labeled gold set.

Inputs:
    DATA/labels/gold_future_domain.csv  (with true_domain_id filled in)
    results/future_skill_weights.csv or future_skill_weights_dummy.csv

Outputs:
    results/future_mapping_evaluation_report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy import stats

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0, center - spread), min(1, center + spread))


def load_gold(labels_dir: Path) -> pd.DataFrame:
    path = labels_dir / "gold_future_domain_merged.csv"
    if not path.exists():
        path = labels_dir / "gold_future_domain.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "true_domain_id" not in df.columns:
        return pd.DataFrame()
    df["true_domain_id"] = df["true_domain_id"].astype(str).str.strip().str.lower()
    df = df[~df["true_domain_id"].isin(["", "nan"])].copy()
    return df


def load_n_domains() -> int:
    """Load number of domains from future_domains.csv for random-chance baseline."""
    path = Path(config.PROJECT_ROOT) / "future_domains.csv"
    if not path.exists():
        path = Path(config.PROJECT_ROOT) / "future_domains_dummy.csv"
    if not path.exists():
        return 25  # fallback if files missing
    df = pd.read_csv(path)
    return len(df)


def load_pipeline_mapping(output_dir: Path) -> pd.DataFrame:
    for name in ["future_skill_weights.csv", "future_skill_weights_dummy.csv"]:
        path = output_dir / name
        if path.exists():
            df = pd.read_csv(path)
            if "best_domain_id" in df.columns:
                return df
    return pd.DataFrame()


def evaluate(
    gold: pd.DataFrame, mapping: pd.DataFrame, margin_threshold: float = 0.05, n_domains: int | None = None
) -> dict:
    if gold.empty:
        return {"status": "no_gold_data"}

    item_col = "item_text"
    if item_col not in gold.columns:
        return {"status": "missing_item_text_column"}

    gold_lower = gold.copy()
    gold_lower["_item_key"] = gold_lower[item_col].astype(str).str.strip().str.lower()

    results = []
    for _, row in gold_lower.iterrows():
        key = row["_item_key"]
        item_type = str(row.get("item_type", "")).strip().lower()
        true_id = row["true_domain_id"]

        col = "skill" if item_type == "skill" else "knowledge"
        if col not in mapping.columns:
            col = "skill" if "skill" in mapping.columns else "knowledge"

        match = mapping[mapping[col].astype(str).str.strip().str.lower() == key]
        if match.empty:
            results.append({
                "item": key, "true_domain": true_id,
                "predicted_domain": "", "correct": False,
                "is_none_or_unclear": True,  # Exclude from evaluable (no prediction to compare)
                "margin": 0.0, "status": "not_in_mapping",
            })
            continue

        m = match.iloc[0]
        pred_id = str(m.get("best_domain_id", "")).strip().lower()
        top2_id = str(m.get("top2_domain_id", "")).strip().lower()
        top3_id = str(m.get("top3_domain_id", "")).strip().lower()
        margin = float(m.get("mapping_margin", 0))

        is_correct = (true_id == pred_id)
        top3_preds = {p for p in (pred_id, top2_id, top3_id) if p}
        is_top3_correct = true_id in top3_preds if top3_preds else is_correct
        is_none = true_id in ("none", "unclear")

        results.append({
            "item": key, "true_domain": true_id,
            "predicted_domain": pred_id, "correct": is_correct,
            "top2_domain": top2_id, "top3_domain": top3_id,
            "top3_correct": is_top3_correct,
            "is_none_or_unclear": is_none, "margin": round(margin, 4),
        })

    results_df = pd.DataFrame(results)

    # Ensure boolean: pandas can store as float if mixed/NaN, and ~ fails on float
    mask = (results_df["is_none_or_unclear"] == True)
    evaluable = results_df[~mask].copy()
    n_eval = len(evaluable)
    n_correct = evaluable["correct"].sum() if n_eval > 0 else 0
    top1_acc = n_correct / n_eval if n_eval > 0 else 0.0

    n_top3_correct = evaluable["top3_correct"].sum() if n_eval > 0 and "top3_correct" in evaluable.columns else 0
    top3_acc = n_top3_correct / n_eval if n_eval > 0 else 0.0

    high_margin = evaluable[evaluable["margin"] >= margin_threshold] if n_eval > 0 else evaluable
    n_hm = len(high_margin)
    top1_acc_hm = high_margin["correct"].sum() / n_hm if n_hm > 0 else 0.0
    top3_acc_hm = high_margin["top3_correct"].sum() / n_hm if n_hm > 0 and "top3_correct" in high_margin.columns else 0.0

    none_rate = mask.sum() / len(results_df) if len(results_df) > 0 else 0.0

    # Wilson 95% CI and binomial test vs random chance (1/N_domains)
    if n_domains is None:
        n_domains = load_n_domains()
    p0 = 1.0 / n_domains

    top1_ci = wilson_ci(int(n_correct), n_eval)
    top3_ci = wilson_ci(int(n_top3_correct), n_eval)

    def _binom_pvalue(k: int, n: int, p: float) -> float:
        if n == 0:
            return 1.0
        try:
            bt = stats.binomtest(k, n, p=p, alternative="greater")
            return float(bt.pvalue)
        except Exception:
            return 1.0

    top1_pvalue = _binom_pvalue(int(n_correct), n_eval, p0)
    top3_pvalue = _binom_pvalue(int(n_top3_correct), n_eval, p0)

    # Mean Reciprocal Rank (MRR): gives rank-weighted credit to top-3 predictions
    reciprocal_ranks = []
    for _, r in evaluable.iterrows():
        true_d = r["true_domain"]
        pred_d = r["predicted_domain"]
        ranked = [pred_d]
        if "top2_domain" in r:
            ranked.append(r.get("top2_domain", ""))
        if "top3_domain" in r:
            ranked.append(r.get("top3_domain", ""))
        rr = 0.0
        for rank_pos, cand in enumerate(ranked, start=1):
            if cand == true_d:
                rr = 1.0 / rank_pos
                break
        reciprocal_ranks.append(rr)
    mrr = float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0

    confusion = Counter()
    incorrect_mask = (evaluable["correct"] != True)
    for _, r in evaluable[incorrect_mask].iterrows():
        pair = f"{r['true_domain']} -> {r['predicted_domain']}"
        confusion[pair] += 1

    return {
        "status": "ok",
        "total_items": len(results_df),
        "evaluable_items": n_eval,
        "top1_accuracy": round(top1_acc, 4),
        "top1_accuracy_ci": [round(top1_ci[0], 4), round(top1_ci[1], 4)],
        "top1_pvalue_vs_chance": round(top1_pvalue, 6),
        "top3_accuracy": round(top3_acc, 4),
        "top3_accuracy_ci": [round(top3_ci[0], 4), round(top3_ci[1], 4)],
        "top3_pvalue_vs_chance": round(top3_pvalue, 6),
        "mrr": round(mrr, 4),
        "n_domains_baseline": n_domains,
        "margin_threshold": margin_threshold,
        "high_margin_items": n_hm,
        "top1_accuracy_high_margin": round(top1_acc_hm, 4),
        "top3_accuracy_high_margin": round(top3_acc_hm, 4),
        "none_unclear_rate": round(none_rate, 4),
        "mean_margin": round(float(results_df["margin"].mean()), 4),
        "median_margin": round(float(results_df["margin"].median()), 4),
        "top_confusion_pairs": confusion.most_common(10),
        "per_item": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate future-domain mapping against gold labels."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR))
    parser.add_argument("--output", type=str, default="future_mapping_evaluation_report.json")
    parser.add_argument("--margin_threshold", type=float, default=0.05,
                        help="Margin threshold for high-margin accuracy subset (default: 0.05)")
    parser.add_argument("--spektrum-code", type=str, default=None,
                        help="Spektrum Keahlian code for provenance / per-Bidang stratification")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    labels = Path(args.labels_dir)

    gold = load_gold(labels)
    if gold.empty:
        print("[WARN] No gold future-domain labels found. "
              "Export and label gold_future_domain.csv first.")
        report = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(gold)} gold domain labels")
        mapping = load_pipeline_mapping(out_dir)
        if mapping.empty:
            print("[WARN] No pipeline mapping found (future_skill_weights*.csv)")
            report = {"status": "no_mapping"}
        else:
            print(f"[INFO] Loaded {len(mapping)} mapping entries")
            report = evaluate(gold, mapping, margin_threshold=args.margin_threshold)
            if report.get("status") == "ok":
                print(f"  Top-1 accuracy: {report['top1_accuracy']:.3f} "
                      f"(95% CI: [{report.get('top1_accuracy_ci', [0,0])[0]:.3f}, {report.get('top1_accuracy_ci', [0,0])[1]:.3f}], "
                      f"p vs chance: {report.get('top1_pvalue_vs_chance', 1):.4f})")
                print(f"  Top-3 accuracy: {report.get('top3_accuracy', 0):.3f} "
                      f"(95% CI: [{report.get('top3_accuracy_ci', [0,0])[0]:.3f}, {report.get('top3_accuracy_ci', [0,0])[1]:.3f}], "
                      f"p vs chance: {report.get('top3_pvalue_vs_chance', 1):.4f})")
                print(f"  MRR: {report.get('mrr', 0):.3f}")
                print(f"  Top-1 (high-margin): {report.get('top1_accuracy_high_margin', 0):.3f}")
                print(f"  None/unclear rate: {report['none_unclear_rate']:.3f}")
                print(f"  Mean margin: {report['mean_margin']:.3f}")

    if args.spektrum_code and str(args.spektrum_code).strip():
        report["spektrum_code"] = str(args.spektrum_code).strip()

    out_path = out_dir / args.output
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved report to {out_path}")


if __name__ == "__main__":
    main()
