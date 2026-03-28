"""
evaluate_extraction.py

Evaluates extraction **precision** using a labeled gold set.

NOTE ON RECALL: True recall is not estimable with this gold-set design.
The gold set contains items sampled from pipeline *outputs*, so we can
measure "what fraction of extracted items are correct" (precision) but NOT
"what fraction of true items were extracted" (recall).  Any metric labelled
recall_estimate in older reports is numerically identical to precision and
should be interpreted as such.  F1 is therefore omitted.

Scientific methods (see SCIENTIFIC_METHODOLOGY.md):
    - Binomial test: H0 precision=0.5 vs H1 precision>0.5
    - Effect sizes: odds ratio, risk difference
    - Wilson score CI for precision
    - Two-proportion z-test for pairwise source comparison
    - Bonferroni correction (per-source); Benjamini-Hochberg (pairwise)

Inputs:
    DATA/labels/gold_skills.csv   (with is_correct filled in)
    DATA/labels/gold_knowledge.csv (with is_correct filled in)

Outputs:
    results/extraction_evaluation_report.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy import stats

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def _load_labeled(path: Path, id_col: str = "gold_id") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "is_correct" not in df.columns:
        return pd.DataFrame()
    df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
    df = df[df["is_correct"].isin(["yes", "no"])].copy()
    df["is_correct_bin"] = (df["is_correct"] == "yes").astype(int)
    return df


def compute_precision(y_true: List[int]) -> Dict[str, float]:
    """Compute extraction precision from gold-set labels.

    precision = (items labeled correct) / (total labeled items).

    True recall is NOT estimable: the gold set samples from pipeline
    outputs, so we cannot know how many real skills were *missed*.
    """
    n = len(y_true)
    if n == 0:
        return {"n": 0, "n_correct": 0, "n_incorrect": 0, "precision": 0.0}

    tp = sum(y_true)
    fp = n - tp
    precision = tp / n

    return {
        "n": n,
        "n_correct": tp,
        "n_incorrect": fp,
        "precision": round(precision, 4),
    }


def _binom_test_precision(n_correct: int, n: int, p0: float = 0.5) -> Tuple[float, bool]:
    """Test H0: precision = p0 vs H1: precision > p0. Returns (p_value, significant_at_005)."""
    if n == 0:
        return (1.0, False)
    try:
        bt = stats.binomtest(n_correct, n, p=p0, alternative="greater")
        return (float(bt.pvalue), bool(bt.pvalue < 0.05))
    except Exception:
        return (1.0, False)


def _effect_sizes(precision: float, p0: float = 0.5) -> Dict[str, float]:
    """Odds ratio vs chance and risk difference. For p0=0.5, odds_ratio = p/(1-p)."""
    risk_diff = round(precision - p0, 4)
    if precision <= 0:
        return {"odds_ratio": 0.0, "risk_difference": risk_diff}
    if precision >= 1:
        return {"odds_ratio": 999.0, "risk_difference": risk_diff}
    odds = precision / (1 - precision)
    odds_chance = p0 / (1 - p0) if p0 < 1 else 0
    odds_ratio = odds / odds_chance if odds_chance > 0 else odds
    return {"odds_ratio": round(min(999.0, odds_ratio), 4), "risk_difference": risk_diff}


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (round(max(0, lo), 4), round(min(1, hi), 4))


def _add_binomial_and_effect(metrics: Dict, p0: float = 0.5) -> None:
    """Add binomial test and effect sizes to a precision metrics dict."""
    n_correct = metrics.get("n_correct", 0)
    n = metrics.get("n", 0)
    precision = metrics.get("precision", 0.0)
    p_val, sig = _binom_test_precision(n_correct, n, p0)
    metrics["p_value_vs_chance"] = round(p_val, 6)
    metrics["significant_at_005"] = bool(sig)
    metrics.update(_effect_sizes(precision, p0))


def _two_proportion_z_test(n1: int, x1: int, n2: int, x2: int) -> float:
    """Two-proportion z-test (unpaired). Returns two-tailed p-value."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    if p_pool <= 0 or p_pool >= 1:
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0 if p1 != p2 else 1.0
    z = (p1 - p2) / se
    # Two-tailed: 2 * (1 - Phi(|z|)); Phi(x) = 0.5*(1+erf(x/sqrt(2)))
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _benjamini_hochberg(p_values: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    for r, i in enumerate(order, start=1):
        adjusted[i] = min(1.0, p_values[i] * n / r)
    return adjusted


def _pairwise_comparisons(results: List[Dict]) -> List[Dict]:
    """Pairwise two-proportion z-tests between sources; BH correction."""
    by_source = {r["source"]: r for r in results if r.get("source") != "all"}
    sources = list(by_source.keys())
    if len(sources) < 2:
        return []

    pairs = []
    p_vals = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            a, b = sources[i], sources[j]
            ra, rb = by_source[a], by_source[b]
            n1, x1 = ra["n"], ra["n_correct"]
            n2, x2 = rb["n"], rb["n_correct"]
            p = _two_proportion_z_test(n1, x1, n2, x2)
            odds_a = (ra["precision"] / (1 - ra["precision"])) if ra["precision"] < 1 else 999
            odds_b = (rb["precision"] / (1 - rb["precision"])) if rb["precision"] < 1 else 999
            odds_ratio = odds_a / odds_b if odds_b > 0 else 0
            pairs.append({"A": a, "B": b, "p_value": p, "odds_ratio": round(odds_ratio, 4)})
            p_vals.append(p)

    adj = _benjamini_hochberg(p_vals)
    for i, p in enumerate(pairs):
        p["p_value"] = round(p["p_value"], 6)
        p["p_adjusted"] = round(adj[i], 6)
        p["significant_after_correction"] = bool(adj[i] < 0.05)
    return pairs


def _hybrid_vs_llmonly_test(
    hybrid_n: int, hybrid_correct: int,
    llmonly_n: int, llmonly_correct: int,
) -> Dict:
    """Primary RQ1 test: Hybrid vs LLM-only precision (one-tailed two-proportion z-test).

    H0: precision_hybrid = precision_llmonly
    H1: precision_hybrid > precision_llmonly

    This is the primary RQ1 test.  The per-source binomial tests vs chance are
    secondary.  Requires LLM-only gold set labels from a separate ablation run
    (pipeline.py --extraction-mode llm_only).
    """
    if hybrid_n == 0 or llmonly_n == 0:
        return {"status": "insufficient_data"}

    p_hybrid = hybrid_correct / hybrid_n
    p_llmonly = llmonly_correct / llmonly_n
    p_pool = (hybrid_correct + llmonly_correct) / (hybrid_n + llmonly_n)

    if p_pool <= 0 or p_pool >= 1:
        return {"status": "degenerate"}

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / hybrid_n + 1 / llmonly_n))
    z = (p_hybrid - p_llmonly) / se if se > 0 else 0.0
    p_one_tailed = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

    ci_hybrid = wilson_ci(p_hybrid, hybrid_n)
    ci_llmonly = wilson_ci(p_llmonly, llmonly_n)

    return {
        "status": "ok",
        "hypothesis": "H1: precision_hybrid > precision_llmonly (one-tailed)",
        "hybrid_precision": round(p_hybrid, 4),
        "hybrid_precision_ci_95": list(ci_hybrid),
        "hybrid_n": hybrid_n,
        "llmonly_precision": round(p_llmonly, 4),
        "llmonly_precision_ci_95": list(ci_llmonly),
        "llmonly_n": llmonly_n,
        "precision_difference": round(p_hybrid - p_llmonly, 4),
        "z_statistic": round(z, 4),
        "p_value_one_tailed": round(float(p_one_tailed), 6),
        "significant_at_005": bool(p_one_tailed < 0.05),
        "interpretation": (
            "Hybrid significantly outperforms LLM-only" if p_one_tailed < 0.05
            else "No significant improvement of Hybrid over LLM-only"
        ),
    }


def _bert_contribution_analysis(df: pd.DataFrame) -> Dict:
    """Explain BERT's recall contribution when BERT standalone precision < 0.5.

    When BERT precision is below chance, BERT may still contribute recall:
    items that BERT found but LLM missed.  This function quantifies that
    contribution to support the hybrid design rationale.

    Requires columns: source, is_correct_bin.
    """
    if "source" not in df.columns:
        return {"status": "no_source_column"}

    bert_mask = df["source"].str.contains("BERT", case=False, na=False)
    llm_mask = df["source"].str.contains("LLM", case=False, na=False)
    bert_df = df[bert_mask].copy()

    if bert_df.empty:
        return {"status": "no_bert_items"}

    # Items unique to BERT: source contains BERT but NOT LLM (i.e. not fused BERT+LLM)
    bert_only_mask = bert_mask & ~llm_mask
    bert_only_df = df[bert_only_mask].copy()

    n_bert = len(bert_df)
    n_bert_only = len(bert_only_df)
    n_llm = int(llm_mask.sum())
    pct_bert_only = round(n_bert_only / n_bert, 4) if n_bert > 0 else 0.0

    bert_precision = float(bert_df["is_correct_bin"].mean()) if n_bert > 0 else 0.0
    bert_only_precision = float(bert_only_df["is_correct_bin"].mean()) if n_bert_only > 0 else 0.0

    return {
        "status": "ok",
        "n_bert_total": n_bert,
        "n_bert_only_unique": n_bert_only,
        "n_llm_total": n_llm,
        "pct_bert_items_unique_to_bert": pct_bert_only,
        "bert_overall_precision": round(bert_precision, 4),
        "bert_only_precision": round(bert_only_precision, 4),
        "interpretation": (
            f"Of {n_bert} BERT-extracted items, {n_bert_only} ({pct_bert_only:.1%}) are "
            f"unique to BERT (not found by LLM). These unique items have precision="
            f"{bert_only_precision:.3f}. Even if BERT overall precision is below chance, "
            f"these unique items represent potential recall contribution: skills the LLM "
            f"would have missed entirely."
        ),
    }


def _apply_multi_comparison_correction(results: List[Dict], alpha: float = 0.05) -> None:
    """Apply Bonferroni correction: alpha_adjusted = alpha / k for k tests vs chance."""
    by_source = [r for r in results if r.get("source") != "all"]
    if not by_source:
        return
    k = len(by_source)
    alpha_adj = alpha / k
    for r in by_source:
        p = r.get("p_value_vs_chance", 1.0)
        r["p_value_adjusted"] = round(min(1.0, p * k), 6)  # Bonferroni
        r["significant_after_correction"] = bool(p < alpha_adj)


def evaluate_by_source(df: pd.DataFrame, source_col: str = "source") -> List[Dict]:
    results = []
    if source_col not in df.columns:
        metrics = compute_precision(df["is_correct_bin"].tolist())
        metrics["source"] = "all"
        ci = wilson_ci(metrics["precision"], metrics["n"])
        metrics["precision_ci_95"] = list(ci)
        _add_binomial_and_effect(metrics)
        results.append(metrics)
        return results

    for src, grp in df.groupby(source_col):
        metrics = compute_precision(grp["is_correct_bin"].tolist())
        metrics["source"] = str(src)
        ci = wilson_ci(metrics["precision"], metrics["n"])
        metrics["precision_ci_95"] = list(ci)
        _add_binomial_and_effect(metrics)
        results.append(metrics)

    overall = compute_precision(df["is_correct_bin"].tolist())
    overall["source"] = "all"
    ci = wilson_ci(overall["precision"], overall["n"])
    overall["precision_ci_95"] = list(ci)
    _add_binomial_and_effect(overall)
    results.append(overall)

    _apply_multi_comparison_correction(results)
    return results


def _fleiss_kappa(matrix) -> float:
    """Compute Fleiss' Kappa for 3+ raters on binary categories.

    ``matrix`` is (N x C) where N = items and C = categories (e.g. 2 for
    binary).  Each cell is the count of raters who placed item i into
    category c.
    """
    N, C = matrix.shape
    n_raters = matrix[0].sum()
    if N == 0 or n_raters <= 1:
        return 0.0

    p_j = matrix.sum(axis=0) / (N * n_raters)
    P_i = (matrix ** 2).sum(axis=1)
    P_i = (P_i - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()

    if P_e >= 1.0:
        return 0.0
    return float((P_bar - P_e) / (1 - P_e))


def _load_irr_from_gold_labels(labels_dir: Path, kind: str) -> Dict:
    """Load IRR from gold_labels/ (multi-reviewer UI format).

    Overlap items are identified by the ``is_overlap`` column in the gold
    template (randomly sampled at export time).  Falls back to first-N
    if the column is absent.
    """
    path = labels_dir / "gold_labels" / f"{kind}_labels.csv"
    if not path.exists():
        return {"status": "no_gold_labels"}
    df = pd.read_csv(path)
    if "gold_id" not in df.columns or "labeler_id" not in df.columns or "is_correct" not in df.columns:
        return {"status": "missing_columns"}
    df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
    df = df[df["is_correct"].isin(["yes", "no"])].copy()
    df["is_correct_bin"] = (df["is_correct"] == "yes").astype(int)

    template_path = labels_dir / ("gold_skills.csv" if kind == "skill" else "gold_knowledge.csv")
    if template_path.exists():
        tpl = pd.read_csv(template_path)
        if "is_overlap" in tpl.columns and "gold_id" in tpl.columns:
            overlap_ids = tpl.loc[
                tpl["is_overlap"].astype(str).str.strip().str.lower().isin(["true", "1"]),
                "gold_id",
            ].tolist()
        elif "gold_id" in tpl.columns:
            overlap_ids = tpl["gold_id"].head(30).tolist()
        else:
            overlap_ids = []
    else:
        overlap_ids = df["gold_id"].unique()[:30].tolist()

    df_overlap = df[df["gold_id"].isin(overlap_ids)].copy()
    labelers = df_overlap["labeler_id"].astype(str).str.strip().unique()
    labelers = [x for x in labelers if x and x != ""]
    if len(labelers) < 2:
        return {"status": "single_labeler", "labeler": str(labelers[0]) if len(labelers) == 1 else "none"}

    pivot = df_overlap.pivot_table(
        index="gold_id", columns="labeler_id", values="is_correct_bin", aggfunc="first"
    ).dropna(how="all")
    valid_labelers = [c for c in labelers if c in pivot.columns]
    if len(valid_labelers) < 2:
        return {"status": "insufficient_overlap", "overlap_items": len(pivot)}
    pivot = pivot[valid_labelers].dropna()
    if len(pivot) < 5:
        return {"status": "insufficient_overlap", "overlap_items": len(pivot)}

    result: Dict = {
        "status": "ok",
        "labelers": [str(l) for l in valid_labelers],
        "overlap_items": len(pivot),
    }

    if len(valid_labelers) == 2:
        l1, l2 = valid_labelers[0], valid_labelers[1]
        a, b = pivot[l1].values, pivot[l2].values
        agree = (a == b).sum()
        n = len(a)
        po = agree / n
        pe = ((a == 1).sum() / n * (b == 1).sum() / n +
              (a == 0).sum() / n * (b == 0).sum() / n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        result["observed_agreement"] = round(po, 4)
        result["cohens_kappa"] = round(kappa, 4)
    else:
        import numpy as np
        mat_values = pivot[valid_labelers].values
        n_items = mat_values.shape[0]
        count_matrix = np.zeros((n_items, 2))
        for i in range(n_items):
            row = mat_values[i]
            count_matrix[i, 1] = row.sum()
            count_matrix[i, 0] = len(valid_labelers) - row.sum()
        fk = _fleiss_kappa(count_matrix)
        result["fleiss_kappa"] = round(fk, 4)
        result["n_raters"] = len(valid_labelers)

        l1, l2 = valid_labelers[0], valid_labelers[1]
        a, b = pivot[l1].values, pivot[l2].values
        agree = (a == b).sum()
        n = len(a)
        po = agree / n
        pe = ((a == 1).sum() / n * (b == 1).sum() / n +
              (a == 0).sum() / n * (b == 0).sum() / n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        result["observed_agreement_pair"] = round(po, 4)
        result["cohens_kappa_pair"] = round(kappa, 4)

    return result


def evaluate_irr(df: pd.DataFrame, id_col: str = "gold_id") -> Dict:
    """Compute inter-rater reliability if multiple labelers present.

    Uses Cohen's Kappa for exactly 2 raters and Fleiss' Kappa when 3 or
    more raters are present.
    """
    import numpy as np

    if "labeler_id" not in df.columns:
        return {"status": "no_labeler_id"}
    df_filled = df[df["labeler_id"].astype(str).str.strip() != ""].copy()
    labelers = list(df_filled["labeler_id"].unique())
    if len(labelers) < 2:
        return {"status": "single_labeler", "labeler": str(labelers[0]) if len(labelers) == 1 else "none"}

    pivot = df_filled.pivot_table(
        index=id_col, columns="labeler_id",
        values="is_correct_bin", aggfunc="first",
    ).dropna()

    if len(pivot) < 5:
        return {"status": "insufficient_overlap", "overlap_items": len(pivot)}

    valid_labelers = [l for l in labelers if l in pivot.columns]
    if len(valid_labelers) < 2:
        return {"status": "labelers_not_in_overlap"}

    result: Dict = {
        "status": "ok",
        "labelers": [str(l) for l in valid_labelers],
        "overlap_items": len(pivot),
    }

    if len(valid_labelers) == 2:
        l1, l2 = valid_labelers[0], valid_labelers[1]
        a = pivot[l1].values
        b = pivot[l2].values
        agree = (a == b).sum()
        n = len(a)
        po = agree / n
        pe = ((a == 1).sum() / n * (b == 1).sum() / n +
              (a == 0).sum() / n * (b == 0).sum() / n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        result["observed_agreement"] = round(po, 4)
        result["cohens_kappa"] = round(kappa, 4)
    else:
        mat_values = pivot[valid_labelers].values
        n_items = mat_values.shape[0]
        count_matrix = np.zeros((n_items, 2))
        for i in range(n_items):
            row = mat_values[i]
            count_matrix[i, 1] = row.sum()
            count_matrix[i, 0] = len(valid_labelers) - row.sum()
        fk = _fleiss_kappa(count_matrix)
        result["fleiss_kappa"] = round(fk, 4)
        result["n_raters"] = len(valid_labelers)

        l1, l2 = valid_labelers[0], valid_labelers[1]
        a = pivot[l1].values
        b = pivot[l2].values
        agree = (a == b).sum()
        n = len(a)
        po = agree / n
        pe = ((a == 1).sum() / n * (b == 1).sum() / n +
              (a == 0).sum() / n * (b == 0).sum() / n)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
        result["observed_agreement_pair"] = round(po, 4)
        result["cohens_kappa_pair"] = round(kappa, 4)

    return result


def evaluate_bloom_classification(df: pd.DataFrame) -> Dict:
    """Validate pipeline Bloom classification against gold-labeled Bloom levels.

    Requires both ``bloom`` (pipeline prediction) and ``bloom_label`` (gold
    label) columns to be present and non-empty.  Returns overall accuracy
    and per-level precision.
    """
    if "bloom" not in df.columns or "bloom_label" not in df.columns:
        return {"status": "missing_columns"}

    sub = df.copy()
    sub["bloom"] = sub["bloom"].astype(str).str.strip()
    sub["bloom_label"] = sub["bloom_label"].astype(str).str.strip()
    sub = sub[(sub["bloom_label"] != "") & (sub["bloom_label"].str.lower() != "nan")].copy()

    if len(sub) < 5:
        return {"status": "insufficient_data", "n": len(sub)}

    sub["match"] = sub["bloom"].str.lower() == sub["bloom_label"].str.lower()
    n = len(sub)
    n_correct = int(sub["match"].sum())
    accuracy = round(n_correct / n, 4)
    ci = wilson_ci(accuracy, n)

    per_level: List[Dict] = []
    for level, grp in sub.groupby(sub["bloom_label"].str.lower()):
        lvl_n = len(grp)
        lvl_correct = int(grp["match"].sum())
        per_level.append({
            "bloom_level": str(level),
            "n": lvl_n,
            "correct": lvl_correct,
            "accuracy": round(lvl_correct / lvl_n, 4) if lvl_n > 0 else 0.0,
        })

    return {
        "status": "ok",
        "n": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "accuracy_ci_95": list(ci),
        "per_level": sorted(per_level, key=lambda x: x["bloom_level"]),
    }


def evaluate_type_classification(df: pd.DataFrame) -> Dict:
    """Validate pipeline type (hard/soft) classification against gold labels."""
    if "type" not in df.columns or "type_label" not in df.columns:
        return {"status": "missing_columns"}

    sub = df.copy()
    sub["type"] = sub["type"].astype(str).str.strip().str.lower()
    sub["type_label"] = sub["type_label"].astype(str).str.strip().str.lower()
    sub = sub[(sub["type_label"] != "") & (sub["type_label"].str.lower() != "nan")].copy()

    if len(sub) < 5:
        return {"status": "insufficient_data", "n": len(sub)}

    sub["match"] = sub["type"] == sub["type_label"]
    n = len(sub)
    n_correct = int(sub["match"].sum())
    accuracy = round(n_correct / n, 4)
    ci = wilson_ci(accuracy, n)

    return {
        "status": "ok",
        "n": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "accuracy_ci_95": list(ci),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extraction quality using gold-set labels."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR))
    parser.add_argument("--output", type=str, default="extraction_evaluation_report.json")
    parser.add_argument("--spektrum-code", type=str, default=None,
                        help="Spektrum Keahlian code for provenance / per-Bidang stratification")
    parser.add_argument("--llmonly-labels-dir", type=str, default=None,
                        help="Path to DATA/labels for the LLM-only ablation run "
                             "(pipeline.py --extraction-mode llm_only). Required for "
                             "the primary RQ1 test (Hybrid vs LLM-only).")
    args = parser.parse_args()

    labels = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: Dict = {}

    # Prefer merged file (multi-reviewer majority vote); fallback to direct gold_skills.csv
    skills_path = labels / "gold_skills_merged.csv"
    if not skills_path.exists():
        skills_path = labels / "gold_skills.csv"
    skills_df = _load_labeled(skills_path)
    if skills_df.empty:
        print("[WARN] No labeled skills found (gold_skills.csv empty or missing is_correct)")
        report["skills"] = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(skills_df)} labeled skills")
        irr = _load_irr_from_gold_labels(labels, "skill") if (labels / "gold_labels" / "skill_labels.csv").exists() else evaluate_irr(skills_df)
        by_src = evaluate_by_source(skills_df)
        bloom_eval = evaluate_bloom_classification(skills_df)
        type_eval = evaluate_type_classification(skills_df)
        # RQ1 primary test: Hybrid vs LLM-only (requires --llmonly-labels-dir)
        rq1_primary = {"status": "no_llmonly_labels",
                       "note": "Run with --llmonly-labels-dir to enable the primary RQ1 test."}
        if args.llmonly_labels_dir:
            lo_labels = Path(args.llmonly_labels_dir)
            lo_path = lo_labels / "gold_skills_merged.csv"
            if not lo_path.exists():
                lo_path = lo_labels / "gold_skills.csv"
            lo_df = _load_labeled(lo_path)
            if not lo_df.empty:
                hybrid_overall = next((r for r in by_src if r["source"] == "all"), None)
                lo_metrics = compute_precision(lo_df["is_correct_bin"].tolist())
                if hybrid_overall:
                    rq1_primary = _hybrid_vs_llmonly_test(
                        hybrid_overall["n"], hybrid_overall["n_correct"],
                        lo_metrics["n"], lo_metrics["n_correct"],
                    )
                print(f"  RQ1 primary: Hybrid P={hybrid_overall['precision']:.3f} vs "
                      f"LLM-only P={lo_metrics['precision']:.3f} "
                      f"({rq1_primary.get('interpretation', '')})")
            else:
                print("[WARN] --llmonly-labels-dir specified but no labeled gold set found.")

        # BERT contribution analysis (run when BERT precision < 0.5)
        bert_contrib = {"status": "skipped"}
        bert_source_row = next((r for r in by_src if r.get("source", "").upper() == "BERT"), None)
        if bert_source_row and bert_source_row.get("precision", 1.0) < 0.5:
            bert_contrib = _bert_contribution_analysis(skills_df)
            if bert_contrib.get("status") == "ok":
                print(f"  BERT contribution: {bert_contrib['n_bert_only_unique']} items unique "
                      f"to BERT ({bert_contrib['pct_bert_items_unique_to_bert']:.1%}), "
                      f"precision={bert_contrib['bert_only_precision']:.3f}")

        report["skills"] = {
            "rq1_primary_hypothesis": rq1_primary,
            "by_source": by_src,
            "pairwise_comparisons": _pairwise_comparisons(by_src),
            "bert_contribution_analysis": bert_contrib,
            "irr": irr,
            "bloom_validation": bloom_eval,
            "type_validation": type_eval,
        }
        overall = [r for r in report["skills"]["by_source"] if r["source"] == "all"]
        if overall:
            o = overall[0]
            print(f"  Skills overall: P={o['precision']:.3f} "
                  f"(95% CI {o['precision_ci_95']}), n={o['n']}")
        if bloom_eval.get("status") == "ok":
            print(f"  Bloom classification accuracy: {bloom_eval['accuracy']:.3f} "
                  f"(n={bloom_eval['n']})")
        elif bloom_eval.get("status") == "insufficient_data":
            print(f"  Bloom validation: insufficient data (n={bloom_eval.get('n', 0)})")
        if type_eval.get("status") == "ok":
            print(f"  Type classification accuracy: {type_eval['accuracy']:.3f} "
                  f"(n={type_eval['n']})")

    knowledge_path = labels / "gold_knowledge_merged.csv"
    if not knowledge_path.exists():
        knowledge_path = labels / "gold_knowledge.csv"
    knowledge_df = _load_labeled(knowledge_path)
    if knowledge_df.empty:
        print("[WARN] No labeled knowledge found")
        report["knowledge"] = {"status": "no_data"}
    else:
        print(f"[INFO] Loaded {len(knowledge_df)} labeled knowledge")
        irr = _load_irr_from_gold_labels(labels, "knowledge") if (labels / "gold_labels" / "knowledge_labels.csv").exists() else evaluate_irr(knowledge_df)
        by_src = evaluate_by_source(knowledge_df)
        report["knowledge"] = {
            "by_source": by_src,
            "pairwise_comparisons": _pairwise_comparisons(by_src),
            "irr": irr,
        }
        overall = [r for r in report["knowledge"]["by_source"] if r["source"] == "all"]
        if overall:
            o = overall[0]
            print(f"  Knowledge overall: P={o['precision']:.3f} "
                  f"(95% CI {o['precision_ci_95']}), n={o['n']}")

    if args.spektrum_code and str(args.spektrum_code).strip():
        report["spektrum_code"] = str(args.spektrum_code).strip()

    out_path = out_dir / args.output
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Saved extraction evaluation report to {out_path}")


if __name__ == "__main__":
    main()
