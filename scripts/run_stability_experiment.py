"""
run_stability_experiment.py

Phase X — production-volume stability experiment.

Question: at what corpus size N does the cluster + competency set stabilize?
A paper-grade reproducibility study. For N ∈ {500, 1000, 2500, 5000, 10000}
and seeds ∈ {1, 2, 3}, run the full v2 pipeline (Phase 2.1 + 2.2 + 2.5) and
measure:

    - n_clusters
    - n_competencies (post-grounding-gate)
    - top-20 competency Jaccard between seeds at same N (stability metric)
    - mean grounding score
    - mean cluster cohesion

Output:
    results/stability_experiment/
        N_<N>_seed_<seed>/competencies.json, batch_reasonings.json
        summary.csv          — one row per (N, seed)
        jaccard_matrix.csv   — top-20 Jaccard between (seed_i, seed_j) per N
        stability_curve.png  — Jaccard top-20 vs N, ±std bars (the paper figure)

This driver consumes pre-computed Phase 1 output (advanced_skills.csv +
advanced_knowledge.csv) and SUBSAMPLES it at N items × 3 seeds. It does
NOT re-run Skill-LLM extraction for each N (that would require a fresh
Kaggle batch per N, infeasible). Subsampling the existing extraction is
a reasonable proxy for stability of the downstream clustering + generation.

Usage:
    python scripts/run_stability_experiment.py
        [--phase1-input results.old2]
        [--ns 500,1000,2500,5000]
        [--seeds 1,2,3]
        [--model gpt-5.4-mini]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_clustering_on_real_data import load_real_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("stability")


def _subsample_items(items, n: int, seed: int):
    """Random subsample without replacement."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    return rng.sample(list(items), n)


def _top_k_competency_titles(competencies, k: int = 20):
    """Return the top-K competency titles by future_weight desc as the
    'representative competency set' for stability comparison.
    Titles are normalized (lowercase, whitespace collapsed) for Jaccard.
    """
    import re
    sorted_c = sorted(
        competencies, key=lambda c: -float(c.get("future_weight", 0.0))
    )[:k]
    norm = [re.sub(r"\s+", " ", c.get("title", "").strip().lower()) for c in sorted_c]
    return [t for t in norm if t]


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-input", default="results.old2")
    parser.add_argument("--ns", default="500,1000,2500,5000",
                        help="Comma-separated item counts to evaluate")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--output-dir", default="results/stability_experiment")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Phase 2.2 LLM calls (clustering-only stability)")
    args = parser.parse_args()

    ns = [int(n.strip()) for n in args.ns.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    in_dir = PROJECT_ROOT / args.phase1_input
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading Phase 1 items from %s", in_dir)
    full_items = load_real_items(in_dir)
    log.info("loaded %d items total", len(full_items))

    from clustering import ClusteringConfig, cluster_skills
    from competency_v2_schema import GeneratorConfig
    from competency_generator_v2 import generate_competencies_v2

    rows = []
    titles_by_run: Dict[Tuple[int, int], List[str]] = {}

    for n in ns:
        if n > len(full_items):
            log.warning("N=%d exceeds available items (%d); using full set", n, len(full_items))
            n_actual = len(full_items)
        else:
            n_actual = n

        for seed in seeds:
            sub = _subsample_items(full_items, n_actual, seed)
            log.info("=== N=%d, seed=%d, items=%d ===", n, seed, len(sub))
            t0 = time.time()

            cl_cfg = ClusteringConfig(seed=seed)
            clusters, cl_report = cluster_skills(sub, config=cl_cfg)
            log.info("  clustering: %d clusters, mean cohesion %.3f",
                     len(clusters), cl_report.cohesion_mean)

            if args.skip_llm:
                # Quick mode — use clusters' summary_labels as proxy "competencies"
                comps_dicts = []
                titles = [c.summary_label for c in clusters if c.stream == "hard_plus_knowledge"]
                titles_by_run[(n, seed)] = sorted(titles, reverse=True)[:20]
                mean_grounding = 0.0
            else:
                g_cfg = GeneratorConfig(model=args.model)
                comps, brs = generate_competencies_v2(
                    clusters=clusters, config=g_cfg, apply_grounding_gate=True,
                )
                comps_dicts = [c.to_dict() for c in comps]
                # Save per-run
                run_dir = out_dir / f"N_{n}_seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "competencies.json").write_text(
                    json.dumps(comps_dicts, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                (run_dir / "batch_reasonings.json").write_text(
                    json.dumps([br.to_dict() for br in brs], indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                titles_by_run[(n, seed)] = _top_k_competency_titles(comps_dicts, k=20)
                mean_grounding = (
                    sum(float(c.get("grounding_score", 0.0)) for c in comps_dicts) / max(1, len(comps_dicts))
                )

            rt = time.time() - t0
            rows.append({
                "N": n,
                "seed": seed,
                "n_items_used": len(sub),
                "n_clusters": len(clusters),
                "n_competencies": len(comps_dicts),
                "cohesion_mean": round(cl_report.cohesion_mean, 4),
                "cohesion_min": round(cl_report.cohesion_min, 4),
                "mean_grounding": round(mean_grounding, 4),
                "runtime_seconds": round(rt, 1),
            })

    # ----- Jaccard top-20 between seeds at each N -----
    log.info("computing Jaccard top-20 between seeds...")
    jaccard_rows = []
    for n in ns:
        for i, si in enumerate(seeds):
            for sj in seeds[i + 1 :]:
                ti = titles_by_run.get((n, si), [])
                tj = titles_by_run.get((n, sj), [])
                j = _jaccard(ti, tj)
                jaccard_rows.append({"N": n, "seed_a": si, "seed_b": sj, "jaccard_top20": round(j, 4)})

    # Mean Jaccard per N
    by_n_jaccard: Dict[int, List[float]] = {}
    for r in jaccard_rows:
        by_n_jaccard.setdefault(r["N"], []).append(r["jaccard_top20"])
    n_summary = []
    for n in ns:
        vals = by_n_jaccard.get(n, [])
        if vals:
            import statistics
            n_summary.append({
                "N": n,
                "mean_jaccard_top20": round(statistics.mean(vals), 4),
                "std_jaccard_top20": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                "n_pairs": len(vals),
            })

    # ----- Write outputs -----
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out_dir / "jaccard_matrix.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["N", "seed_a", "seed_b", "jaccard_top20"])
        w.writeheader()
        w.writerows(jaccard_rows)

    with open(out_dir / "n_summary.csv", "w", encoding="utf-8", newline="") as f:
        if n_summary:
            w = csv.DictWriter(f, fieldnames=list(n_summary[0].keys()))
            w.writeheader()
            w.writerows(n_summary)

    # ----- Plot the stability curve -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns_plot = [r["N"] for r in n_summary]
        means = [r["mean_jaccard_top20"] for r in n_summary]
        stds = [r["std_jaccard_top20"] for r in n_summary]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(ns_plot, means, yerr=stds, fmt="o-", capsize=4, color="#4477aa")
        ax.axhline(0.80, color="red", linestyle="--", label="stability threshold 0.80")
        ax.set_xlabel("Corpus size N (items)")
        ax.set_ylabel("Mean Jaccard between seeds (top-20 competencies)")
        ax.set_title("v2 pipeline stability — top-20 competencies vs N")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "stability_curve.png", dpi=140)
        plt.close(fig)
    except Exception as e:
        log.warning("plot failed: %s", e)

    log.info("=== summary ===")
    for r in n_summary:
        log.info("  N=%d : mean_jaccard=%.3f ± %.3f (n_pairs=%d)",
                 r["N"], r["mean_jaccard_top20"], r["std_jaccard_top20"], r["n_pairs"])
    log.info("outputs in %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
