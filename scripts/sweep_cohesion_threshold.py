"""
sweep_cohesion_threshold.py

Run Phase 2.1 clustering at multiple cohesion thresholds on the same corpus,
side-by-side. Helps pick the right threshold before Phase 2.2 lands.

For each threshold, captures:
    - n_clusters (total, hard+knowledge, soft)
    - cohesion mean / median / min / max
    - n_items grouped / ungrouped
    - top-K clusters by size (with their summary_label)
    - whether specific "marker" clusters survive

Outputs:
    results/clustering_sweep/sweep_summary.csv      — one row per threshold
    results/clustering_sweep/clusters_t<NN>.csv     — clusters at each threshold
    results/clustering_sweep/sweep_metrics.png      — side-by-side comparison plots
    results/clustering_sweep/sweep_report.txt       — human-readable summary

Usage:
    python scripts/sweep_cohesion_threshold.py
    python scripts/sweep_cohesion_threshold.py --thresholds 0.40,0.45,0.50,0.55,0.60,0.65
    python scripts/sweep_cohesion_threshold.py --input-dir results.old2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Re-use the real-data loader from the existing smoke-test script
from scripts.test_clustering_on_real_data import load_real_items


def _write_sweep_summary_csv(rows, out_path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_clusters_csv(clusters, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cluster_id",
                "stream",
                "method",
                "n_items",
                "n_unique_jobs",
                "cohesion_score",
                "summary_label",
                "top_terms",
            ]
        )
        for c in clusters:
            w.writerow(
                [
                    c.id,
                    c.stream,
                    c.method,
                    c.n_items,
                    c.n_unique_jobs,
                    f"{c.cohesion_score:.4f}",
                    c.summary_label,
                    " | ".join(c.top_terms),
                ]
            )


def _maybe_write_metric_plot(rows, out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thresholds = [r["cohesion_threshold"] for r in rows]
        n_clusters = [r["n_clusters"] for r in rows]
        n_ungrouped = [r["n_items_ungrouped"] for r in rows]
        cohesion_mean = [r["cohesion_mean"] for r in rows]
        cohesion_min = [r["cohesion_min"] for r in rows]
        items_grouped = [r["n_items_grouped"] for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))

        ax = axes[0, 0]
        ax.plot(thresholds, n_clusters, "o-", color="#4477aa", label="total clusters")
        ax.set_xlabel("cohesion threshold")
        ax.set_ylabel("number of clusters")
        ax.set_title("Cluster count vs threshold")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(thresholds, items_grouped, "s-", color="#228833", label="items in clusters")
        ax.plot(thresholds, n_ungrouped, "x-", color="#cc3311", label="items ungrouped")
        ax.set_xlabel("cohesion threshold")
        ax.set_ylabel("number of items")
        ax.set_title("Item routing (grouped vs ungrouped)")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1, 0]
        ax.plot(thresholds, cohesion_mean, "o-", color="#aa3377", label="mean cohesion")
        ax.plot(thresholds, cohesion_min, "v-", color="#bbbbbb", label="min cohesion")
        for t in thresholds:
            ax.axvline(t, color="#dddddd", alpha=0.3, linestyle=":")
        ax.plot(thresholds, thresholds, "--", color="black", alpha=0.5, label="y = threshold")
        ax.set_xlabel("cohesion threshold")
        ax.set_ylabel("cohesion")
        ax.set_title("Surviving-cluster cohesion vs threshold")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1, 1]
        method_keys = ("n_hdbscan", "n_recovered", "n_split")
        method_labels = ("HDBSCAN", "agglomerative recovery", "oversize split")
        bottom = [0] * len(thresholds)
        colors = ["#4477aa", "#ee9933", "#aa3377"]
        for k, lbl, col in zip(method_keys, method_labels, colors):
            vals = [r[k] for r in rows]
            ax.bar(
                [f"{t:.2f}" for t in thresholds],
                vals,
                bottom=bottom,
                color=col,
                label=lbl,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xlabel("cohesion threshold")
        ax.set_ylabel("number of clusters by method")
        ax.set_title("Cluster source breakdown")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle("Cohesion threshold sweep — Phase 2.1 clustering", fontsize=12, y=1.00)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] failed to write sweep plot: {e}")
        return False


def _write_sweep_report_txt(rows, all_clusters_by_t, out_path: Path) -> None:
    lines = []
    lines.append("=" * 88)
    lines.append("Phase 2.1 cohesion-threshold sweep — summary")
    lines.append("=" * 88)
    lines.append("")
    lines.append(
        f"{'thr':>5s} {'n_clusters':>10s} {'hpk':>4s} {'soft':>5s} "
        f"{'grouped':>8s} {'ungrouped':>10s} {'mean':>7s} {'min':>7s}"
    )
    for r in rows:
        lines.append(
            f"{r['cohesion_threshold']:>5.2f} {r['n_clusters']:>10d} "
            f"{r['n_clusters_hpk']:>4d} {r['n_clusters_soft']:>5d} "
            f"{r['n_items_grouped']:>8d} {r['n_items_ungrouped']:>10d} "
            f"{r['cohesion_mean']:>7.3f} {r['cohesion_min']:>7.3f}"
        )
    lines.append("")

    for t, clusters in all_clusters_by_t.items():
        lines.append("-" * 88)
        lines.append(f"[THRESHOLD = {t:.2f}]  ({len(clusters)} clusters)")
        lines.append("-" * 88)
        sorted_c = sorted(clusters, key=lambda c: c.n_items, reverse=True)
        for c in sorted_c[:15]:
            lines.append(
                f"  {c.id:<18s} stream={c.stream:<21s} n={c.n_items:>3d}  "
                f"cohesion={c.cohesion_score:.3f}  → {c.summary_label}"
            )
        if len(sorted_c) > 15:
            lines.append(f"  ... and {len(sorted_c) - 15} more")
        lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep cohesion threshold on real Phase 1 output.")
    parser.add_argument("--input-dir", default="results.old2")
    parser.add_argument("--output-dir", default="results/clustering_sweep")
    parser.add_argument(
        "--thresholds",
        default="0.40,0.45,0.50,0.55,0.60,0.65",
        help="comma-separated cohesion thresholds to sweep",
    )
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-global-freq", type=int, default=2)
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading items from {in_dir}")
    items = load_real_items(in_dir)
    print(f"[INFO] loaded {len(items)} items")

    from clustering import ClusteringConfig, cluster_skills

    rows = []
    all_clusters_by_t = {}

    for t in thresholds:
        print(f"\n[SWEEP] cohesion_threshold = {t:.2f}")
        config = ClusteringConfig(
            cohesion_threshold=t,
            min_cluster_size=args.min_cluster_size,
            min_global_frequency=args.min_global_freq,
        )
        clusters, report = cluster_skills(items, config=config)

        cohesions = [c.cohesion_score for c in clusters]
        items_grouped = sum(c.n_items for c in clusters)
        n_hpk = sum(1 for c in clusters if c.stream == "hard_plus_knowledge")
        n_soft = sum(1 for c in clusters if c.stream == "soft_skill")

        row = {
            "cohesion_threshold": t,
            "n_clusters": len(clusters),
            "n_clusters_hpk": n_hpk,
            "n_clusters_soft": n_soft,
            "n_hdbscan": report.n_clusters_hdbscan,
            "n_recovered": report.n_clusters_recovered,
            "n_split": report.n_clusters_split,
            "n_dropped_low_cohesion": report.n_clusters_dropped_low_cohesion,
            "n_items_grouped": items_grouped,
            "n_items_ungrouped": report.n_items_ungrouped,
            "cohesion_mean": round(report.cohesion_mean, 4),
            "cohesion_median": round(report.cohesion_median, 4),
            "cohesion_min": round(report.cohesion_min, 4),
            "cohesion_max": round(max(cohesions), 4) if cohesions else 0.0,
            "runtime_seconds": round(report.runtime_seconds, 2),
        }
        rows.append(row)
        all_clusters_by_t[t] = clusters

        # Write per-threshold cluster CSV
        _write_clusters_csv(clusters, out_dir / f"clusters_t{int(t * 100):02d}.csv")
        print(
            f"        produced {len(clusters)} clusters "
            f"(hpk={n_hpk}, soft={n_soft}); "
            f"mean cohesion={report.cohesion_mean:.3f}, "
            f"min={report.cohesion_min:.3f}, "
            f"items_grouped={items_grouped}, "
            f"ungrouped={report.n_items_ungrouped}"
        )

    # Sweep summary CSV
    _write_sweep_summary_csv(rows, out_dir / "sweep_summary.csv")
    # Comparison plot
    if _maybe_write_metric_plot(rows, out_dir / "sweep_metrics.png"):
        print(f"\n[INFO] wrote sweep_metrics.png")
    # Human-readable report
    _write_sweep_report_txt(rows, all_clusters_by_t, out_dir / "sweep_report.txt")

    print(f"\n[INFO] outputs in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
