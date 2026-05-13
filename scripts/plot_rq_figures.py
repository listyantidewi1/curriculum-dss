"""
plot_rq_figures.py

Generate paper-grade figures for the v2 sprint research questions.

Figures produced (each saved to results/figures/v2/):

    rq1b_competency_grounding_distribution.png
        Histogram of grounding_score across all surviving competencies
        from results/competency_v2_pipeline_e2e_v1/. RQ1b — automated
        quality gate; gate threshold 0.80 marked.

    rq5_model_ab_comparison.png
        Bar/line plot from results/competency_v2_comparison/comparison.csv
        showing per-model: n_competencies, cluster coverage, mean grounding,
        latency. The 6-model A/B made on 2026-05-12.

    rq2_cluster_cohesion_distribution.png
        Histogram of cluster cohesion scores from any clustering smoke run.
        Threshold 0.55 (current default) marked.

    rq5_provider_choice_summary.png
        Side-by-side of working models' titles count + grounding + latency
        ranked by best speed-quality combo.

Designed to be safe to re-run anytime — reads the latest output dirs and
overwrites figures.

Usage:
    python scripts/plot_rq_figures.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_competencies(path: Path):
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def plot_rq1b_grounding(out_dir: Path, pipeline_dir: Path) -> bool:
    """Histogram of grounding_score with the 0.80 gate marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = _load_competencies(pipeline_dir / "competencies.json")
    if not comps:
        print(f"[SKIP] rq1b — no competencies at {pipeline_dir}")
        return False
    scores = [
        float(c.get("grounding_score") or c.get("grounding_score_preview", 0.0))
        for c in comps
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, range=(0.0, 1.05), color="#4477aa", edgecolor="black")
    ax.axvline(0.80, color="red", linestyle="--", label="grounding gate = 0.80")
    ax.set_xlabel("grounding_score (verified / total related_skills)")
    ax.set_ylabel("number of competencies")
    ax.set_title(f"RQ1b — competency grounding distribution (n={len(comps)})")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "rq1b_competency_grounding_distribution.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def plot_rq2_cluster_cohesion(out_dir: Path, clustering_dir: Path) -> bool:
    """Histogram of cluster cohesion scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clusters_path = clustering_dir / "clusters.json"
    if not clusters_path.exists():
        print(f"[SKIP] rq2 — no clusters at {clustering_dir}")
        return False
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    if not clusters:
        return False
    scores = [float(c.get("cohesion_score", 0.0)) for c in clusters]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=15, color="#228833", edgecolor="black")
    ax.axvline(0.55, color="red", linestyle="--", label="cohesion gate = 0.55")
    ax.set_xlabel("intra-cluster cohesion (mean pairwise SBERT cosine)")
    ax.set_ylabel("number of clusters")
    ax.set_title(f"RQ2 — Phase 2.1 cluster cohesion distribution (n={len(clusters)})")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "rq2_cluster_cohesion_distribution.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def plot_rq5_model_ab(out_dir: Path, comparison_dir: Path) -> bool:
    """Per-model bar plot from the 6-model A/B."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = comparison_dir / "comparison.csv"
    if not csv_path.exists():
        print(f"[SKIP] rq5 model A/B — comparison.csv not found at {csv_path}")
        return False
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return False

    models = [r["model"] for r in rows]
    n_comps = [int(r["n_competencies"]) for r in rows]
    coverage = [int(r["n_clusters_covered"]) for r in rows]
    grounding = [float(r["mean_grounding"]) for r in rows]
    latency = [float(r["latency_seconds"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    ax = axes[0, 0]
    bars = ax.barh(models, n_comps, color="#4477aa", edgecolor="black")
    ax.set_xlabel("Competencies produced")
    ax.set_title("Competencies surviving (post-grounding gate)")
    for b, v in zip(bars, n_comps):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[0, 1]
    ax.barh(models, coverage, color="#228833", edgecolor="black")
    ax.set_xlabel("Clusters covered (of 10 hard+knowledge)")
    ax.set_title("Cluster coverage")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 0]
    ax.barh(models, grounding, color="#aa3377", edgecolor="black")
    ax.set_xlim(0.8, 1.02)
    ax.set_xlabel("Mean grounding score")
    ax.set_title("Mean grounding (gate ≥ 0.80)")
    ax.axvline(0.80, color="red", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1, 1]
    ax.barh(models, latency, color="#ee9933", edgecolor="black")
    ax.set_xlabel("Total wall-clock (s) — same 10 clusters")
    ax.set_title("Latency (lower = faster)")
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("RQ5 — Phase 2.2 competency-generator model A/B (2026-05-12)", fontsize=12, y=1.00)
    fig.tight_layout()
    out_path = out_dir / "rq5_model_ab_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def plot_rq5b_education_demand(out_dir: Path, pipeline_dir: Path) -> bool:
    """Aggregate education-level demand distribution across all competencies.

    Skipped silently if no competency has populated education_levels_demanded
    (which happens when jobs_metadata.csv didn't match the source job_ids,
    e.g., results.old2 + 1K-job metadata mismatch).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = _load_competencies(pipeline_dir / "competencies.json")
    if not comps:
        return False

    from collections import Counter
    stage_totals: Counter = Counter()
    n_comp_with_data = 0
    for c in comps:
        edu = c.get("education_levels_demanded") or {}
        if not edu:
            continue
        n_comp_with_data += 1
        for stage, frac in edu.items():
            stage_totals[stage] += float(frac)
    if not stage_totals:
        print(f"[SKIP] rq5b education — no competency had populated education_levels_demanded")
        return False

    # Normalize: average frac per stage across competencies-with-data
    norm = {k: v / n_comp_with_data for k, v in stage_totals.items()}
    stages = sorted(norm.keys(), key=lambda s: -norm[s])
    fractions = [norm[s] for s in stages]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(stages, fractions, color="#4477aa", edgecolor="black")
    for b, v in zip(bars, fractions):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("Mean fraction of source jobs demanding this stage")
    ax.set_xlabel("Education stage (Indonesian KKNI labels)")
    ax.set_title(f"RQ5 — Phase 2.4 education-level demand (n_competencies={n_comp_with_data})")
    ax.set_ylim(0, max(fractions) * 1.15)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = out_dir / "rq5b_education_demand.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[OK] {out_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-dir",
        default="results/competency_v2_pipeline_e2e_v1",
        help="Dir containing competencies.json + clusters.json",
    )
    parser.add_argument(
        "--comparison-dir",
        default="results/competency_v2_comparison",
        help="Dir containing the 6-model A/B comparison.csv",
    )
    parser.add_argument("--output-dir", default="results/figures/v2")
    args = parser.parse_args()

    pipeline_dir = PROJECT_ROOT / args.pipeline_dir
    comparison_dir = PROJECT_ROOT / args.comparison_dir
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    if plot_rq1b_grounding(out_dir, pipeline_dir):
        n_ok += 1
    if plot_rq2_cluster_cohesion(out_dir, pipeline_dir):
        n_ok += 1
    if plot_rq5_model_ab(out_dir, comparison_dir):
        n_ok += 1
    if plot_rq5b_education_demand(out_dir, pipeline_dir):
        n_ok += 1

    print()
    print(f"[INFO] generated {n_ok} figures in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
