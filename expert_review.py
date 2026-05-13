"""
expert_review.py

Phase X — structured expert-review pipeline for paper-grade competency evaluation.

Per the locked plan (3 reviewers, 175 unique competencies = 50 unique × 3 + 25
shared, rubric per docs/EXPERT_REVIEW_RUBRIC.md):

    Per-competency rating, 5 dimensions:
      validity            (1-5 Likert)
      relevance           (1-5 Likert)
      specificity         (1-5 Likert)
      reasoning_quality   (1-5 Likert)  — NEW for v2 per locked design Q3
      recommend           (Yes / No)

Inter-rater reliability (IRR) on the shared 25:
    Fleiss' Kappa for the Likert columns (k=3 raters, ordinal)
    Cohen's Kappa pairwise averaged for the binary recommend column
    Free-marginal (Randolph's) Kappa for skewed marginals

Targets (Landis & Koch 1977):
    < 0.20  Slight     (reject; redo rubric)
    0.21-0.40 Fair     (flag in paper)
    0.41-0.60 Moderate (minimum acceptable)
    0.61-0.80 Substantial (target)
    > 0.80   Almost perfect

Outputs:
    feedback_store/expert_review/
        review_assignments.csv   — which reviewer rates which competencies
        review_responses.csv      — collected ratings (one row per (reviewer, comp, dim))
        inter_rater_report.json   — Fleiss' Kappa per dim + Cohen's Kappa for recommend
        IRR_report.md             — human-readable report for the paper
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REVIEW_DIR = Path("feedback_store") / "expert_review"
ASSIGNMENTS_PATH = REVIEW_DIR / "review_assignments.csv"
RESPONSES_PATH = REVIEW_DIR / "review_responses.csv"
IRR_JSON_PATH = REVIEW_DIR / "inter_rater_report.json"
IRR_MD_PATH = REVIEW_DIR / "IRR_report.md"

LIKERT_DIMENSIONS = ("validity", "relevance", "specificity", "reasoning_quality")
BINARY_DIMENSIONS = ("recommend",)


# --------------------------------------------------------------------------- #
# Assignment generator
# --------------------------------------------------------------------------- #


def generate_assignments(
    competency_ids: Sequence[str],
    reviewer_ids: Sequence[str] = ("rev_1", "rev_2", "rev_3"),
    n_unique_per_reviewer: int = 50,
    n_shared: int = 25,
    seed: int = 42,
) -> List[dict]:
    """Generate per-reviewer assignments.

    Each reviewer rates `n_unique_per_reviewer` competencies unique to them
    plus the same `n_shared` competencies used for IRR. Returns rows for
    `review_assignments.csv`.
    """
    rng = random.Random(seed)
    pool = list(competency_ids)
    if len(pool) < len(reviewer_ids) * n_unique_per_reviewer + n_shared:
        raise ValueError(
            f"need >= {len(reviewer_ids) * n_unique_per_reviewer + n_shared} competencies "
            f"in pool; got {len(pool)}"
        )
    rng.shuffle(pool)
    shared = pool[:n_shared]
    rest = pool[n_shared:]

    rows = []
    for i, rev in enumerate(reviewer_ids):
        unique = rest[i * n_unique_per_reviewer : (i + 1) * n_unique_per_reviewer]
        for cid in shared:
            rows.append({"reviewer_id": rev, "competency_id": cid, "subset": "shared"})
        for cid in unique:
            rows.append({"reviewer_id": rev, "competency_id": cid, "subset": "unique"})
    return rows


def write_assignments(rows: List[dict], path: Path = ASSIGNMENTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["reviewer_id", "competency_id", "subset"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Kappa computations
# --------------------------------------------------------------------------- #


def fleiss_kappa(rating_matrix: List[List[int]]) -> float:
    """Fleiss' Kappa for k raters rating N items into c ordinal categories.

    `rating_matrix` is a list of N rows; each row is a length-c list where
    entry j = number of raters who assigned category j to that item.

    Returns kappa (between -1 and 1; >= 0.6 is "substantial").
    """
    N = len(rating_matrix)
    if N == 0:
        return 0.0
    c = len(rating_matrix[0])
    if c == 0:
        return 0.0
    n_raters = sum(rating_matrix[0])  # constant across items by construction

    # Per-item agreement P_i
    P_is = []
    for row in rating_matrix:
        if sum(row) != n_raters:
            continue  # skip items not rated by all raters
        P_i = (sum(x ** 2 for x in row) - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 0.0
        P_is.append(P_i)
    if not P_is:
        return 0.0
    P_bar = statistics.mean(P_is)

    # Category-marginal P_j
    total = N * n_raters
    P_js = [sum(rating_matrix[i][j] for i in range(N)) / total for j in range(c)]
    Pe = sum(p ** 2 for p in P_js)

    if Pe >= 1.0:
        return 1.0
    return (P_bar - Pe) / (1.0 - Pe)


def cohen_kappa_pairwise(rater_a: List[int], rater_b: List[int]) -> float:
    """Cohen's Kappa for two raters' parallel rating sequences."""
    if len(rater_a) != len(rater_b) or not rater_a:
        return 0.0
    n = len(rater_a)
    categories = sorted(set(rater_a) | set(rater_b))
    if len(categories) < 2:
        return 1.0  # everyone agreed on the same category

    # Observed agreement
    Po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    # Expected agreement
    a_counts = Counter(rater_a)
    b_counts = Counter(rater_b)
    Pe = sum((a_counts[c] / n) * (b_counts[c] / n) for c in categories)
    if Pe >= 1.0:
        return 1.0
    return (Po - Pe) / (1.0 - Pe)


def randolph_kappa(rating_matrix: List[List[int]]) -> float:
    """Free-marginal (Randolph 2005) Kappa — addresses Cohen's paradox on
    highly-skewed marginals. Useful when most ratings are the same value.

    Same `rating_matrix` shape as `fleiss_kappa`.
    """
    N = len(rating_matrix)
    if N == 0:
        return 0.0
    c = len(rating_matrix[0])
    n_raters = sum(rating_matrix[0])
    if n_raters <= 1 or c == 0:
        return 0.0

    P_is = []
    for row in rating_matrix:
        P_i = (sum(x ** 2 for x in row) - n_raters) / (n_raters * (n_raters - 1))
        P_is.append(P_i)
    P_bar = statistics.mean(P_is)
    Pe_free = 1.0 / c
    if Pe_free >= 1.0:
        return 1.0
    return (P_bar - Pe_free) / (1.0 - Pe_free)


def landis_koch_label(kappa: float) -> str:
    if kappa < 0.20:
        return "Slight"
    if kappa < 0.40:
        return "Fair"
    if kappa < 0.60:
        return "Moderate"
    if kappa < 0.80:
        return "Substantial"
    return "Almost perfect"


# --------------------------------------------------------------------------- #
# Build rating matrices from review_responses.csv
# --------------------------------------------------------------------------- #


def _read_responses(path: Path = RESPONSES_PATH) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_likert_matrix(
    responses: List[dict],
    dimension: str,
    shared_competency_ids: Sequence[str],
    reviewer_ids: Sequence[str],
    n_categories: int = 5,
) -> List[List[int]]:
    """Build a (N items × n_categories) matrix for Fleiss' Kappa."""
    matrix = []
    for cid in shared_competency_ids:
        row = [0] * n_categories
        for rev in reviewer_ids:
            for r in responses:
                if (r["competency_id"] == cid and
                        r["reviewer_id"] == rev and
                        r["dimension"] == dimension):
                    try:
                        v = int(r["rating"])
                    except (TypeError, ValueError):
                        continue
                    if 1 <= v <= n_categories:
                        row[v - 1] += 1
                    break
        if sum(row) == len(reviewer_ids):
            matrix.append(row)
    return matrix


def build_binary_pairs(
    responses: List[dict],
    dimension: str,
    shared_competency_ids: Sequence[str],
    reviewer_a: str,
    reviewer_b: str,
) -> Tuple[List[int], List[int]]:
    """Return parallel rating sequences for Cohen's Kappa."""
    a_seq, b_seq = [], []
    for cid in shared_competency_ids:
        a_val = b_val = None
        for r in responses:
            if r["competency_id"] == cid and r["dimension"] == dimension:
                v = 1 if str(r["rating"]).strip().lower() in ("yes", "1", "true") else 0
                if r["reviewer_id"] == reviewer_a:
                    a_val = v
                elif r["reviewer_id"] == reviewer_b:
                    b_val = v
        if a_val is not None and b_val is not None:
            a_seq.append(a_val)
            b_seq.append(b_val)
    return a_seq, b_seq


# --------------------------------------------------------------------------- #
# IRR top-level
# --------------------------------------------------------------------------- #


def compute_irr(
    assignments_path: Path = ASSIGNMENTS_PATH,
    responses_path: Path = RESPONSES_PATH,
) -> dict:
    """Compute Fleiss' + Cohen's + Randolph's Kappa across the shared subset."""
    if not assignments_path.exists():
        raise FileNotFoundError(f"{assignments_path} not found; run generate_assignments first")

    with open(assignments_path, encoding="utf-8") as f:
        assignments = list(csv.DictReader(f))
    shared_cids = sorted({a["competency_id"] for a in assignments if a.get("subset") == "shared"})
    reviewer_ids = sorted({a["reviewer_id"] for a in assignments})

    responses = _read_responses(responses_path)

    report = {
        "n_shared_competencies": len(shared_cids),
        "reviewer_ids": reviewer_ids,
        "likert_dimensions": {},
        "binary_dimensions": {},
        "interpretation": "Landis & Koch 1977 thresholds",
    }

    for dim in LIKERT_DIMENSIONS:
        matrix = build_likert_matrix(responses, dim, shared_cids, reviewer_ids)
        fk = fleiss_kappa(matrix)
        rk = randolph_kappa(matrix)
        report["likert_dimensions"][dim] = {
            "n_items_with_full_ratings": len(matrix),
            "fleiss_kappa": round(fk, 4),
            "fleiss_label": landis_koch_label(fk),
            "randolph_kappa": round(rk, 4),
            "randolph_label": landis_koch_label(rk),
        }

    for dim in BINARY_DIMENSIONS:
        per_pair = []
        for i, rev_a in enumerate(reviewer_ids):
            for rev_b in reviewer_ids[i + 1 :]:
                a, b = build_binary_pairs(responses, dim, shared_cids, rev_a, rev_b)
                if a and b:
                    ck = cohen_kappa_pairwise(a, b)
                    per_pair.append({"pair": f"{rev_a}_vs_{rev_b}", "kappa": round(ck, 4), "n": len(a)})
        mean_ck = statistics.mean(p["kappa"] for p in per_pair) if per_pair else 0.0
        report["binary_dimensions"][dim] = {
            "pairwise": per_pair,
            "mean_cohen_kappa": round(mean_ck, 4),
            "label": landis_koch_label(mean_ck),
        }

    return report


def write_irr_report(report: dict, json_path: Path = IRR_JSON_PATH, md_path: Path = IRR_MD_PATH) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Inter-Rater Reliability Report")
    lines.append("")
    lines.append(f"Shared competencies (k=3 raters): **{report['n_shared_competencies']}**")
    lines.append(f"Reviewers: {', '.join(report['reviewer_ids'])}")
    lines.append("")
    lines.append("## Likert dimensions (Fleiss' Kappa)")
    lines.append("")
    lines.append("| Dimension | n items | Fleiss κ | Label | Randolph κ | Label |")
    lines.append("|---|---|---|---|---|---|")
    for dim, d in report["likert_dimensions"].items():
        lines.append(
            f"| {dim} | {d['n_items_with_full_ratings']} | {d['fleiss_kappa']} | "
            f"{d['fleiss_label']} | {d['randolph_kappa']} | {d['randolph_label']} |"
        )
    lines.append("")
    lines.append("## Binary dimensions (Cohen's Kappa, averaged pairwise)")
    lines.append("")
    for dim, d in report["binary_dimensions"].items():
        lines.append(f"### {dim}")
        lines.append(f"Mean pairwise Cohen's Kappa: **{d['mean_cohen_kappa']}** ({d['label']})")
        for p in d["pairwise"]:
            lines.append(f"  - {p['pair']}: κ={p['kappa']} (n={p['n']})")
        lines.append("")
    lines.append("## Acceptance (Landis & Koch 1977)")
    lines.append("")
    lines.append("| Kappa | Label | Action |")
    lines.append("|---|---|---|")
    lines.append("| < 0.20 | Slight | Reject; redo rubric |")
    lines.append("| 0.21–0.40 | Fair | Flag in paper limitations |")
    lines.append("| 0.41–0.60 | Moderate | **Minimum acceptable** |")
    lines.append("| 0.61–0.80 | Substantial | Target |")
    lines.append("| > 0.80 | Almost perfect | Excellent |")
    md_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("assign", help="generate review_assignments.csv from a pipeline output")
    g.add_argument("--pipeline-dir", default="results/competency_v2_pipeline_e2e_v1")
    g.add_argument("--reviewers", default="rev_1,rev_2,rev_3")
    g.add_argument("--n-unique", type=int, default=50)
    g.add_argument("--n-shared", type=int, default=25)
    g.add_argument("--seed", type=int, default=42)

    i = sub.add_parser("irr", help="compute Fleiss' + Cohen's Kappa from review_responses.csv")

    args = parser.parse_args()

    if args.cmd == "assign":
        pdir = Path(args.pipeline_dir)
        if not pdir.is_absolute():
            pdir = Path(__file__).resolve().parent / pdir if False else Path.cwd() / pdir
        comps_path = pdir / "competencies.json"
        comps = json.loads(comps_path.read_text(encoding="utf-8"))
        cids = [c["id"] for c in comps]
        reviewers = [r.strip() for r in args.reviewers.split(",")]
        rows = generate_assignments(
            cids,
            reviewer_ids=reviewers,
            n_unique_per_reviewer=args.n_unique,
            n_shared=args.n_shared,
            seed=args.seed,
        )
        write_assignments(rows)
        print(f"[OK] wrote {ASSIGNMENTS_PATH} ({len(rows)} assignment rows)")
        # Also drop an empty review_responses.csv template
        if not RESPONSES_PATH.exists():
            with open(RESPONSES_PATH, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["reviewer_id", "competency_id", "dimension", "rating", "notes", "timestamp"])
            print(f"[OK] wrote empty template {RESPONSES_PATH}")
        return 0

    elif args.cmd == "irr":
        report = compute_irr()
        write_irr_report(report)
        print(f"[OK] wrote {IRR_JSON_PATH} and {IRR_MD_PATH}")
        # Print summary
        for dim, d in report["likert_dimensions"].items():
            print(f"  Fleiss κ ({dim}): {d['fleiss_kappa']} ({d['fleiss_label']})")
        for dim, d in report["binary_dimensions"].items():
            print(f"  Cohen κ ({dim}, mean pairwise): {d['mean_cohen_kappa']} ({d['label']})")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
