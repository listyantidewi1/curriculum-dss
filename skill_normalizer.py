"""
skill_normalizer.py  —  Item 1: Skill normalization / entity linking

Groups near-duplicate skill strings extracted by the pipeline into canonical
forms using SBERT cosine-similarity clustering.

Examples of equivalent skills that get merged:
    "Python"  /  "Python programming"  /  "Python scripting"   -> "Python programming"
    "REST API"  /  "RESTful API"  /  "REST APIs"               -> "REST API"

Usage (standalone):
    python skill_normalizer.py --input results/advanced_skills.csv --output results/advanced_skills_normalized.csv

Usage (library):
    from skill_normalizer import normalize_skills, apply_normalization
    mapping = normalize_skills(skill_list)           # {raw -> canonical}
    df = apply_normalization(df, skill_col="skill")  # adds canonical_skill column
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Cosine similarity threshold to merge two skill strings into one cluster.
# 0.82 is tight enough to avoid false merges but catches obvious variants.
DEFAULT_THRESHOLD = 0.82

# -------------------------------------------------------------------
# Core normalization
# -------------------------------------------------------------------

def _lazy_sbert(model_name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def normalize_skills(
    skills: List[str],
    threshold: float = DEFAULT_THRESHOLD,
    model=None,
) -> Dict[str, str]:
    """
    Return {raw_skill -> canonical_skill} mapping.

    The canonical name for a cluster is the most-frequent variant; ties are
    broken by preferring the shortest string (clearest label).

    Algorithm: greedy single-pass clustering — each new skill joins the first
    existing cluster whose centroid has cosine-similarity >= threshold with
    the new embedding; otherwise starts a new cluster.  Centroids are updated
    as the mean of all member embeddings (re-normalised).
    """
    if not skills:
        return {}

    unique: List[str] = list(dict.fromkeys(
        s.strip() for s in skills if s and str(s).strip()
    ))
    if len(unique) == 1:
        return {unique[0]: unique[0]}

    if model is None:
        model = _lazy_sbert()

    embeddings = model.encode(
        unique,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )  # shape (n, d), L2-normalised -> dot product == cosine similarity

    # Frequency map (counts from the original list, not deduplicated unique)
    freq: Dict[str, int] = {}
    for s in skills:
        key = s.strip()
        freq[key] = freq.get(key, 0) + 1

    # Greedy clustering
    cluster_centroids: List[np.ndarray] = []
    cluster_members: Dict[int, List[int]] = {}   # cluster_id -> [unique_idx, ...]
    cluster_id_map: Dict[int, int] = {}           # unique_idx -> cluster_id

    for i, emb in enumerate(embeddings):
        best_cid = -1
        best_sim = threshold  # minimum accepted similarity

        for cid, centroid in enumerate(cluster_centroids):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_cid = cid

        if best_cid == -1:
            # New cluster
            best_cid = len(cluster_centroids)
            cluster_centroids.append(emb.copy())
            cluster_members[best_cid] = [i]
        else:
            # Join existing cluster, update centroid
            cluster_members[best_cid].append(i)
            member_embs = embeddings[cluster_members[best_cid]]
            centroid = np.mean(member_embs, axis=0)
            norm = np.linalg.norm(centroid)
            cluster_centroids[best_cid] = centroid / (norm + 1e-10)

        cluster_id_map[i] = best_cid

    # Choose canonical name per cluster
    mapping: Dict[str, str] = {}
    for cid, members in cluster_members.items():
        candidates = [(unique[i], freq.get(unique[i], 1)) for i in members]
        # Highest frequency; ties broken by shortest string
        canonical = max(candidates, key=lambda x: (x[1], -len(x[0])))[0]
        for i in members:
            mapping[unique[i]] = canonical

    return mapping


def apply_normalization(
    df: pd.DataFrame,
    skill_col: str = "skill",
    threshold: float = DEFAULT_THRESHOLD,
    model=None,
) -> pd.DataFrame:
    """
    Add a ``canonical_skill`` column to *df*.

    The original ``skill`` column is preserved unchanged.  Downstream
    aggregation (e.g. in ``recommendations.build_skill_demand``) should
    group on ``canonical_skill`` rather than ``skill``.
    """
    skills = df[skill_col].fillna("").astype(str).tolist()
    mapping = normalize_skills(skills, threshold=threshold, model=model)
    df = df.copy()
    df["canonical_skill"] = df[skill_col].map(
        lambda s: mapping.get(str(s).strip(), str(s).strip())
    )
    return df


# -------------------------------------------------------------------
# Standalone CLI
# -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize/deduplicate skill strings using SBERT clustering"
    )
    parser.add_argument("--input", required=True, help="Input CSV with a 'skill' column")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--skill-col", default="skill", help="Column name containing raw skill strings"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for merging (default {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--report", default=None, help="Optional JSON path to save cluster report"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.skill_col not in df.columns:
        raise SystemExit(f"[ERROR] Column '{args.skill_col}' not found in {args.input}")

    print(f"[INFO] Normalizing {len(df)} rows, {df[args.skill_col].nunique()} unique skills …")
    df = apply_normalization(df, skill_col=args.skill_col, threshold=args.threshold)

    before = df[args.skill_col].nunique()
    after = df["canonical_skill"].nunique()
    print(f"[INFO] Unique skills: {before} → {after} (merged {before - after} variants)")

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved normalized output → {args.output}")

    if args.report:
        # Cluster summary: canonical -> list of original variants
        clusters: Dict[str, List[str]] = {}
        for raw, canonical in zip(df[args.skill_col], df["canonical_skill"]):
            clusters.setdefault(canonical, [])
            if raw not in clusters[canonical]:
                clusters[canonical].append(raw)
        multi = {k: v for k, v in clusters.items() if len(v) > 1}
        report = {
            "unique_before": before,
            "unique_after": after,
            "clusters_merged": len(multi),
            "merged_clusters": multi,
        }
        Path(args.report).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[INFO] Cluster report → {args.report}")


if __name__ == "__main__":
    main()
