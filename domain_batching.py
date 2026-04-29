"""
domain_batching.py

Helper module for domain-based batching in competency generation.
Provides normalized-key lookup, domain assignment, batch building,
and merge logic for strongly-similar domain batches.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse embedding model from pipeline
from pipeline import AdvancedPipelineConfig

TREND_ORDER = {"Strong_Growth": 3, "Moderate_Growth": 2, "Stable": 1, "Decline": 0}


def normalize_for_grouping(text: str) -> str:
    """
    Deterministic normalization for grouping equivalent skills/knowledge.
    Matches future_weight_mapping logic: lowercase, collapse punctuation/spaces.
    """
    if not text or not isinstance(text, str):
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_future_domains(path: Path) -> pd.DataFrame:
    """Load future_domains.csv and build domain_text for embedding."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"domain_id", "future_domain", "example_terms", "trend_label", "trend_score"}
    if required - set(df.columns):
        return pd.DataFrame()
    df["trend_score"] = pd.to_numeric(df["trend_score"], errors="coerce").fillna(0.0)
    df["domain_text"] = (
        df["future_domain"].astype(str).str.strip()
        + ". "
        + df["example_terms"].astype(str).str.strip()
    )
    return df


def compute_domain_embeddings(
    domains_df: pd.DataFrame,
    embedder,
) -> Dict[str, np.ndarray]:
    """Compute embeddings for each future_domain. Keys are future_domain strings."""
    if domains_df.empty:
        return {}
    texts = domains_df["domain_text"].tolist()
    domains = domains_df["future_domain"].astype(str).str.strip().tolist()
    emb = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {d: emb[i] for i, d in enumerate(domains)}


def assign_domain_for_skill(
    skill: str,
    domains_df: pd.DataFrame,
    embedder,
) -> Tuple[str, float, str]:
    """
    Assign a skill to the nearest future domain via embedding similarity.
    Returns (best_future_domain, future_weight, trend_label).
    Used as on-the-fly fallback for skills not in future_skill_weights.
    """
    if domains_df.empty:
        return ("Unmapped", 0.0, "")
    skill_text = str(skill).strip()
    if not skill_text:
        return ("Unmapped", 0.0, "")
    # Build domain texts (future_domain + example_terms)
    domain_texts = domains_df["domain_text"].tolist()
    skill_emb = embedder.encode(
        [skill_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    domain_emb = embedder.encode(
        domain_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    sims = np.dot(domain_emb, skill_emb.T).flatten()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    row = domains_df.iloc[best_idx]
    trend_score = float(row.get("trend_score", 0) or 0)
    future_weight = best_sim * trend_score
    return (
        str(row["future_domain"]),
        future_weight,
        str(row.get("trend_label", "")),
    )


UNRELATED_DOMAINS = {"Uncertain", "Unmapped"}
_SUBCLUSTER_THRESHOLD = 15  # only sub-cluster if more than this many skills land in Uncertain/Unmapped
_SUBCLUSTER_TARGET_SIZE = 10  # each sub-cluster aims for ~this many skills


def subcluster_unrelated_skills(
    skills: List[str],
    embedder,
    target_size: int = _SUBCLUSTER_TARGET_SIZE,
) -> List[List[str]]:
    """Cluster a list of unrelated skills (Uncertain/Unmapped) into thematic sub-groups.

    Uses SBERT embeddings + agglomerative clustering. Returns a list of sub-groups
    (each a list of skill strings). Falls back to a single group containing all
    skills if clustering is unavailable or fails.
    """
    if not skills or embedder is None or len(skills) <= _SUBCLUSTER_THRESHOLD:
        return [list(skills)]
    try:
        from sklearn.cluster import AgglomerativeClustering
    except Exception:
        # sklearn not installed — graceful no-op.
        return [list(skills)]
    try:
        emb = embedder.encode(
            [str(s).strip() for s in skills],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        n_clusters = max(2, len(skills) // target_size)
        n_clusters = min(n_clusters, len(skills))
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(emb)
        groups: Dict[int, List[str]] = {}
        for s, lbl in zip(skills, labels):
            groups.setdefault(int(lbl), []).append(s)
        # Return groups in size-descending order so larger themes are processed first.
        return [g for _, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    except Exception as e:
        print(f"[WARN] Sub-clustering of unrelated skills failed ({e}); using a single group.")
        return [list(skills)]


def build_domain_batches(
    skills_sorted: List[str],
    skill_to_domain: Dict[str, str],
    skill_to_domain_data: Dict[str, Dict],
    max_per_batch: int,
    domain_order: str = "mean_future_weight",
    embedder=None,
) -> List[Tuple[str, List[str]]]:
    """
    Group skills by domain and chunk within each domain.
    Returns list of (domain_name, [skills]) tuples.
    Preserves sort order within each domain.

    domain_order: "mean_future_weight" (desc) or "trend_first" (Strong_Growth first).

    embedder: optional SBERT model. When provided, the "Uncertain" and "Unmapped"
    pseudo-domains are sub-clustered into thematic sub-groups (so unrelated skills
    do not end up in the same LLM call, which used to produce inaccurate competencies).
    """
    from collections import defaultdict

    domain_to_skills: Dict[str, List[str]] = defaultdict(list)
    for s in skills_sorted:
        domain = skill_to_domain.get(s.strip(), "Unmapped")
        domain_to_skills[domain].append(s)

    # Order domains
    if domain_order == "trend_first":
        domain_scores = {}
        for domain, skills in domain_to_skills.items():
            trends = [
                skill_to_domain_data.get(s.strip(), {}).get("trend_label", "")
                for s in skills
            ]
            best_trend = max(
                (TREND_ORDER.get(t, -1) for t in trends if t),
                default=-1,
            )
            domain_scores[domain] = (best_trend, -len(skills))  # secondary: more skills first
        sorted_domains = sorted(
            domain_to_skills.keys(),
            key=lambda d: domain_scores.get(d, (-1, 0)),
            reverse=True,
        )
    else:
        # mean_future_weight desc
        domain_scores = {}
        for domain, skills in domain_to_skills.items():
            fws = [
                skill_to_domain_data.get(s.strip(), {}).get("future_weight", -999)
                for s in skills
            ]
            domain_scores[domain] = sum(fws) / len(fws) if fws else -999
        sorted_domains = sorted(
            domain_to_skills.keys(),
            key=lambda d: domain_scores.get(d, -999),
            reverse=True,
        )

    result: List[Tuple[str, List[str]]] = []
    for domain in sorted_domains:
        skills = domain_to_skills[domain]
        # For Uncertain / Unmapped, sub-cluster first so each LLM call sees a thematic group.
        if domain in UNRELATED_DOMAINS and embedder is not None:
            sub_groups = subcluster_unrelated_skills(skills, embedder)
            for k, sub in enumerate(sub_groups):
                label = f"{domain} (cluster {k+1})" if len(sub_groups) > 1 else domain
                for i in range(0, len(sub), max_per_batch):
                    chunk = sub[i : i + max_per_batch]
                    result.append((label, chunk))
            if len(sub_groups) > 1:
                print(
                    f"[INFO] Sub-clustered {len(skills)} '{domain}' skills "
                    f"into {len(sub_groups)} thematic sub-groups."
                )
        else:
            for i in range(0, len(skills), max_per_batch):
                chunk = skills[i : i + max_per_batch]
                result.append((domain, chunk))
    return result


def merge_similar_domain_batches(
    batches: List[Tuple[str, List[str]]],
    domain_embeddings: Dict[str, np.ndarray],
    threshold: float,
    max_per_batch: int = 30,
) -> List[Tuple[str, List[str]]]:
    """
    Merge domain batches only when their domains are strongly similar
    (cosine similarity >= threshold). After merging, re-chunk any batch
    that exceeds max_per_batch to keep LLM calls manageable.
    """
    if not batches or threshold <= 0:
        return batches
    if len(batches) <= 1:
        return batches

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    domain_list = list({b[0] for b in batches})
    n = len(domain_list)
    parent = list(range(n))
    domain_to_id = {d: i for i, d in enumerate(domain_list)}

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            di, dj = domain_list[i], domain_list[j]
            if di in ("Unmapped", "Uncertain") or dj in ("Unmapped", "Uncertain"):
                continue
            emb_i = domain_embeddings.get(di)
            emb_j = domain_embeddings.get(dj)
            if emb_i is not None and emb_j is not None:
                sim = cos_sim(emb_i, emb_j)
                if sim >= threshold:
                    union(i, j)

    clusters: Dict[int, List[Tuple[str, List[str]]]] = {}
    for domain, skills in batches:
        idx = domain_to_id.get(domain, -1)
        if idx < 0:
            clusters.setdefault(-1, []).append((domain, skills))
        else:
            root = find(idx)
            clusters.setdefault(root, []).append((domain, skills))

    result = []
    for root, cluster_batches in clusters.items():
        merged_skills = []
        merged_domain = cluster_batches[0][0]
        for _, skills in cluster_batches:
            merged_skills.extend(skills)
        # Re-chunk to max_per_batch so LLM calls stay manageable
        for i in range(0, len(merged_skills), max_per_batch):
            result.append((merged_domain, merged_skills[i : i + max_per_batch]))

    return result
