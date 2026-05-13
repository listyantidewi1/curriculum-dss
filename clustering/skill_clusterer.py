"""
skill_clusterer.py

Phase 2.1 entry point: cluster SkillItem + KnowledgeItem instances produced
by Phase 1 fusion into semantically coherent Cluster objects ready for
Phase 2.2 competency generation.

See docs/PHASE_2_1_DESIGN.md for the architecture and rationale.

Top-level entry point:

    from clustering import cluster_skills, ClusteringConfig
    clusters, report = cluster_skills(items, config=ClusteringConfig())

`items` is a list of SkillItem | KnowledgeItem instances. The clusterer
produces a list of Cluster objects and a ClusteringReport with audit metrics.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cluster_naming import label_cluster
from .cluster_schema import (
    CLUSTERER_VERSION,
    Cluster,
    ClusteringConfig,
    ClusteringReport,
    ItemT,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Item helpers — work duck-typed so we don't have to import pipeline.py
# (which would create a circular import on package load)
# --------------------------------------------------------------------------- #


def _item_text(item: ItemT) -> str:
    return getattr(item, "text", "")


def _item_is_skill(item: ItemT) -> bool:
    """True for SkillItem, False for KnowledgeItem.

    Discriminator: SkillItem has a `type` attr (SkillType.HARD or SkillType.SOFT);
    KnowledgeItem does not.
    """
    return hasattr(item, "type")


def _item_is_hard_skill(item: ItemT) -> bool:
    if not _item_is_skill(item):
        return False
    t = getattr(item, "type", None)
    if t is None:
        return False
    val = getattr(t, "value", str(t))
    return str(val).lower() == "hard"


def _item_is_soft_skill(item: ItemT) -> bool:
    if not _item_is_skill(item):
        return False
    t = getattr(item, "type", None)
    if t is None:
        return False
    val = getattr(t, "value", str(t))
    return str(val).lower() == "soft"


def _item_sentence_id(item: ItemT) -> str:
    return getattr(item, "sentence_id", None) or ""


def _job_id_from_sentence_id(sid: str) -> str:
    """Heuristic: strip a trailing `_<digits>` suffix to recover the job_id.

    Pipeline convention: sentence_ids look like "job_smoke_0000" or "job123_0001".
    If the trailing segment after the last `_` is all digits, treat everything
    before it as the job_id. Otherwise, fall back to the sentence_id itself.
    """
    if not sid:
        return ""
    if "_" not in sid:
        return sid
    head, _sep, tail = sid.rpartition("_")
    return head if tail.isdigit() else sid


# --------------------------------------------------------------------------- #
# Canonicalization — collapse text variants and merge provenance
# --------------------------------------------------------------------------- #


def _canonical_key(text: str) -> str:
    """Lowercase + strip + whitespace-collapse. Used as the dedup key."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


# Single-token "skills" that carry no real signal — they're either fragment
# words the extractor latched onto from a longer phrase, or overly generic
# topics that mean nothing in isolation. Dropping these tightens
# related_skills lists noticeably without losing real content. The list is
# intentionally small + obvious; expand only when a clear bad pattern recurs.
_GENERIC_SINGLE_TOKEN_SKILLS = frozenset({
    "platform", "platforms",
    "code", "codes", "coding",
    "engineering", "engineer",
    "tool", "tools", "tooling",
    "system", "systems",
    "technology", "technologies", "tech",
    "software", "hardware",
    "development", "developer",
    "design", "designs",
    "data",
    "service", "services",
    "solution", "solutions",
    "application", "applications", "app", "apps",
    "framework", "frameworks",
    "language", "languages",
    "method", "methods",
    "process", "processes",
    "skill", "skills",
    "knowledge",
    "experience",
    "concept", "concepts",
    "principle", "principles",
    # Tier 2D additions (2026-05-13): generic adjective/abstract tokens that
    # historically pulled unrelated postings into clusters (e.g. "organizational"
    # pulled a church A/V technician into the Azure-DevOps cluster).
    "organizational", "organization", "organize",
    "maintenance", "maintain",
    "presentation", "presentations",
    "communication", "communications",  # soft-skill mention; clusterer should not pivot on it
    "collaboration", "collaborate",
    "operational", "operations",
    "modern",
    "advanced",
    "various", "varied",
})


def _is_too_generic_skill(text: str) -> bool:
    """Return True if `text` is a single-token, semantically empty skill."""
    if not text:
        return True
    norm = re.sub(r"\s+", " ", text.strip().lower())
    if " " in norm:
        return False  # multi-token phrases keep their context, allow them
    return norm in _GENERIC_SINGLE_TOKEN_SKILLS


class _CanonicalItem:
    """Lightweight container that holds a canonicalized SkillItem/KnowledgeItem.

    Carries:
      - canonical text (the most-frequent surface form among merged items)
      - merged sentence_ids / sentence_texts / extractor_sources / job_ids
      - n_sources: number of underlying raw items
      - is_skill: True if any underlying item is a SkillItem
      - is_hard / is_soft: True if any underlying SkillItem is HARD / SOFT
      - originals: the list of underlying raw items (so we can rehydrate
        the cluster's `items` field at the end)
    """

    __slots__ = (
        "canonical_text",
        "sentence_ids",
        "sentence_texts",
        "extractor_sources",
        "job_ids",
        "job_title_counts",
        "is_hard",
        "is_soft",
        "is_knowledge",
        "originals",
    )

    def __init__(self) -> None:
        self.canonical_text: str = ""
        self.sentence_ids: set = set()
        self.sentence_texts: set = set()
        self.extractor_sources: set = set()
        self.job_ids: set = set()
        self.job_title_counts: defaultdict = defaultdict(int)
        self.is_hard: bool = False
        self.is_soft: bool = False
        self.is_knowledge: bool = False
        self.originals: List[ItemT] = []

    def absorb(self, item: ItemT) -> None:
        self.originals.append(item)
        sid = _item_sentence_id(item)
        if sid:
            self.sentence_ids.add(sid)
            jid = _job_id_from_sentence_id(sid)
            if jid:
                self.job_ids.add(jid)
        stxt = getattr(item, "sentence_text", None)
        if stxt:
            self.sentence_texts.add(stxt)
        es = getattr(item, "extractor_source", None) or getattr(item, "source", None)
        if es:
            self.extractor_sources.add(str(es))
        jt = getattr(item, "job_title", None)
        if jt:
            self.job_title_counts[jt] += 1
        if _item_is_hard_skill(item):
            self.is_hard = True
        if _item_is_soft_skill(item):
            self.is_soft = True
        if not _item_is_skill(item):
            self.is_knowledge = True

    @property
    def modal_job_title(self) -> str:
        """Return the most-frequent job_title across this canonical item's
        underlying originals, or '' when none of the items carried a title.

        Used by the clusterer's role-context embedding to render a per-item
        prefix that reflects the dominant role for this skill.
        """
        if not self.job_title_counts:
            return ""
        return max(self.job_title_counts.items(), key=lambda kv: kv[1])[0]

    @property
    def n_sources(self) -> int:
        return len(self.originals)


def _canonicalize(
    items: Sequence[ItemT],
    noise_tracker: Optional["NoiseAuditTracker"] = None,
) -> Dict[str, _CanonicalItem]:
    """Group items by canonical text key, merging provenance.

    Picks the surface form that appears most frequently among the originals
    as the `canonical_text` (preserves the dominant casing/spelling).

    When `noise_tracker` is provided, every item dropped as a single-token
    generic skill is recorded.
    """
    buckets: Dict[str, _CanonicalItem] = {}
    n_dropped_generic = 0
    for it in items:
        text = _item_text(it)
        key = _canonical_key(text)
        if not key:
            continue
        # Drop generic single-token skills (Phase 2.1 quality filter).
        # See _GENERIC_SINGLE_TOKEN_SKILLS for the list + rationale.
        if _is_too_generic_skill(text):
            n_dropped_generic += 1
            if noise_tracker is not None:
                noise_tracker.record(
                    stage="generic_single_token",
                    text=text,
                    n_unique_jobs=0,
                )
            continue
        bucket = buckets.setdefault(key, _CanonicalItem())
        bucket.absorb(it)
    if n_dropped_generic:
        logger.info(
            "canonicalize: dropped %d single-token generic skills (e.g., 'platform', 'tool')",
            n_dropped_generic,
        )

    # Pick most-frequent surface form per bucket
    for key, bucket in buckets.items():
        counts: Dict[str, int] = defaultdict(int)
        for orig in bucket.originals:
            counts[_item_text(orig)] += 1
        bucket.canonical_text = max(counts.items(), key=lambda kv: kv[1])[0]

    return buckets


# --------------------------------------------------------------------------- #
# Embedding — SBERT + on-disk cache
# --------------------------------------------------------------------------- #


_embedder_cache: dict = {}


def _get_embedder(model_name: str):
    """Lazy-load the SBERT embedder, cache the instance for the process lifetime."""
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    # Local import — keeps clustering package importable even when ML stack absent
    from sentence_transformers import SentenceTransformer

    logger.info("loading SBERT model %s", model_name)
    model = SentenceTransformer(model_name)
    _embedder_cache[model_name] = model
    return model


def _embedding_cache_path(cache_dir: str, model_name: str, text: str) -> Path:
    h = hashlib.sha256(f"{model_name}\x00{text}".encode("utf-8")).hexdigest()
    return Path(cache_dir) / h[:2] / f"{h}.npy"


def _embed_texts(
    texts: Sequence[str],
    config: ClusteringConfig,
    prefixes: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Embed `texts` via SBERT, caching each per (model, full_input) on disk.

    Args:
        texts: surface forms to embed.
        config: ClusteringConfig (controls model + cache dir + uniform prefix).
        prefixes: optional list of per-item prefixes. When provided, the
            embedded string for item i is `prefixes[i] + texts[i]`. When None,
            falls back to the uniform `config.template_prefix`.

    Returns a (len(texts), embedding_dim) numpy array, L2-normalized.
    """
    cache_dir = Path(config.embedding_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if prefixes is None:
        templated = [config.template_prefix + t for t in texts]
    else:
        if len(prefixes) != len(texts):
            raise ValueError(
                f"prefixes len ({len(prefixes)}) must match texts len ({len(texts)})"
            )
        templated = [pre + t for pre, t in zip(prefixes, texts)]

    out: List[np.ndarray] = [None] * len(texts)  # type: ignore[list-item]
    missing_idx: List[int] = []
    for i, t in enumerate(templated):
        p = _embedding_cache_path(config.embedding_cache_dir, config.embedder_model, t)
        if p.exists():
            try:
                out[i] = np.load(p)
                continue
            except Exception:
                pass
        missing_idx.append(i)

    if missing_idx:
        model = _get_embedder(config.embedder_model)
        batch_texts = [templated[i] for i in missing_idx]
        logger.info("embedding %d items (cache miss)", len(batch_texts))
        batch = model.encode(
            batch_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for k, i in enumerate(missing_idx):
            out[i] = batch[k].astype(np.float32)
            p = _embedding_cache_path(config.embedding_cache_dir, config.embedder_model, templated[i])
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.save(p, out[i])
            except Exception as e:
                logger.warning("failed to write embedding cache for idx %d: %s", i, e)

    return np.vstack(out).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cohesion + centroid
# --------------------------------------------------------------------------- #


def _pairwise_cosine_stats(embeddings: np.ndarray) -> Tuple[float, float]:
    """Return (mean, std) of pairwise cosine similarities for the given rows.

    Embeddings must be L2-normalized (SBERT with normalize_embeddings=True).
    For L2-normalized vectors, cosine = dot product.
    Handles n=1 (returns 1.0, 0.0 by convention — single item is "perfectly cohesive").
    """
    n = embeddings.shape[0]
    if n <= 1:
        return 1.0, 0.0
    sims = embeddings @ embeddings.T
    iu = np.triu_indices(n, k=1)
    pair = sims[iu]
    return float(pair.mean()), float(pair.std())


def _centroid(embeddings: np.ndarray) -> np.ndarray:
    c = embeddings.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm
    return c.astype(np.float32)


# --------------------------------------------------------------------------- #
# Clustering — HDBSCAN + agglomerative recovery + oversize split
# --------------------------------------------------------------------------- #


def _hdbscan_labels(
    embeddings: np.ndarray,
    config: ClusteringConfig,
) -> np.ndarray:
    """Run HDBSCAN on the (n_items, d) embedding matrix.

    Returns an int array of cluster labels, with -1 indicating noise.
    """
    import hdbscan  # local import — keeps package importable without hdbscan

    n = embeddings.shape[0]
    if n < config.min_cluster_size:
        # Not enough items to form any cluster; everything is noise.
        return np.full(n, -1, dtype=int)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        metric="euclidean",  # SBERT embeddings are L2-normalized → euclidean ≡ √(2 - 2·cos)
        cluster_selection_method=config.cluster_selection_method,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
    )
    return clusterer.fit_predict(embeddings)


def _agglomerative_recovery_labels(
    embeddings: np.ndarray,
    config: ClusteringConfig,
    label_offset: int,
) -> np.ndarray:
    """Try to recover dense pockets from HDBSCAN-noise points via agglomerative clustering.

    Uses scipy hierarchical clustering with Ward linkage (Padovano-style).
    Iteratively merges down to coarser clusters; keeps the finest level at
    which the average intra-cluster cohesion >= cohesion_threshold.

    Returns an int array of cluster labels (>= label_offset for valid clusters,
    -1 for items that still don't fit).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = embeddings.shape[0]
    if n < config.min_cluster_size:
        return np.full(n, -1, dtype=int)

    # Sweep number-of-clusters from coarse to fine; pick the finest grouping
    # whose every cluster meets cohesion_threshold.
    try:
        condensed = pdist(embeddings, metric="cosine")
        Z = linkage(condensed, method="average")
    except Exception as e:
        logger.warning("agglomerative_recovery linkage failed: %s", e)
        return np.full(n, -1, dtype=int)

    out = np.full(n, -1, dtype=int)
    # Try cluster counts from 2 up to n // min_cluster_size
    max_k = max(2, n // config.min_cluster_size)
    found_any = False
    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        # For each cluster, check cohesion + min size
        keep_labels = set()
        for lbl in set(labels):
            mask = labels == lbl
            if mask.sum() < config.min_cluster_size:
                continue
            cohesion, _std = _pairwise_cosine_stats(embeddings[mask])
            if cohesion >= config.cohesion_threshold:
                keep_labels.add(lbl)
        if keep_labels:
            # Map kept labels to dense ids starting at label_offset
            next_id = label_offset
            for lbl in sorted(keep_labels):
                mask = labels == lbl
                out[mask] = next_id
                next_id += 1
            found_any = True
            break  # take the first viable k from coarse to fine

    if not found_any:
        return np.full(n, -1, dtype=int)
    return out


def _split_oversize_cluster(
    embeddings: np.ndarray,
    config: ClusteringConfig,
    depth: int = 0,
) -> List[np.ndarray]:
    """Recursively split a cluster that exceeds max_cluster_size.

    Returns a list of np.ndarray of item indices (each one a sub-cluster).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = embeddings.shape[0]
    indices = np.arange(n)

    if n <= config.max_cluster_size or depth >= config.max_split_recursion:
        return [indices]

    target_k = max(2, (n + config.max_cluster_size - 1) // config.max_cluster_size)
    try:
        condensed = pdist(embeddings, metric="cosine")
        Z = linkage(condensed, method="ward")
        labels = fcluster(Z, t=target_k, criterion="maxclust")
    except Exception as e:
        logger.warning("oversize_split failed at depth %d: %s", depth, e)
        return [indices]

    out: List[np.ndarray] = []
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        sub_idx = indices[mask]
        if len(sub_idx) <= config.max_cluster_size:
            out.append(sub_idx)
        else:
            # Recurse on the sub-cluster's embeddings
            sub_emb = embeddings[mask]
            sub_splits = _split_oversize_cluster(sub_emb, config, depth=depth + 1)
            for s in sub_splits:
                out.append(sub_idx[s])
    return out


# --------------------------------------------------------------------------- #
# Cluster assembly
# --------------------------------------------------------------------------- #


def _assemble_cluster(
    canonical_items: List[_CanonicalItem],
    member_idx: np.ndarray,
    embeddings: np.ndarray,
    background_texts: Sequence[str],
    stream: str,
    method: str,
    cluster_id: str,
    config: ClusteringConfig,
    parent_id: str = None,
) -> Cluster:
    members = [canonical_items[i] for i in member_idx]
    member_emb = embeddings[member_idx]
    cohesion, cohesion_std = _pairwise_cosine_stats(member_emb)
    centroid = _centroid(member_emb)

    # Flatten the originals back out for the cluster.items field
    all_originals: List[ItemT] = []
    for m in members:
        all_originals.extend(m.originals)

    n_skill_items = sum(1 for o in all_originals if _item_is_skill(o))
    n_knowledge_items = sum(1 for o in all_originals if not _item_is_skill(o))

    unique_jobs: set = set()
    for m in members:
        unique_jobs.update(m.job_ids)

    member_texts = [m.canonical_text for m in members]
    label, top_terms = label_cluster(
        member_texts,
        background_texts,
        max_unigrams=config.naming_max_unigrams,
        max_bigrams=config.naming_max_bigrams,
        max_trigrams=config.naming_max_trigrams,
    )

    return Cluster(
        id=cluster_id,
        stream=stream,
        method=method,
        items=all_originals,
        n_items=len(all_originals),
        n_unique_jobs=len(unique_jobs),
        n_skill_items=n_skill_items,
        n_knowledge_items=n_knowledge_items,
        cohesion_score=cohesion,
        cohesion_std=cohesion_std,
        centroid_embedding=centroid,
        summary_label=label,
        top_terms=top_terms,
        seed=config.seed,
        embedder_model=config.embedder_model,
        clusterer_version=CLUSTERER_VERSION,
        parent_cluster_id=parent_id,
    )


def _cluster_one_stream(
    canonical_items: List[_CanonicalItem],
    stream: str,
    id_prefix: str,
    config: ClusteringConfig,
    noise_tracker: Optional["NoiseAuditTracker"] = None,
) -> Tuple[List[Cluster], Dict[str, int]]:
    """Cluster one stream end-to-end. Returns (clusters, audit_counts)."""
    audit = {
        "n_hdbscan": 0,
        "n_recovered": 0,
        "n_split": 0,
        "n_dropped_low_cohesion": 0,
        "n_ungrouped": 0,
    }
    if not canonical_items:
        return [], audit

    texts = [c.canonical_text for c in canonical_items]

    # Role-context embedding (Phase 2.1 Tier 2): when enabled and each item
    # has a job_title, build a per-item prefix so the same surface text gets
    # different embeddings in different role contexts. Items without a
    # title fall back to the uniform `template_prefix`.
    prefixes: Optional[List[str]] = None
    if getattr(config, "enable_role_context", False):
        n_with_title = 0
        prefixes = []
        for ci in canonical_items:
            jt = ci.modal_job_title
            if jt:
                prefixes.append(config.role_context_template.format(job_title=jt))
                n_with_title += 1
            else:
                prefixes.append(config.template_prefix)
        logger.info(
            "role-context embedding active: %d/%d items have a job_title prefix",
            n_with_title, len(canonical_items),
        )

    embeddings = _embed_texts(texts, config, prefixes=prefixes)

    # 1) HDBSCAN primary pass
    raw_labels = _hdbscan_labels(embeddings, config)

    # 2) Compute cohesion per HDBSCAN cluster; drop those below threshold
    surviving_clusters: List[Tuple[np.ndarray, str]] = []  # (member_idx, method)
    label_set = sorted(set(raw_labels) - {-1})
    for lbl in label_set:
        member_idx = np.where(raw_labels == lbl)[0]
        cohesion, _std = _pairwise_cosine_stats(embeddings[member_idx])
        if cohesion >= config.cohesion_threshold:
            surviving_clusters.append((member_idx, "hdbscan"))
            audit["n_hdbscan"] += 1
        else:
            # Demote these items back to noise pool for the recovery pass.
            # Record each member for the noise audit so the user can see
            # which whole clusters fell out.
            for idx in member_idx:
                raw_labels[idx] = -1
                if noise_tracker is not None:
                    ci = canonical_items[idx]
                    noise_tracker.record(
                        stage="cluster_dropped_low_cohesion",
                        text=ci.canonical_text or "",
                        n_unique_jobs=len(ci.job_ids),
                        extra={"cohesion": float(cohesion)},
                    )
            audit["n_dropped_low_cohesion"] += 1

    # 3) Agglomerative recovery on noise points
    if config.enable_agglomerative_recovery:
        noise_idx = np.where(raw_labels == -1)[0]
        if len(noise_idx) >= config.min_cluster_size:
            noise_emb = embeddings[noise_idx]
            recovery_labels = _agglomerative_recovery_labels(
                noise_emb, config, label_offset=10_000
            )
            for r_lbl in sorted(set(recovery_labels) - {-1}):
                member_local = np.where(recovery_labels == r_lbl)[0]
                # map local indices back to global indices via noise_idx
                global_idx = noise_idx[member_local]
                surviving_clusters.append((global_idx, "agglomerative_recovery"))
                audit["n_recovered"] += 1
                # Mark these items no longer ungrouped
                for gi in global_idx:
                    raw_labels[gi] = 100_000  # sentinel = "recovered"

    audit["n_ungrouped"] = int((raw_labels == -1).sum())

    # 3b) LLM-as-arbiter cluster refinement (Phase 2.1 v8 sprint).
    # Runs after agglomerative recovery but BEFORE oversize split, so the
    # split operates on refined memberships. Only fires when enabled in
    # config. The refiner may move items between clusters and noise; cohesion
    # must improve monotonically for any move to commit.
    refinement_report_dict = None
    if getattr(config, "enable_cluster_refinement", False) and surviving_clusters:
        from .cluster_refiner import refine_cluster_memberships, RefinementConfig
        # Build heuristic labels per surviving cluster so the arbiter prompt
        # has a topical anchor. Cheap: use cluster_naming.label_cluster on
        # canonical texts for that cluster only.
        cluster_labels_for_refiner: List[str] = []
        background_texts = [c.canonical_text for c in canonical_items]
        for member_idx, _meth in surviving_clusters:
            member_texts = [canonical_items[i].canonical_text for i in member_idx]
            try:
                label, _terms = label_cluster(
                    member_texts,
                    background_texts,
                    max_unigrams=config.naming_max_unigrams,
                    max_bigrams=config.naming_max_bigrams,
                    max_trigrams=config.naming_max_trigrams,
                )
            except Exception:
                label = ""
            cluster_labels_for_refiner.append(label)

        refiner_cfg = RefinementConfig(
            enabled=True,
            model=getattr(config, "refinement_model", "gpt-5.4-mini"),
            max_iterations=getattr(config, "refinement_max_iterations", 2),
            n_noise_candidates=getattr(config, "refinement_n_noise_candidates", 10),
            noise_candidate_min_cosine=getattr(config, "refinement_noise_candidate_min_cosine", 0.40),
            min_cluster_size_after_moves=config.min_cluster_size,
        )
        logger.info(
            "cluster refinement (LLM-as-arbiter) starting on %d clusters",
            len(surviving_clusters),
        )
        surviving_clusters, raw_labels, refinement_report = refine_cluster_memberships(
            canonical_items=canonical_items,
            embeddings=embeddings,
            surviving_clusters=surviving_clusters,
            raw_labels=raw_labels,
            cluster_labels=cluster_labels_for_refiner,
            config=refiner_cfg,
        )
        refinement_report_dict = refinement_report.to_dict()
        logger.info(
            "cluster refinement complete: %d items moved out, %d items moved in, %d clusters dropped",
            refinement_report.total_items_moved_out,
            refinement_report.total_items_moved_in,
            refinement_report.n_clusters_dropped_below_min_size,
        )
        audit["n_ungrouped"] = int((raw_labels == -1).sum())  # refresh after moves
        audit["refinement"] = refinement_report_dict

        # Mirror refinement move-outs into the noise audit (v8.1).
        # `refinement_report.per_cluster` is a list of dicts (the
        # _ClusterAudit dataclasses get serialised before assignment).
        if noise_tracker is not None:
            for pc in refinement_report.per_cluster:
                for m in pc.get("move_decisions", []) or []:
                    if m.get("verdict") == "move_out":
                        text = m.get("text") or ""
                        if not text:
                            continue
                        noise_tracker.record(
                            stage="refinement_move_out",
                            text=text,
                            n_unique_jobs=0,
                            reason=m.get("reason") or "",
                            extra={
                                "from_cluster_idx": pc.get("cluster_idx"),
                                "initial_cohesion": pc.get("initial_cohesion"),
                                "final_cohesion": pc.get("final_cohesion"),
                            },
                        )

    # 3c) Post-refinement second-pass agglomerative recovery (v8.1).
    # When refinement is enabled AND the post-refinement recovery is on,
    # items that the LLM moved out get a second chance to form a cluster
    # together. Uses a stricter cohesion threshold than the first recovery
    # to prevent a "junk drawer" cluster of unrelated moved-out items.
    if (
        getattr(config, "enable_cluster_refinement", False)
        and getattr(config, "enable_post_refinement_recovery", True)
        and (raw_labels == -1).any()
    ):
        from copy import copy
        stricter_cfg = copy(config)
        stricter_cfg.cohesion_threshold = getattr(
            config, "post_refinement_recovery_cohesion", 0.65,
        )
        noise_idx = np.where(raw_labels == -1)[0]
        if len(noise_idx) >= config.min_cluster_size:
            noise_emb = embeddings[noise_idx]
            recovery_labels = _agglomerative_recovery_labels(
                noise_emb, stricter_cfg, label_offset=200_000
            )
            n_recovered = 0
            for r_lbl in sorted(set(recovery_labels) - {-1}):
                member_local = np.where(recovery_labels == r_lbl)[0]
                global_idx = noise_idx[member_local]
                surviving_clusters.append((global_idx, "post_refinement_recovery"))
                audit["n_recovered"] += 1
                n_recovered += 1
                for gi in global_idx:
                    raw_labels[gi] = 200_000  # sentinel = "post-refinement recovered"
            if n_recovered > 0:
                logger.info(
                    "post-refinement recovery: %d new clusters from %d noise items "
                    "(stricter cohesion >= %.2f)",
                    n_recovered, len(noise_idx), stricter_cfg.cohesion_threshold,
                )
            audit["n_ungrouped"] = int((raw_labels == -1).sum())

    # 4) Oversize split + cluster assembly
    background_texts = list(texts)
    final_clusters: List[Cluster] = []
    cluster_counter = 0

    for member_idx, method in surviving_clusters:
        if config.enable_oversize_split and len(member_idx) > config.max_cluster_size:
            local_emb = embeddings[member_idx]
            sub_splits = _split_oversize_cluster(local_emb, config)
            parent_id = f"{id_prefix}{cluster_counter:04d}"
            # If a single chunk came back, no split happened — treat normally
            if len(sub_splits) == 1:
                cid = parent_id
                final_clusters.append(
                    _assemble_cluster(
                        canonical_items,
                        member_idx,
                        embeddings,
                        background_texts,
                        stream=stream,
                        method=method,
                        cluster_id=cid,
                        config=config,
                    )
                )
                cluster_counter += 1
            else:
                # Multiple sub-clusters: assemble each as a split child
                for sub_local in sub_splits:
                    sub_global = member_idx[sub_local]
                    cid = f"{id_prefix}{cluster_counter:04d}"
                    final_clusters.append(
                        _assemble_cluster(
                            canonical_items,
                            sub_global,
                            embeddings,
                            background_texts,
                            stream=stream,
                            method="agglomerative_split",
                            cluster_id=cid,
                            config=config,
                            parent_id=parent_id,
                        )
                    )
                    cluster_counter += 1
                    audit["n_split"] += 1
        else:
            cid = f"{id_prefix}{cluster_counter:04d}"
            final_clusters.append(
                _assemble_cluster(
                    canonical_items,
                    member_idx,
                    embeddings,
                    background_texts,
                    stream=stream,
                    method=method,
                    cluster_id=cid,
                    config=config,
                )
            )
            cluster_counter += 1

    # v8.1: capture any items still ungrouped at this point. Sentence labels
    # are sentinel ints; -1 = noise / never-clustered. Tag them with the
    # appropriate stage (HDBSCAN-noise vs. post-refinement-noise) — if the
    # refinement was enabled we know the second-pass recovery already ran,
    # so anything left is post-refinement noise.
    if noise_tracker is not None:
        refinement_active = bool(getattr(config, "enable_cluster_refinement", False))
        noise_stage = "post_refinement_noise" if refinement_active else "hdbscan_noise"
        # Pre-compute a set of texts already recorded at the source stages so
        # we don't double-count them here.
        already_at_source = {
            e.text for e in noise_tracker.entries
            if e.stage in ("cluster_dropped_low_cohesion", "refinement_move_out")
        }
        for idx in np.where(raw_labels == -1)[0]:
            ci = canonical_items[idx]
            text = ci.canonical_text or ""
            if not text or text in already_at_source:
                continue
            noise_tracker.record(
                stage=noise_stage,
                text=text,
                n_unique_jobs=len(ci.job_ids),
            )

    return final_clusters, audit


# --------------------------------------------------------------------------- #
# Top-level entry
# --------------------------------------------------------------------------- #


def cluster_skills(
    items: Sequence[ItemT],
    config: ClusteringConfig = None,
) -> Tuple[List[Cluster], ClusteringReport]:
    """Phase 2.1 entry point.

    Args:
        items: SkillItem | KnowledgeItem instances from pipeline.py fusion.
        config: optional ClusteringConfig override.

    Returns:
        (clusters, report)
            clusters: List[Cluster] across both streams (hard+knowledge, soft)
            report: ClusteringReport with audit metrics
    """
    if config is None:
        config = ClusteringConfig()

    np.random.seed(config.seed)

    t0 = time.time()
    n_input = len(items)

    # v8.1: NoiseAuditTracker — captures items dropped at every stage so we
    # can surface them on the dashboard's Pipeline-audit page.
    from .noise_audit import NoiseAuditTracker
    noise_tracker = NoiseAuditTracker()

    # 1. Canonicalize
    canonical = _canonicalize(items, noise_tracker=noise_tracker)
    n_canonical = len(canonical)

    # 2. Frequency filter (drop items in fewer than min_global_frequency unique jobs)
    filtered: List[_CanonicalItem] = []
    for bucket in canonical.values():
        if len(bucket.job_ids) >= config.min_global_frequency:
            filtered.append(bucket)
        elif not bucket.job_ids and len(bucket.sentence_ids) >= config.min_global_frequency:
            # sentence-only items still pass if there's no job_id but we have multiple sentences
            filtered.append(bucket)
        else:
            # Dropped — record for the noise audit.
            noise_tracker.record(
                stage="below_frequency",
                text=bucket.canonical_text or "",
                n_unique_jobs=len(bucket.job_ids),
                extra={"n_unique_sentences": len(bucket.sentence_ids)},
            )
    n_after_freq = len(filtered)

    # 3. Stream split
    hpk_items = [c for c in filtered if c.is_hard or c.is_knowledge]
    soft_items = [c for c in filtered if c.is_soft and not c.is_hard and not c.is_knowledge]

    # 4. Cluster each stream independently
    hpk_clusters, hpk_audit = _cluster_one_stream(
        hpk_items, stream="hard_plus_knowledge", id_prefix="cluster_h",
        config=config, noise_tracker=noise_tracker,
    )
    soft_clusters, soft_audit = _cluster_one_stream(
        soft_items, stream="soft_skill", id_prefix="cluster_s",
        config=config, noise_tracker=noise_tracker,
    )

    all_clusters = hpk_clusters + soft_clusters

    # 5. Build report
    cohesion_vals = [c.cohesion_score for c in all_clusters]
    if cohesion_vals:
        cohesion_mean = float(np.mean(cohesion_vals))
        cohesion_median = float(np.median(cohesion_vals))
        cohesion_min = float(min(cohesion_vals))
    else:
        cohesion_mean = cohesion_median = cohesion_min = 0.0

    # Surface refinement audit (only on the hard+knowledge stream — soft is
    # currently empty in production; if it gets populated and refined later,
    # combine here).
    refinement_audit = hpk_audit.get("refinement") or soft_audit.get("refinement")

    report = ClusteringReport(
        n_items_input=n_input,
        n_items_canonical=n_canonical,
        n_items_after_frequency_filter=n_after_freq,
        n_items_hard_plus_knowledge=len(hpk_items),
        n_items_soft=len(soft_items),
        n_clusters_hdbscan=hpk_audit["n_hdbscan"] + soft_audit["n_hdbscan"],
        n_clusters_recovered=hpk_audit["n_recovered"] + soft_audit["n_recovered"],
        n_clusters_split=hpk_audit["n_split"] + soft_audit["n_split"],
        n_clusters_dropped_low_cohesion=(
            hpk_audit["n_dropped_low_cohesion"] + soft_audit["n_dropped_low_cohesion"]
        ),
        n_items_ungrouped=hpk_audit["n_ungrouped"] + soft_audit["n_ungrouped"],
        cohesion_mean=cohesion_mean,
        cohesion_median=cohesion_median,
        cohesion_min=cohesion_min,
        runtime_seconds=time.time() - t0,
        config=config,
        refinement=refinement_audit,
        noise_audit=noise_tracker.to_dict(),
    )

    return all_clusters, report
