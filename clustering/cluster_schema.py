"""
cluster_schema.py

Dataclasses for Phase 2.1 skill clustering output.

A Cluster groups semantically related SkillItem / KnowledgeItem instances
(produced by pipeline.py Phase 1 fusion) into a single coherent unit that
Phase 2.2 (competency_generator_v2) will turn into one or more curriculum
competencies.

See docs/PHASE_2_1_DESIGN.md for the architectural rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


# Re-export aliases for type hints. We keep the actual SkillItem/KnowledgeItem
# imports inside skill_clusterer.py to avoid a circular import with pipeline.py
# at module load time.
SkillItemLike = Any  # pipeline.SkillItem
KnowledgeItemLike = Any  # pipeline.KnowledgeItem
ItemT = Union[SkillItemLike, KnowledgeItemLike]


CLUSTERER_VERSION = "v2.1.0"


@dataclass
class ClusteringConfig:
    """Tunable knobs for the clustering pipeline.

    Defaults match the v2 spec (Req 5) and pipeline.py's existing
    EMBEDDING_MODEL choice so clustering centroids live in the same
    semantic space as the rest of the pipeline.
    """

    # Embedder — match pipeline.py's AdvancedPipelineConfig.EMBEDDING_MODEL
    # so centroid_embedding is consumable by downstream phases (e.g. Phase 2.3
    # KKNI labeller) without re-embedding.
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: str = "cache/embeddings"
    template_prefix: str = "skill or knowledge area: "

    # Role-context embedding (Phase 2.1 Tier 2). When enabled, the embedding
    # prefix is rendered per-item using the modal job_title across the
    # canonical item's source postings, so the SAME surface text gets
    # different embeddings in different role contexts. Falls back to
    # `template_prefix` when an item has no job_title.
    enable_role_context: bool = False
    role_context_template: str = "skill in role '{job_title}': "

    # LLM-as-arbiter cluster refinement (Phase 2.1 v8 sprint). When enabled,
    # an LLM is asked to verdict each surviving cluster's members (keep /
    # move_out) plus the top-K SBERT-close noise candidates (join / skip).
    # Cluster cohesion must improve monotonically for a move to commit.
    # See clustering/cluster_refiner.py for the full algorithm.
    enable_cluster_refinement: bool = False
    refinement_model: str = "gpt-5.4-mini"
    refinement_max_iterations: int = 2
    refinement_n_noise_candidates: int = 10
    refinement_noise_candidate_min_cosine: float = 0.40

    # v8.1 sprint: post-refinement second recovery pass. After the LLM
    # arbiter moves items into the noise pool, run agglomerative recovery
    # one more time on the enlarged noise pool — items dropped together by
    # different clusters may form a coherent new cluster. Tighter cohesion
    # threshold than the first pass to prevent a "junk drawer" cluster.
    enable_post_refinement_recovery: bool = True
    post_refinement_recovery_cohesion: float = 0.65

    # Item-level frequency filter
    min_global_frequency: int = 2          # drop items appearing in < N unique jobs

    # HDBSCAN
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.15
    cluster_selection_method: str = "eom"  # excess-of-mass

    # Cohesion gate (Req 5). 0.55 chosen via sweep on real Phase 1 output
    # (results/clustering_sweep/, 2026-05-12). At 0.55 the two incoherent
    # clusters that mixed unrelated domains drop out; every surviving cluster
    # has cohesion >= 0.601 (0.05 safety margin). Req 5 spec floor remains 0.50.
    cohesion_threshold: float = 0.55

    # Cluster-size guard rails
    max_cluster_size: int = 25             # split clusters larger than this
    max_split_recursion: int = 3           # cap on recursive splitting

    # Recovery / split toggles
    enable_agglomerative_recovery: bool = True
    enable_oversize_split: bool = True

    # Naming
    naming_max_unigrams: int = 5
    naming_max_bigrams: int = 3
    naming_max_trigrams: int = 2

    # Reproducibility
    seed: int = 42


@dataclass
class Cluster:
    """A coherent group of SkillItem / KnowledgeItem instances."""

    # Identity
    id: str                                 # e.g. "cluster_h0042" (h = hard+knowledge) or "cluster_s0007" (s = soft)
    stream: str                             # "hard_plus_knowledge" | "soft_skill"
    method: str                             # "hdbscan" | "agglomerative_recovery" | "agglomerative_split"

    # Members
    items: List[ItemT]
    n_items: int
    n_unique_jobs: int
    n_skill_items: int                      # how many of items are SkillItem
    n_knowledge_items: int                  # how many are KnowledgeItem

    # Quality
    cohesion_score: float                   # mean pairwise cosine inside the cluster
    cohesion_std: float                     # std of those pairwise cosines
    centroid_embedding: np.ndarray          # mean of item embeddings; same dim as embedder

    # Naming
    summary_label: str                      # human-readable, ~3-6 words
    top_terms: List[str]                    # top TF-IDF terms (unigrams + bigrams + trigrams)

    # Provenance / reproducibility
    seed: int
    embedder_model: str
    clusterer_version: str = CLUSTERER_VERSION

    # Optional: parent cluster id when this cluster was produced by oversize split
    parent_cluster_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for JSON / CSV / audit logs.

        Embedding array is NOT included by default — it can be ~3KB per cluster
        and is rarely useful in plaintext logs. Call to_dict_with_embedding()
        when persisting for downstream consumers (Phase 2.3 KKNI labeller).
        """
        return {
            "id": self.id,
            "stream": self.stream,
            "method": self.method,
            "n_items": self.n_items,
            "n_unique_jobs": self.n_unique_jobs,
            "n_skill_items": self.n_skill_items,
            "n_knowledge_items": self.n_knowledge_items,
            "cohesion_score": round(float(self.cohesion_score), 4),
            "cohesion_std": round(float(self.cohesion_std), 4),
            "summary_label": self.summary_label,
            "top_terms": list(self.top_terms),
            "items": [getattr(it, "text", str(it)) for it in self.items],
            "seed": self.seed,
            "embedder_model": self.embedder_model,
            "clusterer_version": self.clusterer_version,
            "parent_cluster_id": self.parent_cluster_id,
        }

    def to_dict_with_embedding(self) -> dict:
        """Same as to_dict() but includes centroid_embedding as a list of floats."""
        d = self.to_dict()
        d["centroid_embedding"] = [float(x) for x in self.centroid_embedding.tolist()]
        return d


@dataclass
class ClusteringReport:
    """Audit-friendly summary of one clustering run.

    Phase 2.1 returns (clusters, report). The report is what gets logged for
    the paper / stability experiment; the clusters get passed to Phase 2.2.
    """

    n_items_input: int                          # before canonicalization
    n_items_canonical: int                      # after canonicalization
    n_items_after_frequency_filter: int         # after min_global_frequency
    n_items_hard_plus_knowledge: int
    n_items_soft: int

    n_clusters_hdbscan: int                     # produced by HDBSCAN primary pass
    n_clusters_recovered: int                   # produced by agglomerative recovery
    n_clusters_split: int                       # produced by oversize split
    n_clusters_dropped_low_cohesion: int        # below cohesion_threshold
    n_items_ungrouped: int                      # final noise floor

    cohesion_mean: float                        # across all surviving clusters
    cohesion_median: float
    cohesion_min: float                         # should be >= cohesion_threshold

    runtime_seconds: float
    config: ClusteringConfig = field(default=None)  # type: ignore

    # LLM-as-arbiter refinement audit (Phase 2.1 v8). Populated when
    # ClusteringConfig.enable_cluster_refinement = True. None otherwise.
    refinement: Optional[dict] = None

    # v8.1: noise-audit dict (see clustering/noise_audit.py). Captures every
    # item dropped at every Phase 2.1 stage with text + stage + reason.
    # Persisted separately to `noise_audit.json` by the pipeline runner, but
    # also embedded here so the dashboard can load it from one source.
    noise_audit: Optional[dict] = None

    def to_dict(self) -> dict:
        # Convert config to dict if present
        cfg = self.config
        cfg_dict = None
        if cfg is not None:
            cfg_dict = {k: v for k, v in cfg.__dict__.items()}
        return {
            "n_items_input": self.n_items_input,
            "n_items_canonical": self.n_items_canonical,
            "n_items_after_frequency_filter": self.n_items_after_frequency_filter,
            "n_items_hard_plus_knowledge": self.n_items_hard_plus_knowledge,
            "n_items_soft": self.n_items_soft,
            "n_clusters_hdbscan": self.n_clusters_hdbscan,
            "n_clusters_recovered": self.n_clusters_recovered,
            "n_clusters_split": self.n_clusters_split,
            "n_clusters_dropped_low_cohesion": self.n_clusters_dropped_low_cohesion,
            "n_items_ungrouped": self.n_items_ungrouped,
            "cohesion_mean": round(float(self.cohesion_mean), 4),
            "cohesion_median": round(float(self.cohesion_median), 4),
            "cohesion_min": round(float(self.cohesion_min), 4),
            "runtime_seconds": round(float(self.runtime_seconds), 2),
            "config": cfg_dict,
            "refinement": self.refinement,
            "noise_audit_summary": (
                {
                    "n_items_dropped_total": self.noise_audit.get("n_items_dropped_total", 0),
                    "by_stage": self.noise_audit.get("by_stage", {}),
                }
                if self.noise_audit else None
            ),
        }
