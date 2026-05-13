"""
clustering — Phase 2.1 skill clustering package.

See docs/PHASE_2_1_DESIGN.md.

Public entry point:

    from clustering import cluster_skills, ClusteringConfig
    clusters, report = cluster_skills(items, config=ClusteringConfig())
"""
from .cluster_schema import (
    CLUSTERER_VERSION,
    Cluster,
    ClusteringConfig,
    ClusteringReport,
)
from .skill_clusterer import cluster_skills

__all__ = [
    "CLUSTERER_VERSION",
    "Cluster",
    "ClusteringConfig",
    "ClusteringReport",
    "cluster_skills",
]
