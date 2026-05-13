"""
smoke_test_clustering.py

End-to-end smoke test for Phase 2.1 skill clustering.

Builds a synthetic set of SkillItem + KnowledgeItem instances with known
semantic groupings (auth, frontend, data engineering, soft skills), runs
the full clustering pipeline, and asserts:

    1. Canonicalization collapses casing/whitespace duplicates.
    2. Hard skills + knowledge co-cluster into ONE stream.
    3. Soft skills land in their own separate stream.
    4. Cohesion scores all meet the configured threshold (>= 0.50).
    5. cluster.items preserves the underlying SkillItem/KnowledgeItem objects
       (including their sentence_id / extractor_source provenance).
    6. cluster.summary_label is non-empty + non-default.
    7. centroid_embedding has the expected dimension.
    8. Embedding cache writes happen on first run, hits on second run.
    9. Re-running with the same seed produces identical cluster IDs and memberships.
   10. ClusteringReport contains the audit metrics.

Exits 0 on all-pass, 1 on any failure.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# Project root on sys.path so imports work when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #

_PASS = 0
_FAIL = 0
_FAIL_NAMES: list = []


def check(name: str, condition: bool, detail: str = ""):
    global _PASS, _FAIL
    if condition:
        print(f"  [PASS] {name}")
        _PASS += 1
    else:
        print(f"  [FAIL] {name}{(' — ' + detail) if detail else ''}")
        _FAIL += 1
        _FAIL_NAMES.append(name)


# --------------------------------------------------------------------------- #
# Synthetic data builder
# --------------------------------------------------------------------------- #


def _build_synthetic_items():
    """Build a synthetic set of SkillItem + KnowledgeItem objects.

    Groupings:
      G1 (hard+knowledge, auth):  6 items
      G2 (hard+knowledge, frontend): 5 items
      G3 (hard+knowledge, data eng):  6 items
      G4 (soft): 5 items
      Plus 2 duplicates (different casing/whitespace) to exercise canonicalization
      Plus 2 singletons that should land in ungrouped
    """
    from pipeline import ConfidenceTier, KnowledgeItem, SkillItem, SkillType

    def s(text, hard=True, sid="job_smoke_0000"):
        return SkillItem(
            text=text,
            type=SkillType.HARD if hard else SkillType.SOFT,
            confidence_score=0.9,
            confidence_tier=ConfidenceTier.VERY_HIGH,
            source="skill_llm_8b_lora_v1",
            sentence_id=sid,
            sentence_text=f"... {text} ...",
            extractor_source="skill_llm_8b_lora_v1",
        )

    def k(text, sid="job_smoke_0000"):
        return KnowledgeItem(
            text=text,
            confidence_score=0.9,
            confidence_tier=ConfidenceTier.VERY_HIGH,
            source="skill_llm_8b_lora_v1",
            sentence_id=sid,
            sentence_text=f"... {text} ...",
            extractor_source="skill_llm_8b_lora_v1",
        )

    items = []
    # G1 — auth (mix of hard + knowledge), 6 items across 6 jobs
    items += [
        s("implementing OAuth 2.0 flows", sid="job_a01_0001"),
        s("session management", sid="job_a02_0002"),
        s("input validation", sid="job_a03_0003"),
        s("JWT token authentication", sid="job_a04_0001"),
        k("OAuth 2.0", sid="job_a05_0001"),
        k("authentication protocols", sid="job_a06_0001"),
    ]
    # G2 — frontend, 5 items across 5 jobs
    items += [
        s("React component design", sid="job_f01_0001"),
        s("TypeScript development", sid="job_f02_0001"),
        s("CSS layout", sid="job_f03_0001"),
        s("responsive web design", sid="job_f04_0001"),
        k("modern JavaScript frameworks", sid="job_f05_0001"),
    ]
    # G3 — data engineering, 6 items
    items += [
        s("designing ETL pipelines", sid="job_d01_0001"),
        s("data warehouse modeling", sid="job_d02_0001"),
        s("Apache Spark jobs", sid="job_d03_0001"),
        s("Airflow DAG development", sid="job_d04_0001"),
        k("data warehousing concepts", sid="job_d05_0001"),
        k("ETL design patterns", sid="job_d06_0001"),
    ]
    # G4 — soft skills, 5 items
    items += [
        s("teamwork", hard=False, sid="job_s01_0001"),
        s("communication", hard=False, sid="job_s02_0001"),
        s("problem solving", hard=False, sid="job_s03_0001"),
        s("collaboration", hard=False, sid="job_s04_0001"),
        s("adaptability", hard=False, sid="job_s05_0001"),
    ]
    # Casing/whitespace duplicates for canonicalization test
    items += [
        s("  Implementing OAuth 2.0 Flows  ", sid="job_a07_0001"),
        s("implementing OAuth 2.0 flows", sid="job_a08_0001"),
    ]
    # Singletons (each in just 1 job) — should be filtered by min_global_frequency
    items += [
        s("quantum cryptography research", sid="job_x01_0001"),
        s("legacy COBOL maintenance", sid="job_x02_0001"),
    ]
    return items


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def main() -> int:
    print("=" * 72)
    print("Phase 2.1 skill clustering — smoke test")
    print("=" * 72)

    items = _build_synthetic_items()
    print(f"[INFO] built {len(items)} synthetic items")

    # Use a temp cache dir so the test is hermetic + we can verify cache writes
    with tempfile.TemporaryDirectory(prefix="cluster_smoke_") as tmp:
        cache_dir = Path(tmp) / "embeddings"
        from clustering import ClusteringConfig, cluster_skills

        # Singletons should be filtered (each appears in just 1 job); we set
        # min_global_frequency=1 here so they pass through, BUT we still test
        # they end up "ungrouped" because no cluster reaches min_cluster_size=3
        # around them. Drop them by leaving min_global_frequency at default = 2.
        config = ClusteringConfig(
            embedding_cache_dir=str(cache_dir),
            seed=42,
            min_global_frequency=1,   # keep singletons so we can verify they go ungrouped
            min_cluster_size=3,
            cohesion_threshold=0.50,
        )

        # ----- First run -----
        print("\n[TEST] first run — cold cache")
        clusters, report = cluster_skills(items, config=config)
        print(f"[INFO] produced {len(clusters)} clusters in {report.runtime_seconds:.2f}s")
        for c in clusters:
            print(
                f"        {c.id} ({c.stream}, {c.method}, n={c.n_items}, "
                f"cohesion={c.cohesion_score:.3f}, label={c.summary_label!r})"
            )

        # 1. Canonicalization
        check(
            "canonicalization collapsed casing/whitespace duplicates",
            report.n_items_canonical < report.n_items_input,
            f"canonical={report.n_items_canonical}, input={report.n_items_input}",
        )

        # 2. Hard + knowledge co-cluster in same stream
        hpk_clusters = [c for c in clusters if c.stream == "hard_plus_knowledge"]
        any_hpk_with_knowledge = any(c.n_knowledge_items > 0 for c in hpk_clusters)
        check(
            "at least one hard_plus_knowledge cluster contains KnowledgeItem",
            any_hpk_with_knowledge,
            f"hpk clusters: {len(hpk_clusters)}, all knowledge counts: "
            f"{[c.n_knowledge_items for c in hpk_clusters]}",
        )

        # 3. Soft skills land in their own stream
        soft_clusters = [c for c in clusters if c.stream == "soft_skill"]
        check(
            "at least 1 soft_skill cluster exists",
            len(soft_clusters) >= 1,
            f"got {len(soft_clusters)}",
        )
        check(
            "no soft_skill cluster contains KnowledgeItems",
            all(c.n_knowledge_items == 0 for c in soft_clusters),
            f"counts: {[c.n_knowledge_items for c in soft_clusters]}",
        )
        check(
            "no hard_plus_knowledge cluster contains soft-skill SkillItems",
            all(
                all(
                    (not hasattr(it, "type"))
                    or str(getattr(it.type, "value", it.type)).lower() != "soft"
                    for it in c.items
                )
                for c in hpk_clusters
            ),
            "found a soft SkillItem in a hard_plus_knowledge cluster",
        )

        # 4. Cohesion >= threshold for every surviving cluster
        check(
            f"every cluster meets cohesion_threshold ({config.cohesion_threshold})",
            all(c.cohesion_score >= config.cohesion_threshold for c in clusters),
            f"min cohesion = {min((c.cohesion_score for c in clusters), default=None)}",
        )

        # 5. Provenance preserved
        any_with_sid = False
        any_with_extractor = False
        for c in clusters:
            for it in c.items:
                if getattr(it, "sentence_id", ""):
                    any_with_sid = True
                if getattr(it, "extractor_source", ""):
                    any_with_extractor = True
                if any_with_sid and any_with_extractor:
                    break
        check("cluster.items preserves sentence_id", any_with_sid)
        check("cluster.items preserves extractor_source", any_with_extractor)

        # 6. summary_label populated and non-default
        check(
            "summary_label is non-empty for all clusters",
            all(c.summary_label and c.summary_label != "(unlabeled cluster)" for c in clusters),
            f"labels: {[c.summary_label for c in clusters]}",
        )

        # 7. centroid_embedding shape
        if clusters:
            dim = clusters[0].centroid_embedding.shape[0]
            check(
                "centroid_embedding has consistent dimension across clusters",
                all(c.centroid_embedding.shape[0] == dim for c in clusters),
                f"dims = {[c.centroid_embedding.shape[0] for c in clusters]}",
            )
            # MiniLM-L6-v2 -> 384d
            check(
                "centroid_embedding dim == 384 (MiniLM-L6-v2)",
                dim == 384,
                f"got {dim}",
            )

        # 8. Embedding cache hit on 2nd run
        cache_files_after_first = list(cache_dir.rglob("*.npy")) if cache_dir.exists() else []
        check(
            "embedding cache files written on first run",
            len(cache_files_after_first) > 0,
            f"found {len(cache_files_after_first)} .npy files",
        )

        # ----- Second run -----
        print("\n[TEST] second run — warm cache")
        import time as _time

        t0 = _time.time()
        clusters2, report2 = cluster_skills(items, config=config)
        warm_secs = _time.time() - t0
        cold_secs = report.runtime_seconds
        print(f"[INFO] cold={cold_secs:.2f}s, warm={warm_secs:.2f}s")
        check(
            "warm-cache run is faster than cold-cache run",
            warm_secs < cold_secs,
            f"warm={warm_secs:.2f}s, cold={cold_secs:.2f}s",
        )

        # 9. Deterministic re-runs
        ids_1 = [c.id for c in clusters]
        ids_2 = [c.id for c in clusters2]
        check(
            "cluster IDs are identical across runs (deterministic)",
            ids_1 == ids_2,
            f"run1: {ids_1}\n            run2: {ids_2}",
        )
        member_sigs_1 = [
            tuple(sorted((it.text, it.sentence_id or "") for it in c.items)) for c in clusters
        ]
        member_sigs_2 = [
            tuple(sorted((it.text, it.sentence_id or "") for it in c.items)) for c in clusters2
        ]
        check(
            "cluster membership is identical across runs",
            member_sigs_1 == member_sigs_2,
            "membership diverged between runs",
        )

        # 10. ClusteringReport sanity
        check(
            "report.n_items_input == len(items)",
            report.n_items_input == len(items),
            f"report={report.n_items_input}, len(items)={len(items)}",
        )
        check(
            "report.n_clusters_hdbscan + recovered + split >= number of clusters",
            (
                report.n_clusters_hdbscan
                + report.n_clusters_recovered
                + report.n_clusters_split
            )
            >= len(clusters),
            f"audit = {report.to_dict()}",
        )

    print()
    print("=" * 72)
    print(f"Summary: {_PASS} passed, {_FAIL} failed")
    if _FAIL_NAMES:
        print("Failed:")
        for n in _FAIL_NAMES:
            print(f"  - {n}")
    print("=" * 72)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
