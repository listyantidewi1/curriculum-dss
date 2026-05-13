"""Smoke-test the v8.1 noise audit by running ONLY the clustering stage on
the n1k corpus and dumping the audit shape. No LLM generation needed.

Verifies:
  - canonicalization drops are captured (generic_single_token)
  - frequency-filter drops are captured (below_frequency)
  - HDBSCAN noise is captured (hdbscan_noise or post_refinement_noise)
  - cluster-dropped-low-cohesion drops are captured (if any)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_clustering_on_real_data import load_real_items
from clustering import ClusteringConfig, cluster_skills


def main() -> int:
    in_dir = PROJECT_ROOT / "DATA" / "preprocessing" / "phase1_n1k"
    jobs_meta = PROJECT_ROOT / "DATA" / "preprocessing" / "data_prepared_n1k" / "jobs_metadata.csv"
    print(f"Loading items from {in_dir}...")
    items = load_real_items(in_dir, jobs_metadata_csv=jobs_meta)
    print(f"Loaded {len(items)} items")

    # Run clustering WITHOUT refinement (refinement requires LLM calls — we
    # want a fast smoke that exercises the tracker).
    cfg = ClusteringConfig(enable_role_context=True, enable_cluster_refinement=False)
    clusters, report = cluster_skills(items, config=cfg)

    audit = report.noise_audit or {}
    print(f"\nNoise audit summary:")
    print(f"  total dropped: {audit.get('n_items_dropped_total', 0)}")
    print(f"  by stage: {audit.get('by_stage', {})}")
    print()
    print("Sample dropped items per stage:")
    by_stage_items: dict = {}
    for it in audit.get("items", []):
        by_stage_items.setdefault(it.get("stage"), []).append(it)
    for stage, items in by_stage_items.items():
        print(f"\n  [{stage}] ({len(items)} items)")
        for it in items[:5]:
            print(f"    - {it.get('text','?')[:60]:60s} (in {it.get('n_unique_jobs',0)} jobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
