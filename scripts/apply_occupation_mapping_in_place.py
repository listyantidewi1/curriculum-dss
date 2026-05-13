"""Augment an existing run's competencies.json with `occupation_matches`
via the v9.10 mapper. Cheap — SBERT-only, no LLM calls. Use when the run
was generated before `--enable-occupation-mapping` was wired in.

Usage:
    python scripts/apply_occupation_mapping_in_place.py results/competency_v2_pipeline_n1k_v9_full
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from occupation_mapper import load_occupations, map_competencies


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: apply_occupation_mapping_in_place.py <run_dir>")
        return 2
    run = Path(sys.argv[1])
    if not run.is_absolute():
        run = PROJECT_ROOT / run
    cp = run / "competencies.json"
    if not cp.exists():
        print(f"No competencies.json at {cp}")
        return 1
    competencies = json.loads(cp.read_text(encoding="utf-8"))
    occupations = load_occupations()
    map_competencies(competencies, occupations)  # uses module defaults (ICT-only, cosine >= 0.40)
    cp.write_text(json.dumps(competencies, indent=2, ensure_ascii=False), encoding="utf-8")
    n_mapped = sum(1 for c in competencies if c.get("occupation_matches"))
    print(f"Augmented {len(competencies)} competencies, {n_mapped} now mapped to >=1 occupation.")
    print(f"Written: {cp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
