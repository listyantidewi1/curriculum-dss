"""smoke_occupation_mapping.py — apply the v9.10 occupation mapper to an
existing run's competencies.json without re-running the full pipeline.
Verifies that:

  - Software-engineering competencies map to the expected SKKNI occupations
    (Programmer, Software Developer, System Analyst, etc.).
  - Niche competencies (mentorship, secure delivery leadership) still find
    a sensible occupation home OR honestly produce an empty match list.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from occupation_mapper import load_occupations, map_competencies


def main() -> int:
    run = PROJECT_ROOT / "results" / "competency_v2_pipeline_n1k_v9_full"
    cp = run / "competencies.json"
    if not cp.exists():
        print(f"No competencies.json at {cp} — run the v9 pipeline first.")
        return 1
    competencies = json.loads(cp.read_text(encoding="utf-8"))
    occupations = load_occupations()
    map_competencies(competencies, occupations, top_k=3, min_cosine=0.40)

    print(f"{len(competencies)} competencies, {len(occupations)} SKKNI occupations")
    print()
    n_mapped = sum(1 for c in competencies if c.get("occupation_matches"))
    print(f"Mapped to >=1 occupation: {n_mapped}/{len(competencies)}")
    print()
    for c in sorted(competencies, key=lambda x: -x.get("priority_score", 0)):
        title = (c.get("title") or "?")[:55]
        matches = c.get("occupation_matches") or []
        if matches:
            mtxt = " | ".join(f"{m['occupation_en']} ({m['cosine']:.2f})" for m in matches[:3])
        else:
            mtxt = "(no match >= 0.45)"
        print(f"  {title}")
        print(f"     -> {mtxt}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
