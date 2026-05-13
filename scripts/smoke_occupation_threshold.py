"""Calibrate the occupation match threshold — for each unmatched competency
in v9, show the top 3 ICT-sector candidates regardless of threshold so we
can pick a defensible default."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from occupation_mapper import load_occupations, _embed

import numpy as np


def main() -> int:
    run = PROJECT_ROOT / "results" / "competency_v2_pipeline_n1k_v9_full"
    comps = json.loads((run / "competencies.json").read_text(encoding="utf-8"))
    occupations = [
        o for o in load_occupations()
        if o.sector == "Teknologi Informasi dan Komunikasi"
    ]
    print(f"{len(occupations)} ICT occupations")
    occ_texts = [o.embedding_text() for o in occupations]
    occ_emb = _embed(occ_texts)
    comp_texts = []
    for c in comps:
        parts = [c.get("title", ""), c.get("description", "")]
        rs = c.get("related_skills") or []
        if rs:
            parts.append("Related skills: " + ", ".join(rs[:10]))
        comp_texts.append(". ".join(p for p in parts if p))
    comp_emb = _embed(comp_texts)
    sims = comp_emb @ occ_emb.T
    for i, c in enumerate(comps):
        title = (c.get("title") or "?")[:55]
        row = sims[i]
        order = np.argsort(-row)
        top = [(occupations[j].name_en, float(row[j])) for j in order[:3]]
        print(f"\n  {title}")
        for name, s in top:
            print(f"    {s:.3f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
