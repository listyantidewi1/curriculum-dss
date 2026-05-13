"""smoke_title_evidence.py — verify the Tier 2D title-evidence audit catches
the failure modes flagged in the 2026-05-13 v6 review:

  1. "Apply integration patterns to transform data across Microsoft-based systems"
     should flag "Microsoft" / "Microsoft-based" because only 2 of 7 sentences
     literally name Microsoft as the platform.
  2. "Organize delivery workflows with Azure DevOps" should flag "Azure DevOps"
     because the church A/V technician sentence #4 doesn't contain it.
  3. "Build C# web applications with .NET Core and ASP.NET" should NOT flag
     C#, .NET Core, or ASP.NET because 6 of 7 sentences contain them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from competency_generator_v2 import _audit_title_evidence, _extract_product_terms


def main() -> int:
    run = PROJECT_ROOT / "results/competency_v2_pipeline_n1k_v6_role_context/competencies.json"
    data = json.loads(run.read_text(encoding="utf-8"))
    for c in data:
        title = c.get("title", "?")
        sents = c.get("source_sentences") or []
        terms = _extract_product_terms(title)
        concerns = _audit_title_evidence(title, sents)
        terms_str = ", ".join(t for t, _ in terms) or "(none)"
        if concerns:
            print(f"\nTITLE: {title}")
            print(f"  extracted terms: {terms_str}")
            print(f"  source sentences: {len(sents)}")
            for c in concerns:
                print(f"  [{c['severity']:7s}] '{c['term']}' — {c['literal_hits']}/{c['n_sentences']} sentences (need {c['min_required']})")
        else:
            print(f"OK: {title[:60]}  (terms: {terms_str})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
