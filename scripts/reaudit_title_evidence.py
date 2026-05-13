"""reaudit_title_evidence.py — re-run the title-evidence audit against an
existing run's competencies.json in place. Used after fixing the extractor
when we don't want to pay for a full pipeline re-run.

Usage:
    python scripts/reaudit_title_evidence.py results/competency_v2_pipeline_n1k_v7_tier2d
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from competency_generator_v2 import _audit_title_evidence


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: reaudit_title_evidence.py <run_dir>")
        return 2
    run = Path(sys.argv[1])
    if not run.is_absolute():
        run = PROJECT_ROOT / run
    cp = run / "competencies.json"
    if not cp.exists():
        print(f"No competencies.json at {cp}")
        return 1
    data = json.loads(cp.read_text(encoding="utf-8"))
    n_with_concerns = 0
    for c in data:
        title = c.get("title", "")
        sents = c.get("source_sentences") or []
        c["title_evidence_concerns"] = _audit_title_evidence(title, sents)
        if c["title_evidence_concerns"]:
            n_with_concerns += 1
    cp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Re-audited {len(data)} competencies, {n_with_concerns} now carry title_evidence_concerns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
