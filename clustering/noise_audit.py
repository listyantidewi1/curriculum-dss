"""
noise_audit.py — track items dropped at every Phase 2.1 stage.

Every item that's extracted by Phase 1 either ends up in a cluster (and
therefore in a competency downstream) OR gets dropped at one of these
stages:

    1. generic_single_token  — single-token generic word filtered during
                               canonicalization (e.g. "platform", "math")
    2. below_frequency       — canonical item appears in fewer than
                               `min_global_frequency` unique jobs
    3. hdbscan_noise         — HDBSCAN tagged as noise; agglomerative
                               recovery did not pick it up either
    4. cluster_dropped_low_cohesion — whole cluster fell below the
                               cohesion threshold; its items returned
                               to noise and were not recovered
    5. refinement_move_out   — LLM arbiter moved the item out of its
                               cluster (Phase 2.1 v8 sprint)
    6. post_refinement_noise — even the second agglomerative recovery
                               pass (v8.1) couldn't place this item

The audit gives curriculum designers transparency: every skill the
extractor surfaced, if it didn't reach a competency, has a recorded
reason and stage.

Output schema (serialised to `noise_audit.json` alongside `clusters.json`):

    {
      "schema_version": "v1",
      "n_items_dropped_total": int,
      "by_stage": {stage_name: count, ...},
      "items": [
        {
          "text": str,
          "stage": str,
          "reason": str,
          "n_unique_jobs": int,    (when known)
          "extra": dict | null,    (stage-specific context)
        },
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_STAGE_REASONS = {
    "generic_single_token": "single-token generic word filtered during canonicalization",
    "below_frequency":      "canonical item appears in too few unique job postings",
    "hdbscan_noise":        "HDBSCAN tagged as noise; first recovery did not recover",
    "cluster_dropped_low_cohesion": "the cluster this item belonged to fell below cohesion threshold",
    "refinement_move_out":  "LLM arbiter moved this item out of its cluster (off-theme)",
    "post_refinement_noise": "did not form a cluster even with the post-refinement recovery pass",
}


@dataclass
class NoiseAuditEntry:
    text: str
    stage: str
    reason: str = ""
    n_unique_jobs: int = 0
    extra: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "stage": self.stage,
            "reason": self.reason or _STAGE_REASONS.get(self.stage, ""),
            "n_unique_jobs": int(self.n_unique_jobs),
            "extra": self.extra,
        }


@dataclass
class NoiseAuditTracker:
    """Append-only tracker for items dropped at any Phase 2.1 stage.

    Usage:
        tracker = NoiseAuditTracker()
        tracker.record("generic_single_token", "math", n_unique_jobs=12)
        ...
        tracker.serialize(path/"noise_audit.json")
    """
    entries: List[NoiseAuditEntry] = field(default_factory=list)

    def record(
        self,
        stage: str,
        text: str,
        n_unique_jobs: int = 0,
        reason: str = "",
        extra: Optional[dict] = None,
    ) -> None:
        if not text:
            return
        self.entries.append(NoiseAuditEntry(
            text=text, stage=stage, reason=reason,
            n_unique_jobs=n_unique_jobs, extra=extra,
        ))

    def to_dict(self) -> dict:
        by_stage: dict = {}
        for e in self.entries:
            by_stage[e.stage] = by_stage.get(e.stage, 0) + 1
        return {
            "schema_version": "v1",
            "n_items_dropped_total": len(self.entries),
            "by_stage": by_stage,
            "items": [e.to_dict() for e in self.entries],
        }

    def serialize(self, path: Path) -> None:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
