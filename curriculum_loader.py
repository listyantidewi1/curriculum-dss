"""
curriculum_loader.py

Helpers to load a school/department curriculum definition from CSV/JSON.
The returned structure is normalized for dashboard use and future mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _split_phrases(value: str) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def load_curriculum_file(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Curriculum file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Curriculum JSON must be a list of objects")
        normalized = []
        for row in data:
            normalized.append(
                {
                    "component_id": str(row.get("id", "")).strip(),
                    "component_name": str(row.get("name", "")).strip(),
                    "phrases": [str(x).strip() for x in row.get("phrases", []) if str(x).strip()],
                }
            )
        return normalized

    df = pd.read_csv(path)
    # bloom_level was previously required but is ignored after pipeline-redesign-v2
    # (Req 1.5: legacy CSVs with a bloom_level column still load without error).
    required = {"component_id", "component_name", "phrases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Curriculum CSV missing columns: {sorted(missing)}")

    normalized = []
    for _, row in df.iterrows():
        normalized.append(
            {
                "component_id": str(row["component_id"]).strip(),
                "component_name": str(row["component_name"]).strip(),
                "phrases": _split_phrases(row["phrases"]),
            }
        )
    return normalized

