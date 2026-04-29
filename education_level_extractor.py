"""
education_level_extractor.py
============================

Parse education-requirement signals out of job postings and map them to KKNI
levels (1-9, see ``kkni.py``). The output is the *minimum* education level a
posting demands so we can later aggregate per-skill demand by KKNI band.

Design notes
------------
* English-first regexes (current data is English-only). Indonesian terms are
  included but not exercised by the prototyping data — they are forward-
  compatible for when the user adds an Indonesian corpus.
* When a posting lists multiple acceptable degrees (e.g. "Bachelor's required,
  Master's preferred"), we record the **lowest** since that's the minimum bar.
* When no signal is found, ``min_education_kkni`` is left as None — downstream
  code treats unknown as "any KKNI level" so the competency still appears in
  every stage filter.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

# Patterns are ordered: each tuple is (compiled_regex, kkni_level, label).
# Lower KKNI levels come first so the *minimum* is the first match.
# Word boundaries are used liberally so "S1" inside "AWS S1" doesn't false-match.

_PATTERNS: List[Tuple[re.Pattern, int, str]] = [
    # KKNI 2 — high school (SMA, no vocational)
    (re.compile(r"\b(?:high\s*school|hs|secondary\s*school|sma|senior\s*high)\b", re.I), 2, "SMA"),
    # KKNI 3 — vocational high school (SMK, D1)
    (re.compile(r"\b(?:smk|vocational(?:\s*(?:school|high\s*school|education))?|trade\s*school|d[\s-]?1|diploma\s*(?:1|i)\b)\b", re.I), 3, "SMK/D1"),
    # KKNI 4 — D2
    (re.compile(r"\b(?:d[\s-]?2|diploma\s*(?:2|ii)\b)\b", re.I), 4, "D2"),
    # KKNI 5 — D3 / Associate
    (re.compile(r"\b(?:d[\s-]?3|diploma\s*(?:3|iii)\b|associate(?:'?s)?(?:\s*degree)?)\b", re.I), 5, "D3"),
    # KKNI 6 — Bachelor / D4 / S1
    # The bare S1 abbreviation is ambiguous (matches AWS S1, EC2 S1 storage tiers).
    # We require it to appear in an academic context: preceded or followed by a
    # known education keyword within a short window. Other unambiguous tokens
    # (bachelor, sarjana, B.Sc, etc.) match without that constraint.
    (re.compile(
        r"\b(?:bachelor(?:'?s)?(?:\s*degree)?|undergraduate(?:\s*degree)?|"
        r"b\.?\s*(?:sc|s|eng|a|tech)\.?|"
        r"d[\s-]?4|diploma\s*(?:4|iv)\b|"
        r"sarjana(?:\s*(?:terapan|s1))?|s\.kom|s\.t\.?|s\.e\.?|"
        # bare S1 with academic context: "S1 Teknik", "lulusan S1", "minimal S1", etc.
        r"(?:lulusan|minimal|min\.?|pendidikan|gelar|jurusan)\s+s[\s-]?1|"
        r"s[\s-]?1\s+(?:teknik|ilmu|sistem|informatika|komputer|sains|matematika|elektro|atau\s+setara|sederajat|s\.kom))\b", re.I), 6, "S1/D4"),
    # KKNI 7 — Profesi (e.g., post-S1 professional certification)
    (re.compile(r"\bprofesi\b|\bprofessional\s*degree\b", re.I), 7, "Profesi"),
    # KKNI 8 — Master / S2 / Magister
    (re.compile(
        r"\b(?:master(?:'?s)?(?:\s*degree)?|graduate\s*degree|"
        r"m\.?\s*(?:sc|s|eng|a|ba|tech)\.?|"
        r"magister(?:\s*terapan)?|m\.kom|"
        r"(?:lulusan|minimal|min\.?|pendidikan|gelar|jurusan)\s+s[\s-]?2|"
        r"s[\s-]?2\s+(?:teknik|ilmu|sistem|informatika|komputer|sains|matematika|elektro|atau\s+setara|sederajat))\b", re.I), 8, "S2/Magister"),
    # KKNI 9 — Doctorate / S3 / Doktor / PhD
    (re.compile(
        r"\b(?:ph\.?\s*d|doctor(?:al|ate|'s)?|"
        r"doktor(?:\s*terapan)?|d\.sc|"
        r"(?:lulusan|minimal|min\.?|pendidikan|gelar|jurusan)\s+s[\s-]?3|"
        r"s[\s-]?3\s+(?:teknik|ilmu|sistem|informatika|komputer|sains|matematika|elektro|atau\s+setara|sederajat))\b", re.I), 9, "S3/Doktor"),
]


def extract_education_kkni(text: str) -> Tuple[Optional[int], List[str]]:
    """Find the *minimum* KKNI level required by the posting.

    Returns:
        (kkni_level, labels) where labels is the list of all detected stage
        labels (used for telemetry / debugging). kkni_level is None when no
        signal is found.
    """
    if not text or not isinstance(text, str):
        return None, []
    matches: List[Tuple[int, str]] = []
    for pat, level, label in _PATTERNS:
        if pat.search(text):
            matches.append((level, label))
    if not matches:
        return None, []
    # Minimum education = lowest KKNI level demanded
    min_level = min(m[0] for m in matches)
    labels = sorted({m[1] for m in matches})
    return min_level, labels


def annotate_dataframe(df, text_columns: Iterable[str] = ("description", "requirements")) -> "pd.DataFrame":
    """Add ``min_education_kkni`` and ``education_labels`` columns to a job DataFrame.

    Concatenates the named text columns per row (skipping missing) and runs the
    extractor. The DataFrame is mutated in place and also returned.
    """
    import pandas as pd  # local import; this module is also used standalone in tests

    cols_present = [c for c in text_columns if c in df.columns]
    if not cols_present:
        df["min_education_kkni"] = None
        df["education_labels"] = ""
        return df

    def _row_text(row) -> str:
        parts = []
        for c in cols_present:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        return " \n ".join(parts)

    levels: List[Optional[int]] = []
    labels_col: List[str] = []
    for _, row in df.iterrows():
        lvl, labels = extract_education_kkni(_row_text(row))
        levels.append(lvl)
        labels_col.append("; ".join(labels) if labels else "")
    df["min_education_kkni"] = levels
    df["education_labels"] = labels_col
    return df


def summarize(levels: Iterable[Optional[int]]) -> Dict[int, int]:
    """Count jobs per KKNI level (None = unknown, excluded from the histogram)."""
    counts: Dict[int, int] = {}
    for v in levels:
        if v is None:
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        counts[iv] = counts.get(iv, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# CLI / smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract KKNI education levels from job postings.")
    parser.add_argument("--input", required=True, help="Input CSV with job postings (must have title, description columns)")
    parser.add_argument("--output", default=None, help="Output CSV (defaults to input with _education suffix)")
    parser.add_argument("--text-columns", default="description,requirements",
                        help="Comma-separated list of columns to scan (default: description,requirements)")
    args = parser.parse_args()

    import pandas as pd
    from pathlib import Path

    src = Path(args.input)
    df = pd.read_csv(src)
    cols = [c.strip() for c in args.text_columns.split(",") if c.strip()]
    df = annotate_dataframe(df, text_columns=cols)
    out = Path(args.output) if args.output else src.with_name(src.stem + "_education.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote {out} ({len(df)} rows)")
    print("[INFO] KKNI level distribution:", json.dumps(summarize(df["min_education_kkni"].tolist())))
