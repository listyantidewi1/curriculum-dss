"""
kkni.py
=======

Source of truth for **Kerangka Kualifikasi Nasional Indonesia** (KKNI) —
the Indonesian National Qualification Framework defined by Perpres 8/2012.

Used by:
- generate_competencies.py: assigns a kkni_level (1-9) to each competency.
- education_level_extractor.py: maps job-posting education requirements to KKNI.
- The public dashboard: maps friendly stage chips (SMK / S1 / S2 ...) → KKNI levels.

Design notes
------------
* The 9 KKNI levels are grouped into three job clusters:
    1-3 = Operator
    4-6 = Teknisi / Analis
    7-9 = Ahli (Profesi / Magister / Doktor)
* `BLOOM_TO_KKNI_FLOOR` is the deterministic floor used by `kkni_floor_for_competency`.
  The LLM is allowed to pick `floor` OR `floor + 1` (clamped by `clamp_llm_kkni`).
* `STAGE_TO_KKNI` is the friendly mapping shown on the public UI.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Level descriptors (verbatim summaries from Perpres 8/2012, Lampiran)
# --------------------------------------------------------------------------- #

KKNI_LEVELS: Dict[int, Dict[str, object]] = {
    1: {
        "label": "Operator (basic)",
        "stages": ["SMP", "Pendidikan Dasar"],
        "descriptor": (
            "Mampu melaksanakan tugas sederhana, terbatas, bersifat rutin, "
            "menggunakan alat, aturan, dan proses yang telah ditetapkan, di "
            "bawah bimbingan atasan. Memiliki pengetahuan faktual."
        ),
    },
    2: {
        "label": "Operator",
        "stages": ["SMA", "Pendidikan Menengah"],
        "descriptor": (
            "Mampu melaksanakan satu tugas spesifik dengan mutu yang terukur, "
            "di bawah pengawasan langsung. Memiliki pengetahuan operasional "
            "dasar dan pengetahuan faktual bidang kerja yang spesifik."
        ),
    },
    3: {
        "label": "Operator (skilled)",
        "stages": ["SMK", "D1"],
        "descriptor": (
            "Mampu melaksanakan serangkaian tugas spesifik dengan menerjemahkan "
            "informasi dan menggunakan alat, berdasarkan sejumlah pilihan "
            "prosedur kerja. Memiliki pengetahuan operasional yang lengkap dan "
            "prinsip-prinsip terkait fakta bidang keahlian tertentu."
        ),
    },
    4: {
        "label": "Teknisi / Analis (junior)",
        "stages": ["D2"],
        "descriptor": (
            "Mampu menyelesaikan tugas berlingkup luas dan kasus spesifik dengan "
            "menganalisis informasi secara terbatas dan memilih metode dari "
            "beberapa pilihan baku. Menguasai beberapa prinsip dasar bidang "
            "keahlian tertentu."
        ),
    },
    5: {
        "label": "Teknisi / Analis",
        "stages": ["D3"],
        "descriptor": (
            "Mampu menyelesaikan pekerjaan berlingkup luas, memilih metode dari "
            "beragam pilihan baku/non-baku dengan menganalisis data. Menguasai "
            "konsep teoritis bidang pengetahuan tertentu secara umum, mampu "
            "memformulasikan penyelesaian masalah prosedural."
        ),
    },
    6: {
        "label": "Teknisi / Analis (senior)",
        "stages": ["D4", "S1", "Sarjana", "Sarjana Terapan"],
        "descriptor": (
            "Mampu mengaplikasikan bidang keahliannya dan memanfaatkan ilmu "
            "pengetahuan, teknologi, dan/atau seni dalam penyelesaian masalah, "
            "serta beradaptasi terhadap situasi yang dihadapi. Menguasai konsep "
            "teoritis bidang pengetahuan tertentu secara umum dan bagian khusus "
            "secara mendalam."
        ),
    },
    7: {
        "label": "Ahli (Profesi)",
        "stages": ["Profesi"],
        "descriptor": (
            "Mampu merencanakan dan mengelola sumberdaya, mengevaluasi kerjanya "
            "secara komprehensif untuk menghasilkan langkah-langkah pengembangan "
            "strategis organisasi. Memecahkan permasalahan melalui pendekatan "
            "monodisipliner. Mampu melakukan riset dan mengambil keputusan "
            "strategis dengan akuntabilitas penuh."
        ),
    },
    8: {
        "label": "Ahli (Magister)",
        "stages": ["S2", "Magister", "Magister Terapan"],
        "descriptor": (
            "Mampu mengembangkan pengetahuan, teknologi, dan/atau seni di dalam "
            "bidang keilmuannya melalui riset hingga menghasilkan karya inovatif "
            "dan teruji. Memecahkan permasalahan melalui pendekatan inter atau "
            "multidisipliner."
        ),
    },
    9: {
        "label": "Ahli (Doktor)",
        "stages": ["S3", "Doktor", "Doktor Terapan"],
        "descriptor": (
            "Mampu mengembangkan pengetahuan, teknologi, dan/atau seni baru di "
            "dalam bidang keilmuannya melalui riset hingga menghasilkan karya "
            "kreatif, original, dan teruji. Memecahkan permasalahan melalui "
            "pendekatan inter, multi, dan transdisipliner."
        ),
    },
}

KKNI_MIN_LEVEL = 1
KKNI_MAX_LEVEL = 9


# --------------------------------------------------------------------------- #
# Friendly education-stage chips → KKNI level set
# Used by the public UI's stage-filter and by education-level extraction.
# --------------------------------------------------------------------------- #

STAGE_TO_KKNI: Dict[str, List[int]] = {
    "SMA":     [2],
    "SMK":     [2, 3],
    "D1":      [3],
    "D2":      [4],
    "D3":      [5],
    "D4":      [6],
    "S1":      [6],
    "Profesi": [7],
    "S2":      [7, 8],
    "S3":      [9],
}


# --------------------------------------------------------------------------- #
# Bloom → KKNI deterministic floor
# The LLM is allowed to bump up by +1 (e.g. floor=5 → llm may pick 5 or 6).
# Anything outside [floor, floor+1] gets clamped back to floor.
# --------------------------------------------------------------------------- #

BLOOM_TO_KKNI_FLOOR: Dict[str, int] = {
    "Remember":   1,
    "Understand": 2,
    "Apply":      4,
    "Analyze":    5,
    "Evaluate":   6,
    "Create":     7,
    # Soft-skill default; rarely used since competencies are hard-only after the reframe
    "N/A":        2,
}

_BLOOM_RANK = {b: i for i, b in enumerate(
    ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
)}


def kkni_floor_for_competency(bloom_levels: Iterable[str]) -> int:
    """Compute the deterministic KKNI floor for a competency.

    Takes the Bloom levels of the competency's related_skills and returns the
    floor based on the *highest* Bloom level present.

    Falls back to KKNI 3 (SMK-level operator) if no Bloom levels are usable.
    """
    bloom_levels = [str(b).strip() for b in (bloom_levels or []) if b]
    valid = [b for b in bloom_levels if b in _BLOOM_RANK]
    if not valid:
        return 3
    highest = max(valid, key=lambda b: _BLOOM_RANK[b])
    return BLOOM_TO_KKNI_FLOOR.get(highest, 3)


def clamp_llm_kkni(llm_value: object, floor: int) -> int:
    """Clamp an LLM-proposed KKNI level into the allowed band [floor, floor+1].

    Any non-integer or out-of-band value falls back to the floor.
    """
    try:
        v = int(llm_value)
    except (TypeError, ValueError):
        return floor
    if v < KKNI_MIN_LEVEL:
        return floor
    if v > KKNI_MAX_LEVEL:
        return min(floor + 1, KKNI_MAX_LEVEL)
    if v < floor:
        return floor
    if v > floor + 1:
        return min(floor + 1, KKNI_MAX_LEVEL)
    return v


def kkni_descriptor(level: int) -> str:
    """Return the one-line descriptor for a KKNI level (or empty string if unknown)."""
    rec = KKNI_LEVELS.get(int(level)) if level is not None else None
    return str(rec.get("descriptor", "")) if rec else ""


def kkni_label(level: int) -> str:
    """Return the short label (e.g., 'Teknisi / Analis') for a KKNI level."""
    rec = KKNI_LEVELS.get(int(level)) if level is not None else None
    return str(rec.get("label", "")) if rec else ""


def stages_for_level(level: int) -> List[str]:
    """Return education stages typically associated with a KKNI level."""
    rec = KKNI_LEVELS.get(int(level)) if level is not None else None
    return list(rec.get("stages", [])) if rec else []


def levels_for_stage(stage: str) -> List[int]:
    """Return KKNI levels that map to a friendly education stage chip."""
    if not stage:
        return []
    return list(STAGE_TO_KKNI.get(str(stage).strip(), []))


def kkni_reference_for_prompt(min_level: Optional[int] = None,
                              max_level: Optional[int] = None) -> str:
    """Render a compact KKNI reference table for inclusion in an LLM prompt.

    By default emits all 9 levels; pass min/max to restrict (useful when we
    only want to show the LLM the floor and floor+1 rows).
    """
    lo = max(KKNI_MIN_LEVEL, min_level or KKNI_MIN_LEVEL)
    hi = min(KKNI_MAX_LEVEL, max_level or KKNI_MAX_LEVEL)
    lines = ["KKNI level reference (Perpres 8/2012):"]
    for lvl in range(lo, hi + 1):
        rec = KKNI_LEVELS[lvl]
        stages = ", ".join(rec.get("stages", []) or [])
        lines.append(f"- Level {lvl} ({rec['label']}; stages: {stages}): {rec['descriptor']}")
    return "\n".join(lines)
