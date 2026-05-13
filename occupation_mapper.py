"""
occupation_mapper.py — map generated competencies to SKKNI Indonesian
occupations (skema okupasi Indonesia).

Each row in `DATA/skema_okupasi_indonesia_with_desc.csv` is one occupational
standard (e.g. "Skema Okupasi Pemrogram", "Skema Okupasi Junior Data
Scientist") with an English title + ~2-4 sentence English description that
references the underlying SKKNI competency units.

We embed (title + description) per occupation via multilingual SBERT once,
then for each generated competency compute cosine similarity against all
occupations and keep top-K matches above a threshold. The matches let the
dashboard:

  - show a per-competency "Maps to" chip set
  - filter competencies by occupation (sidebar)
  - flag competencies that don't map to ANY occupation (suspect curriculum
    relevance for Indonesian context)

Defaults:
  - top_k = 3
  - min_cosine = 0.45 (anything below is "no map")

Cost: trivial. Embeds 334 occupation strings once, then competencies (~20
at n1k, ~200 at full scale). Done locally via SBERT — no LLM calls.

Public API:

    from occupation_mapper import load_occupations, map_competencies
    occupations = load_occupations()
    enriched = map_competencies(competencies, occupations)
    # each competency now has `occupation_matches` populated.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_SKKNI_CSV = Path("DATA") / "skema_okupasi_indonesia_with_desc.csv"
# v9.11: community-defined IT occupations that fill gaps in the current
# official SKKNI ICT catalogue (no Cloud Engineer, Frontend Developer, ML
# Engineer, NLP Engineer, UX Designer, Tech Lead, QA, etc.). Auto-merged
# at load time when the file exists; entries are flagged via `is_official=False`
# so the dashboard can display a "(community-defined, non-official)" caption.
DEFAULT_COMMUNITY_CSV = Path("DATA") / "skema_okupasi_community_additions.csv"
# Multilingual paraphrase-MiniLM-L12-v2 performed better on this corpus
# than the English-only MiniLM-L6-v2 — it picks up domain semantics
# encoded in formal SKKNI Indonesian/English vocabulary that the English
# model missed. Cosines are compressed (0.25-0.55 range vs the 0.45-0.75
# of English-only on English-on-English), so the threshold is set lower.
DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 3
DEFAULT_MIN_COSINE = 0.40

# SKKNI has 21 sectors. Unrestricted matching produces severe cross-sector
# noise — e.g. "AI/ML pipeline" matched "Pipeline Operations & Maintenance
# Supervisor" (oil and gas), "Lead engineers" matched "Construction Project
# Management Expert", "Cyber Security" matched "H2S Hazard Handling Officer".
# Default to whitelisting only the ICT sector. The "Industri Kreatif dan
# Komunikasi" sector contains Game Developer + multimedia roles that CAN be
# IT-relevant, but it also contains Interior Designer / Fashion Designer /
# Broadcasters / Photographers that bleed into design-flavoured competencies.
# Users targeting game-dev / multimedia curricula can opt in via the
# `--occupation-sectors` CLI flag.
DEFAULT_RELEVANT_SECTORS = frozenset({
    "Teknologi Informasi dan Komunikasi",     # 28 occupations: programmer, data sci, security, etc.
})

# Sentinel for "explicitly disable sector filter" (vs. "use default whitelist").
_SENTINEL_NO_FILTER = object()
NO_SECTOR_FILTER = _SENTINEL_NO_FILTER  # public re-export


@dataclass
class Occupation:
    """One row from the SKKNI CSV (or the community-additions file)."""
    sector: str
    name_id: str        # Indonesian Title
    name_en: str        # English Translation (used as the canonical id)
    description_id: str # Indonesian Description
    description_en: str # English Description (used for embedding)
    source_type: str = ""
    is_official: bool = True  # False for v9.11 community-defined additions

    def embedding_text(self) -> str:
        """Document we embed for similarity matching. English-only by default
        because generated competencies are English; switch to bilingual if
        the corpus shifts."""
        parts = [self.name_en]
        if self.description_en:
            parts.append(self.description_en)
        return ". ".join(p.strip().rstrip(".") for p in parts if p)


def _read_skkni_rows(path: Path, is_official: bool) -> List[Occupation]:
    """Read one CSV file in the SKKNI schema. Used for both the official
    catalogue and the community-additions file."""
    rows: List[Occupation] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(Occupation(
                sector=(row.get("Sektor") or "").strip(),
                name_id=(row.get("Indonesian Title") or "").strip(),
                name_en=(row.get("English Translation") or "").strip(),
                description_id=(row.get("Indonesian Description") or "").strip(),
                description_en=(row.get("English Description") or "").strip(),
                source_type=(row.get("Source Type") or "").strip(),
                is_official=is_official,
            ))
    return rows


def load_occupations(
    csv_path: Optional[Path] = None,
    community_csv_path: Optional[Path] = None,
    *,
    include_community: bool = True,
) -> List[Occupation]:
    """Load the enriched SKKNI CSV. Returns one Occupation per row.

    When `include_community=True` (the default) and the community-additions
    file exists, its rows are appended with `is_official=False`. The
    community file fills documented gaps in the official ICT catalogue
    (no Cloud Engineer / Frontend Developer / ML Engineer / NLP Engineer /
    UX Designer / Tech Lead / QA Engineer in the Kepmenaker series).

    The dashboard surfaces a "(community-defined, non-official)" disclaimer
    on chips whose match comes from a community entry.
    """
    path = Path(csv_path) if csv_path else DEFAULT_SKKNI_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"SKKNI CSV not found at {path}. Provide `csv_path` or place the "
            "file at DATA/skema_okupasi_indonesia_with_desc.csv."
        )
    official = _read_skkni_rows(path, is_official=True)
    community: List[Occupation] = []
    if include_community:
        community_path = Path(community_csv_path) if community_csv_path else DEFAULT_COMMUNITY_CSV
        community = _read_skkni_rows(community_path, is_official=False)
    out = official + community
    logger.info(
        "Loaded %d SKKNI occupations: %d official from %s + %d community from %s",
        len(out), len(official), path, len(community),
        community_csv_path if community_csv_path else DEFAULT_COMMUNITY_CSV,
    )
    return out


# --------------------------------------------------------------------------- #
# Embedding (re-use the existing SBERT cache from clustering)
# --------------------------------------------------------------------------- #


_EMBEDDER_CACHE: dict = {}


def _get_embedder(model_name: str):
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    # Reuse the same HF symlink workaround that kkni_labeller uses on Windows
    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from sentence_transformers import SentenceTransformer
    logger.info("Loading SBERT model %s for occupation mapping", model_name)
    model = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = model
    return model


def _embed(texts: Iterable[str], model_name: str = DEFAULT_SBERT_MODEL) -> np.ndarray:
    texts = list(texts)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_embedder(model_name)
    emb = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #


def _competency_embedding_text(comp: dict) -> str:
    """The text we embed for a competency. Combines title + description; the
    related_skills list adds breadth without dominating the signal."""
    parts = []
    title = comp.get("title") or ""
    desc = comp.get("description") or ""
    if title:
        parts.append(title)
    if desc:
        parts.append(desc)
    rs = comp.get("related_skills") or []
    if rs:
        parts.append("Related skills: " + ", ".join(rs[:10]))
    return ". ".join(p.strip().rstrip(".") for p in parts if p)


def map_competencies(
    competencies: List[dict],
    occupations: Optional[List[Occupation]] = None,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_cosine: float = DEFAULT_MIN_COSINE,
    sbert_model: str = DEFAULT_SBERT_MODEL,
    relevant_sectors: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Mutate each competency dict in place by adding an `occupation_matches`
    list of `{occupation_en, occupation_id, sector, cosine, description_en}`
    dicts (top-K above threshold). Returns the same list for convenience.

    When NO match meets the threshold, `occupation_matches` is an empty
    list — useful as a flag for "this competency doesn't fit any Indonesian
    occupational standard" (likely curriculum-irrelevant).

    Sector filtering: by default only `DEFAULT_RELEVANT_SECTORS` are
    considered. Pass `relevant_sectors=None` to disable (allow cross-sector
    matches — noisy, not recommended) or override with a different set for
    non-IT corpora.
    """
    if occupations is None:
        occupations = load_occupations()
    # Sector filter — sentinel: passing None to the call disables the
    # default whitelist. We distinguish "user said None" from "user didn't
    # pass anything" by using the special _SENTINEL_NO_FILTER value.
    if relevant_sectors is _SENTINEL_NO_FILTER or relevant_sectors is None:
        # The function signature default is None, meaning "use the module
        # default whitelist". To bypass filtering entirely callers must
        # explicitly pass an empty set or `_SENTINEL_NO_FILTER`. For
        # ergonomics we accept the common case: relevant_sectors=None means
        # "use the default IT whitelist".
        active_sectors = set(DEFAULT_RELEVANT_SECTORS)
    else:
        active_sectors = set(relevant_sectors) if relevant_sectors else set()

    if active_sectors:
        filtered_occupations = [o for o in occupations if o.sector in active_sectors]
        logger.info(
            "Occupation mapping: sector filter ON, %d / %d occupations "
            "retained (sectors: %s)",
            len(filtered_occupations), len(occupations),
            ", ".join(sorted(active_sectors)),
        )
        occupations = filtered_occupations

    if not competencies or not occupations:
        for c in competencies:
            c.setdefault("occupation_matches", [])
        return competencies

    occ_texts = [o.embedding_text() for o in occupations]
    occ_emb = _embed(occ_texts, model_name=sbert_model)  # (N_occ, d)

    comp_texts = [_competency_embedding_text(c) for c in competencies]
    comp_emb = _embed(comp_texts, model_name=sbert_model)  # (N_comp, d)

    # Cosine similarity — both matrices are L2-normalised, so cosine = dot.
    sims = comp_emb @ occ_emb.T  # (N_comp, N_occ)

    for i, comp in enumerate(competencies):
        row = sims[i]
        order = np.argsort(-row)
        matches = []
        for j in order:
            score = float(row[j])
            if score < min_cosine:
                break
            occ = occupations[j]
            matches.append({
                "occupation_en": occ.name_en,
                "occupation_id": occ.name_id,
                "sector": occ.sector,
                "cosine": round(score, 4),
                "description_en": occ.description_en[:400],
                "is_official": bool(occ.is_official),
                "source_type": occ.source_type,
            })
            if len(matches) >= top_k:
                break
        comp["occupation_matches"] = matches
    n_mapped = sum(1 for c in competencies if c.get("occupation_matches"))
    logger.info(
        "Occupation mapping: %d / %d competencies mapped to >= 1 occupation "
        "(top_k=%d, min_cosine=%.2f)",
        n_mapped, len(competencies), top_k, min_cosine,
    )
    return competencies


__all__ = [
    "Occupation",
    "load_occupations",
    "map_competencies",
    "DEFAULT_SKKNI_CSV",
    "DEFAULT_SBERT_MODEL",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_COSINE",
]
