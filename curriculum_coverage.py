"""
curriculum_coverage.py

Curriculum-vs-competencies coverage analysis. The killer UX feature for the
public dashboard: a curriculum designer uploads their existing syllabus
(PDF / .txt / pasted text) and gets back:

    - per-competency coverage score: how well the curriculum already teaches
      this competency (SBERT cosine between competency content and curriculum
      section content, max across sections)
    - per-competency gap: 1 - coverage; high-gap competencies are the
      "you should add this" list
    - overall coverage: weighted mean by priority_score (demand × trend × future_weight)

Approach:
  1. Parse the uploaded curriculum into "sections" (one section per heading
     or per non-empty paragraph for plain text).
  2. SBERT-embed each section + each competency (title + description +
     related_skills joined).
  3. For each competency, find the max cosine similarity vs any section.
  4. Map similarity → coverage label:
        >= 0.65   "well covered"
        0.45-0.65 "partially covered"
        < 0.45    "missing"
  5. Return a `CoverageReport` and per-competency annotations.

Embeddings use the project's default SBERT model (all-MiniLM-L6-v2) so they
share the same semantic space as Phase 2.1 clustering + Phase 2.5 evaluator.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


COVERAGE_THRESHOLDS = {
    "well_covered": 0.65,
    "partially_covered": 0.45,
    # below partially_covered = "missing"
}


@dataclass
class CurriculumSection:
    """One coherent chunk of curriculum text."""
    section_id: str
    title: str
    text: str
    embedding: object = None  # np.ndarray, set lazily


@dataclass
class CompetencyCoverage:
    """Coverage annotation for a single competency."""
    competency_id: str
    competency_title: str
    coverage_score: float                       # max SBERT cosine vs any section
    coverage_label: str                          # "well_covered" | "partially_covered" | "missing"
    best_matching_section_id: Optional[str] = None
    best_matching_section_title: str = ""
    gap_score: float = 0.0                       # 1 - coverage_score
    priority_weighted_gap: float = 0.0           # gap × future_weight (higher = more urgent)


@dataclass
class CoverageReport:
    """Aggregate report across all competencies."""
    n_competencies: int
    n_well_covered: int
    n_partially_covered: int
    n_missing: int

    mean_coverage: float
    weighted_mean_coverage: float                # weighted by future_weight (priority)
    top_gaps: List[CompetencyCoverage] = field(default_factory=list)

    curriculum_section_count: int = 0
    embedder_model: str = ""

    def to_dict(self) -> dict:
        return {
            "n_competencies": self.n_competencies,
            "n_well_covered": self.n_well_covered,
            "n_partially_covered": self.n_partially_covered,
            "n_missing": self.n_missing,
            "mean_coverage": round(float(self.mean_coverage), 4),
            "weighted_mean_coverage": round(float(self.weighted_mean_coverage), 4),
            "curriculum_section_count": self.curriculum_section_count,
            "embedder_model": self.embedder_model,
        }


# --------------------------------------------------------------------------- #
# Curriculum parsing
# --------------------------------------------------------------------------- #


_HEADING_RE = re.compile(
    r"^(?:#{1,4}\s+|"                           # markdown headings
    r"\d+\.\s+|"                                  # numbered "1. ", "2. " etc.
    r"[A-Z][A-Z\s]{2,}$)",                        # ALL CAPS HEADINGS
    re.MULTILINE,
)


def parse_curriculum_text(text: str, min_section_chars: int = 30) -> List[CurriculumSection]:
    """Split curriculum into sections by heading + paragraph.

    Returns a list of CurriculumSection with stable section_ids.
    """
    if not text or not text.strip():
        return []

    # Split on blank-line gaps first (preserves paragraph structure)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    sections: List[CurriculumSection] = []
    current_title = "Section 1"
    current_body: List[str] = []
    section_idx = 0

    def flush():
        nonlocal section_idx, current_title, current_body
        body = " ".join(current_body).strip()
        if len(body) >= min_section_chars:
            section_idx += 1
            sections.append(
                CurriculumSection(
                    section_id=f"sect_{section_idx:03d}",
                    title=current_title,
                    text=body,
                )
            )
        current_body = []

    for para in paragraphs:
        # Treat first line of paragraph as candidate heading
        first_line = para.split("\n", 1)[0].strip()
        is_heading = (
            len(first_line) < 120
            and bool(_HEADING_RE.match(first_line))
        )
        if is_heading:
            flush()
            current_title = re.sub(r"^#{1,4}\s+|^\d+\.\s+", "", first_line).strip()
            remainder = para[len(first_line):].strip()
            if remainder:
                current_body.append(remainder)
        else:
            current_body.append(para)

    flush()

    if not sections:
        # No headings detected — treat each paragraph as its own section
        for i, para in enumerate(paragraphs, 1):
            if len(para) >= min_section_chars:
                sections.append(
                    CurriculumSection(
                        section_id=f"sect_{i:03d}",
                        title=f"Paragraph {i}",
                        text=para,
                    )
                )

    return sections


def translate_sections_to_english(
    sections: List[CurriculumSection],
    *,
    only_if_indonesian: bool = True,
) -> Tuple[List[CurriculumSection], int]:
    """Translate non-English curriculum sections to English so the SBERT match
    against competencies (which are English) doesn't false-fail on language drift.

    Returns (translated_sections, n_translated). The returned sections are
    new objects with the original `title` preserved + translated `text`.
    """
    try:
        from translator import langdetect_lang, translate_to_english
    except ImportError:
        logger.warning("translator unavailable; skipping translation step")
        return sections, 0

    translated = []
    n_translated = 0
    for s in sections:
        lang = langdetect_lang(s.text)
        if only_if_indonesian and lang != "id":
            translated.append(s)
            continue
        en_text = translate_to_english(s.text, skip_if_english=False)
        en_title = translate_to_english(s.title, skip_if_english=False) if s.title else s.title
        translated.append(
            CurriculumSection(
                section_id=s.section_id,
                title=en_title,
                text=en_text,
            )
        )
        n_translated += 1
    return translated, n_translated


def parse_curriculum_pdf(pdf_bytes: bytes, min_section_chars: int = 30) -> List[CurriculumSection]:
    """Extract text from a PDF and parse into sections.

    Uses pypdf. Page boundaries are treated as soft section breaks if no
    markdown/numbered headings are detected.
    """
    try:
        import pypdf
        from io import BytesIO
        reader = pypdf.PdfReader(BytesIO(pdf_bytes))
        all_text = "\n\n".join((p.extract_text() or "") for p in reader.pages)
        return parse_curriculum_text(all_text, min_section_chars=min_section_chars)
    except Exception as e:
        logger.warning("failed to parse curriculum PDF: %s", e)
        return []


# --------------------------------------------------------------------------- #
# Embedding cache (shared with the rest of the v2 stack)
# --------------------------------------------------------------------------- #


_embedder_cache: dict = {}


def _get_embedder(model_name: str):
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        _embedder_cache[model_name] = m
        return m
    except Exception as e:
        logger.warning("SBERT %s unavailable: %s", model_name, e)
        _embedder_cache[model_name] = None
        return None


def _embed(texts: Sequence[str], embedder):
    import numpy as np
    return embedder.encode(
        list(texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


# --------------------------------------------------------------------------- #
# Coverage computation
# --------------------------------------------------------------------------- #


def _competency_text_for_embedding(comp_dict: dict) -> str:
    parts = [
        (comp_dict.get("title") or "").strip(),
        (comp_dict.get("description") or "").strip(),
        " ".join(comp_dict.get("related_skills") or []),
    ]
    return ". ".join([p for p in parts if p])


def compute_coverage(
    competencies: Sequence[dict],
    sections: Sequence[CurriculumSection],
    embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[List[CompetencyCoverage], CoverageReport]:
    """Compute coverage of `competencies` by `sections`.

    `competencies` is a list of dicts (CompetencyV2.to_dict() output), which is
    what the dashboard already has loaded — no need to rehydrate dataclasses.
    """
    if not sections:
        # No curriculum — every competency is "missing" but we honor that quietly
        return [
            CompetencyCoverage(
                competency_id=c.get("id", ""),
                competency_title=c.get("title", ""),
                coverage_score=0.0,
                coverage_label="missing",
                gap_score=1.0,
                priority_weighted_gap=float(c.get("future_weight", 0.0)),
            )
            for c in competencies
        ], CoverageReport(
            n_competencies=len(competencies),
            n_well_covered=0,
            n_partially_covered=0,
            n_missing=len(competencies),
            mean_coverage=0.0,
            weighted_mean_coverage=0.0,
            curriculum_section_count=0,
            embedder_model=embedder_model,
        )

    embedder = _get_embedder(embedder_model)
    if embedder is None:
        raise RuntimeError(
            "SBERT embedder unavailable. Install sentence-transformers in the venv."
        )

    import numpy as np

    # Embed sections once
    section_embs = _embed([s.text for s in sections], embedder)
    for sec, emb in zip(sections, section_embs):
        sec.embedding = emb

    # Embed competencies
    comp_texts = [_competency_text_for_embedding(c) for c in competencies]
    comp_embs = _embed(comp_texts, embedder)

    # Similarity matrix: (n_comp, n_sections)
    sims = comp_embs @ section_embs.T

    annotations: List[CompetencyCoverage] = []
    for i, c in enumerate(competencies):
        if sims.shape[1] == 0:
            cov, best_idx = 0.0, -1
        else:
            best_idx = int(sims[i].argmax())
            cov = float(sims[i, best_idx])

        if cov >= COVERAGE_THRESHOLDS["well_covered"]:
            label = "well_covered"
        elif cov >= COVERAGE_THRESHOLDS["partially_covered"]:
            label = "partially_covered"
        else:
            label = "missing"

        gap = 1.0 - cov
        fw = float(c.get("future_weight", 0.0))
        annotations.append(
            CompetencyCoverage(
                competency_id=c.get("id", ""),
                competency_title=c.get("title", ""),
                coverage_score=cov,
                coverage_label=label,
                best_matching_section_id=sections[best_idx].section_id if best_idx >= 0 else None,
                best_matching_section_title=sections[best_idx].title if best_idx >= 0 else "",
                gap_score=gap,
                priority_weighted_gap=gap * fw,
            )
        )

    # Aggregate report
    n_total = len(annotations)
    n_well = sum(1 for a in annotations if a.coverage_label == "well_covered")
    n_part = sum(1 for a in annotations if a.coverage_label == "partially_covered")
    n_miss = sum(1 for a in annotations if a.coverage_label == "missing")

    if n_total == 0:
        mean_cov = wmean_cov = 0.0
    else:
        mean_cov = sum(a.coverage_score for a in annotations) / n_total
        weights = [float(c.get("future_weight", 0.0)) for c in competencies]
        weight_sum = sum(weights) or 1.0
        wmean_cov = sum(a.coverage_score * w for a, w in zip(annotations, weights)) / weight_sum

    # Top 10 priority-weighted gaps for the "you should add this" list
    top_gaps = sorted(annotations, key=lambda a: -a.priority_weighted_gap)[:10]

    report = CoverageReport(
        n_competencies=n_total,
        n_well_covered=n_well,
        n_partially_covered=n_part,
        n_missing=n_miss,
        mean_coverage=mean_cov,
        weighted_mean_coverage=wmean_cov,
        top_gaps=top_gaps,
        curriculum_section_count=len(sections),
        embedder_model=embedder_model,
    )

    return annotations, report
