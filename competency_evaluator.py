"""
competency_evaluator.py

Phase 2.5 — canonical grounding-gate evaluator for CompetencyV2.

Enforces the v2 hard gate: every emitted competency must have
`grounding_score >= 0.80`, where grounding_score = |verified related_skills|
/ |total related_skills|. A skill counts as "verified" if it appears in
any of the competency's `source_sentences` — either as a substring
(case-insensitive, whitespace-normalized) or with SBERT cosine similarity
≥ `sbert_threshold` against any source sentence's embedding.

Why two methods:
  - **Substring match** is fast, deterministic, and works for the common case
    where the LLM keeps the literal skill phrase from the cluster items.
  - **SBERT cosine** catches paraphrases the LLM legitimately produced
    (e.g. "API design" ≈ "designing REST APIs") without forcing the gate
    to false-fail on perfectly-grounded competencies.

Public API:

    from competency_evaluator import evaluate_competencies, EvaluatorConfig

    passing, failing, report = evaluate_competencies(competencies, config=EvaluatorConfig())

    # `passing` and `failing` are List[CompetencyV2] (the originals, mutated in
    # place to carry grounding_score / grounding_passed / grounding_method /
    # grounding_reasoning fields).
    # `report` is an EvaluationReport with aggregate metrics for audit.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from competency_v2_schema import CompetencyV2

logger = logging.getLogger(__name__)


EVALUATOR_VERSION = "v2.5.0"


@dataclass
class EvaluatorConfig:
    """Knobs for the grounding evaluator.

    Updated 2026-05-12 (honesty audit): substring matching is deprecated.
    Both skill_grounding and sentence_relevance are computed via SBERT cosine.
    The canonical grounding_score = min(skill_grounding, sentence_relevance).
    """

    # Canonical gate. Lowered from 0.80 to 0.50 because the metric semantic
    # changed: it's no longer "fraction of skills matched" (which inflated
    # toward 1.00 via substring) but "min of two SBERT-cosine fractions".
    grounding_threshold: float = 0.50

    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # match pipeline default

    # Per-pair cosine threshold for "this skill / this sentence supports the competency".
    # A skill is verified if it cosine-matches at least one source_sentence at >= this.
    # A sentence is "relevant" if it cosine-matches the competency context at >= this.
    #
    # Calibration sweep on the e2e_v3 corpus (2026-05-13):
    #   sbert=0.50 → 1/12 pass (too strict)
    #   sbert=0.45 → 5/12 pass (discriminating — correctly catches broad clusters)
    #   sbert=0.40 → 11/12 pass (too permissive)
    #   sbert=0.35 → 11/12 pass (passes everything; no signal)
    # 0.45 sits at the inflection point where the metric becomes meaningful.
    sbert_threshold: float = 0.45

    # Optional: skip SBERT entirely (degraded mode for unit tests only)
    disable_sbert: bool = False


@dataclass
class EvaluationReport:
    """Audit-friendly summary of one evaluator run."""

    n_evaluated: int
    n_passed: int
    n_failed: int
    threshold: float

    grounding_mean: float
    grounding_median: float
    grounding_min: float

    n_substring_only: int           # competencies where every verified skill matched by substring
    n_sbert_used: int               # competencies where at least one skill needed SBERT
    n_unverifiable: int             # competencies where at least one skill failed both methods

    runtime_seconds: float = 0.0
    config: Optional[EvaluatorConfig] = None

    def to_dict(self) -> dict:
        cfg_dict = None
        if self.config is not None:
            cfg_dict = {k: v for k, v in self.config.__dict__.items()}
        return {
            "n_evaluated": self.n_evaluated,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "pass_rate": round(self.n_passed / max(1, self.n_evaluated), 4),
            "threshold": self.threshold,
            "grounding_mean": round(self.grounding_mean, 4),
            "grounding_median": round(self.grounding_median, 4),
            "grounding_min": round(self.grounding_min, 4),
            "n_substring_only": self.n_substring_only,
            "n_sbert_used": self.n_sbert_used,
            "n_unverifiable": self.n_unverifiable,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "config": cfg_dict,
            "evaluator_version": EVALUATOR_VERSION,
        }


# --------------------------------------------------------------------------- #
# Matching primitives
# --------------------------------------------------------------------------- #


def _normalize_for_substring(text: str) -> str:
    """Lowercase, collapse whitespace, strip — used for both skill and sentence."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _substring_match(skill: str, sentences: Sequence[str]) -> Optional[int]:
    """Return the index of the first sentence containing the skill (substring),
    or None if no sentence matches.

    Case-insensitive, whitespace-normalized. Punctuation is preserved
    (technical phrases often hinge on it: 'C++' vs 'C', 'CI/CD' vs 'cicd').
    """
    skill_norm = _normalize_for_substring(skill)
    if not skill_norm:
        return None
    for i, sent in enumerate(sentences):
        if skill_norm in _normalize_for_substring(sent):
            return i
    return None


def _sbert_match(
    skill: str,
    sentence_embeddings,                 # np.ndarray (n_sentences, dim) or None
    skill_embedder,                      # callable(text) -> np.ndarray (dim,)
    threshold: float,
) -> Optional[Tuple[int, float]]:
    """Embed the skill and find the source-sentence with max cosine similarity.

    Returns (index, similarity) if max similarity >= threshold, else None.
    Both inputs must be L2-normalized; cosine == dot product.
    """
    if sentence_embeddings is None or len(sentence_embeddings) == 0:
        return None
    skill_vec = skill_embedder(skill)
    sims = sentence_embeddings @ skill_vec  # (n_sentences,)
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return best_idx, best_sim
    return None


# --------------------------------------------------------------------------- #
# Embedding helper — lazy, cached on the module
# --------------------------------------------------------------------------- #


_embedder_cache: dict = {}


def _get_embedder(model_name: str):
    """Load and cache an SBERT model. Returns None if SBERT is unavailable."""
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        _embedder_cache[model_name] = model
        return model
    except Exception as e:
        logger.warning("SBERT model %s unavailable: %s — falling back to substring-only", model_name, e)
        _embedder_cache[model_name] = None
        return None


# --------------------------------------------------------------------------- #
# Per-competency evaluation
# --------------------------------------------------------------------------- #


def evaluate_one(
    competency: CompetencyV2,
    config: EvaluatorConfig,
    embedder=None,
) -> None:
    """Compute canonical grounding_score for one competency. Mutates in place.

    Two-track semantic grounding (post-2026-05-12 honesty audit):

      skill_grounding   = fraction of related_skills that SBERT-match at least
                          one source_sentence at cosine >= sbert_threshold.
      sentence_relevance = fraction of source_sentences that SBERT-match the
                          competency CONTEXT (title + description) at cosine
                          >= sbert_threshold.

    canonical grounding_score = min(skill_grounding, sentence_relevance).

    Why the min: a high skill_grounding alone proves "the words appeared
    somewhere"; a high sentence_relevance alone proves "the sentences are
    on-topic." Both must hold for the competency to be honestly grounded.
    """
    skills = list(competency.related_skills or [])
    sentences = list(competency.source_sentences or [])
    n_skills = len(skills)
    n_sentences = len(sentences)

    if n_skills == 0 or n_sentences == 0:
        competency.grounding_score = 0.0
        competency.skill_grounding_score = 0.0
        competency.sentence_relevance_score = 0.0
        competency.grounding_passed = False
        competency.grounding_method = "no_skills" if n_skills == 0 else "no_sentences"
        competency.grounding_reasoning = (
            f"competency has {n_skills} related_skills and {n_sentences} source_sentences; "
            "both must be > 0 for grounding"
        )
        competency.evaluator_version = EVALUATOR_VERSION
        return

    if config.disable_sbert or embedder is None:
        # Degraded mode — substring only. Reserved for offline tests.
        competency.grounding_score = 0.0
        competency.skill_grounding_score = 0.0
        competency.sentence_relevance_score = 0.0
        competency.grounding_passed = False
        competency.grounding_method = "sbert_disabled"
        competency.grounding_reasoning = "SBERT disabled — cannot semantically ground"
        competency.evaluator_version = EVALUATOR_VERSION
        return

    import numpy as np  # noqa: F401

    # ------ Embeddings ------
    try:
        sentence_embs = embedder.encode(
            sentences, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        skill_embs = embedder.encode(
            skills, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        # Competency context: title + description. Add rationale's first sentence
        # if available to give the embedder more signal.
        context = f"{competency.title}. {competency.description}".strip()
        if competency.rationale:
            first_rat = competency.rationale.split(".")[0].strip()
            if first_rat:
                context = f"{context} {first_rat}."
        context_emb = embedder.encode(
            [context], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )[0]
    except Exception as e:
        logger.warning("embedding failed for %s: %s", competency.id, e)
        competency.grounding_score = 0.0
        competency.skill_grounding_score = 0.0
        competency.sentence_relevance_score = 0.0
        competency.grounding_passed = False
        competency.grounding_method = "embed_error"
        competency.grounding_reasoning = f"embedding failure: {e}"
        competency.evaluator_version = EVALUATOR_VERSION
        return

    # ------ Track 1: skill_grounding ------
    # skill[i] is "verified" iff max_j cos(skill[i], sentence[j]) >= threshold
    skill_to_sent_sim = skill_embs @ sentence_embs.T   # (n_skills, n_sentences)
    per_skill_max = skill_to_sent_sim.max(axis=1)       # (n_skills,)
    per_skill_best_sent = skill_to_sent_sim.argmax(axis=1)
    skill_verified = per_skill_max >= config.sbert_threshold
    n_skill_verified = int(skill_verified.sum())
    skill_grounding = n_skill_verified / n_skills

    # ------ Track 2: sentence_relevance ------
    # sentence[j] is "relevant" iff cos(sentence[j], competency_context) >= threshold
    sent_to_ctx_sim = sentence_embs @ context_emb       # (n_sentences,)
    sent_relevant = sent_to_ctx_sim >= config.sbert_threshold
    n_sent_relevant = int(sent_relevant.sum())
    sentence_relevance = n_sent_relevant / n_sentences

    # ------ Canonical = min ------
    canonical = min(skill_grounding, sentence_relevance)

    # ------ Audit reasoning ------
    weak_skills = [
        f"'{skills[i]}' (best_cos={per_skill_max[i]:.2f})"
        for i in range(n_skills) if not skill_verified[i]
    ]
    weak_sents = [
        f"sent#{j} cos={sent_to_ctx_sim[j]:.2f}"
        for j in range(n_sentences) if not sent_relevant[j]
    ]
    reasoning = (
        f"skill_grounding = {n_skill_verified}/{n_skills} = {skill_grounding:.3f} | "
        f"sentence_relevance = {n_sent_relevant}/{n_sentences} = {sentence_relevance:.3f} | "
        f"canonical (min) = {canonical:.3f} (threshold {config.grounding_threshold:.2f}). "
    )
    if weak_skills:
        reasoning += f"Skills not well-supported: {', '.join(weak_skills[:5])}. "
    if weak_sents:
        reasoning += f"Sentences off-topic: {len(weak_sents)} of {n_sentences} (e.g., {', '.join(weak_sents[:3])}). "

    competency.grounding_score = canonical
    competency.skill_grounding_score = skill_grounding
    competency.sentence_relevance_score = sentence_relevance
    competency.grounding_passed = canonical >= config.grounding_threshold
    competency.grounding_method = "sbert"
    competency.grounding_reasoning = reasoning
    competency.evaluator_version = EVALUATOR_VERSION


# --------------------------------------------------------------------------- #
# Top-level entry
# --------------------------------------------------------------------------- #


def evaluate_competencies(
    competencies: Sequence[CompetencyV2],
    config: Optional[EvaluatorConfig] = None,
) -> Tuple[List[CompetencyV2], List[CompetencyV2], EvaluationReport]:
    """Evaluate a batch of competencies. Returns (passing, failing, report).

    Mutates each competency to populate the canonical grounding fields.
    """
    import time

    if config is None:
        config = EvaluatorConfig()

    t0 = time.time()

    # Pre-load the SBERT embedder so all competencies share it
    embedder = None
    if not config.disable_sbert:
        embedder = _get_embedder(config.sbert_model)

    for c in competencies:
        evaluate_one(c, config, embedder=embedder)

    passing = [c for c in competencies if c.grounding_passed]
    failing = [c for c in competencies if not c.grounding_passed]

    scores = [c.grounding_score for c in competencies]
    if not scores:
        gmean = gmedian = gmin = 0.0
    else:
        import statistics
        gmean = statistics.mean(scores)
        gmedian = statistics.median(scores)
        gmin = min(scores)

    n_substring_only = sum(1 for c in competencies if c.grounding_method == "substring")
    n_sbert_used = sum(1 for c in competencies if c.grounding_method in ("sbert", "mixed", "mixed_with_failures"))
    n_unverifiable = sum(1 for c in competencies if c.grounding_method in ("mixed_with_failures", "all_failed"))

    report = EvaluationReport(
        n_evaluated=len(competencies),
        n_passed=len(passing),
        n_failed=len(failing),
        threshold=config.grounding_threshold,
        grounding_mean=gmean,
        grounding_median=gmedian,
        grounding_min=gmin,
        n_substring_only=n_substring_only,
        n_sbert_used=n_sbert_used,
        n_unverifiable=n_unverifiable,
        runtime_seconds=time.time() - t0,
        config=config,
    )
    return passing, failing, report
