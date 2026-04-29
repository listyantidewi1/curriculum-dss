"""
dashboard/api_public.py
=======================

Read-only public API for the competency recommender. Anonymous-friendly —
all endpoints in this router serve content from the *active published run*
(see ``dashboard/publish.py``). The admin's day-to-day pipeline runs do not
flicker the public surface; the public switches over only when admin publishes.

Endpoints (all under ``/api``):

    GET  /api/stages                     -> stage→KKNI map for filter chips
    GET  /api/published                  -> metadata about the active run
    GET  /api/competencies               -> filtered list of competencies
    GET  /api/competencies/{competency_id} -> single competency detail
    GET  /api/skills/top                 -> top demanded hard skills
    POST /api/coverage/analyze           -> upload curriculum, get coverage report
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

import kkni
from dashboard.publish import active_results_dir, get_active_published_run


router = APIRouter(prefix="/api", tags=["public"])


# --------------------------------------------------------------------------- #
# Lightweight cache: published runs are immutable so we cache by results_dir.
# --------------------------------------------------------------------------- #

_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_published_data(results_dir: Path) -> Dict[str, Any]:
    """Load competency JSON, skill education demand, and skill metadata for a
    published results directory. Cached per-directory."""
    key = str(results_dir.resolve())
    if key in _CACHE:
        return _CACHE[key]

    data: Dict[str, Any] = {
        "competencies": [],
        "skill_education_demand": pd.DataFrame(),
        "skill_education_summary": pd.DataFrame(),
        "verified_skills": pd.DataFrame(),
        "future_skill_weights": pd.DataFrame(),
        "recommendations": pd.DataFrame(),
    }

    cp = results_dir / "competency_proposals.json"
    if cp.exists():
        try:
            data["competencies"] = (
                json.loads(cp.read_text(encoding="utf-8")).get("competencies", []) or []
            )
        except Exception:
            data["competencies"] = []

    for name, key_name in [
        ("skill_education_demand.csv", "skill_education_demand"),
        ("skill_education_summary.csv", "skill_education_summary"),
        ("verified_skills.csv", "verified_skills"),
        ("future_skill_weights.csv", "future_skill_weights"),
        ("recommendations.csv", "recommendations"),
    ]:
        p = results_dir / name
        if p.exists():
            try:
                data[key_name] = pd.read_csv(p)
            except Exception:
                pass

    _CACHE[key] = data
    return data


def invalidate_cache() -> None:
    """Called from the admin publish handler to drop the cache."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _stage_to_kkni_levels(stage: Optional[str]) -> Optional[List[int]]:
    if not stage:
        return None
    levels = kkni.levels_for_stage(stage.strip())
    return levels or None


def _competency_to_card(c: Dict[str, Any]) -> Dict[str, Any]:
    """Pruned dict for the browse endpoint (cards)."""
    return {
        "id": c.get("id"),
        "batch_id": c.get("batch_id"),
        "title": c.get("title", ""),
        "description": c.get("description", ""),
        "kkni_level": c.get("kkni_level"),
        "kkni_descriptor": c.get("kkni_descriptor", ""),
        "future_domain": c.get("batch_domain", "") or c.get("future_domain", ""),
        "n_related_skills": len(c.get("related_skills", []) or []),
        "soft_skills_required": c.get("soft_skills_required", []) or [],
    }


def _competency_full(c: Dict[str, Any]) -> Dict[str, Any]:
    """Full payload for the detail endpoint."""
    return {
        "id": c.get("id"),
        "batch_id": c.get("batch_id"),
        "title": c.get("title", ""),
        "description": c.get("description", ""),
        "kkni_level": c.get("kkni_level"),
        "kkni_floor": c.get("kkni_floor"),
        "kkni_descriptor": c.get("kkni_descriptor", ""),
        "future_domain": c.get("batch_domain", "") or c.get("future_domain", ""),
        "future_relevance": c.get("future_relevance", ""),
        "related_skills": c.get("related_skills", []) or [],
        "soft_skills_required": c.get("soft_skills_required", []) or [],
        "soft_skills_description": c.get("soft_skills_description", ""),
        "occurrence_count": c.get("occurrence_count", 1),
    }


# --------------------------------------------------------------------------- #
# /api/stages
# --------------------------------------------------------------------------- #

@router.get("/stages")
def get_stages():
    """Return the stage chip definitions used by the public UI."""
    return {
        "stages": [
            {"key": stage, "kkni_levels": levels, "label": stage}
            for stage, levels in kkni.STAGE_TO_KKNI.items()
        ],
        "kkni_levels": [
            {
                "level": lvl,
                "label": kkni.kkni_label(lvl),
                "descriptor": kkni.kkni_descriptor(lvl),
                "stages": kkni.stages_for_level(lvl),
            }
            for lvl in range(kkni.KKNI_MIN_LEVEL, kkni.KKNI_MAX_LEVEL + 1)
        ],
    }


# --------------------------------------------------------------------------- #
# /api/published
# --------------------------------------------------------------------------- #

@router.get("/published")
def get_published_meta():
    run = get_active_published_run()
    if not run:
        return {"active": None, "message": "No run has been published yet."}
    return {
        "active": {
            "version_label": run["version_label"],
            "vocational_field": run["vocational_field"],
            "spektrum_code": run["spektrum_code"],
            "n_competencies": run["n_competencies"],
            "n_skills": run["n_skills"],
            "published_at": run["published_at"],
            "notes": run["notes"],
        }
    }


# --------------------------------------------------------------------------- #
# /api/competencies — filtered browse
# --------------------------------------------------------------------------- #

@router.get("/competencies")
def list_competencies(
    stage: Optional[str] = None,
    kkni_level: Optional[int] = None,
    domain: Optional[str] = None,
    bloom: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    """List competencies with optional filtering.

    stage: friendly stage chip (SMK, S1, S2, ...). Maps to a set of KKNI levels.
    kkni_level: explicit KKNI level (overrides stage if both are given).
    domain: future domain string match (case-insensitive substring).
    q: full-text substring match against title/description.
    """
    data = _load_published_data(active_results_dir())
    comps = data["competencies"]

    # KKNI filter (stage chip OR explicit level)
    levels_filter: Optional[set[int]] = None
    if kkni_level is not None:
        try:
            levels_filter = {int(kkni_level)}
        except (TypeError, ValueError):
            levels_filter = None
    elif stage:
        lvls = _stage_to_kkni_levels(stage)
        if lvls:
            levels_filter = set(lvls)

    domain_q = (domain or "").strip().lower()
    text_q = (q or "").strip().lower()
    bloom_q = (bloom or "").strip().lower()

    matched = []
    for c in comps:
        if levels_filter is not None:
            lvl = c.get("kkni_level")
            try:
                if int(lvl) not in levels_filter:
                    continue
            except (TypeError, ValueError):
                continue
        if domain_q:
            d = (c.get("batch_domain") or c.get("future_domain") or "").lower()
            if domain_q not in d:
                continue
        if bloom_q:
            # Inferred from related_skills list won't be present; skip if not annotated
            comp_bloom = (c.get("bloom") or c.get("bloom_level") or "").lower()
            if bloom_q not in comp_bloom:
                continue
        if text_q:
            blob = " ".join([
                str(c.get("title", "")),
                str(c.get("description", "")),
            ]).lower()
            if text_q not in blob:
                continue
        matched.append(c)

    total = len(matched)
    page = matched[max(0, offset): max(0, offset) + max(1, min(limit, 500))]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [_competency_to_card(c) for c in page],
    }


# --------------------------------------------------------------------------- #
# /api/competencies/{id}
# --------------------------------------------------------------------------- #

@router.get("/competencies/{competency_id}")
def get_competency(competency_id: str):
    data = _load_published_data(active_results_dir())
    comps = data["competencies"]
    # Match either bare id (e.g. "C5") or composite "C5_b1"
    for c in comps:
        cid = str(c.get("id", ""))
        bid = c.get("batch_id")
        composite = f"{cid}_b{bid}" if bid else cid
        if competency_id in (cid, composite):
            payload = _competency_full(c)
            payload["education_demand"] = _education_demand_for_skills(
                payload["related_skills"], data["skill_education_demand"]
            )
            return payload
    raise HTTPException(status_code=404, detail=f"Competency {competency_id} not found")


def _education_demand_for_skills(skills: List[str], demand_df: pd.DataFrame) -> Dict[int, int]:
    """Aggregate jobs-by-KKNI-level for the given skills."""
    if demand_df.empty or not skills:
        return {}
    keys = {str(s).strip().lower() for s in skills if str(s).strip()}
    df = demand_df.copy()
    df["_skill_lc"] = df["skill"].astype(str).str.strip().str.lower()
    sub = df[df["_skill_lc"].isin(keys) & df["kkni_level"].notna()]
    if sub.empty:
        return {}
    grouped = sub.groupby("kkni_level")["n_jobs"].sum()
    return {int(k): int(v) for k, v in grouped.items()}


# --------------------------------------------------------------------------- #
# /api/skills/top
# --------------------------------------------------------------------------- #

@router.get("/skills/top")
def top_skills(
    stage: Optional[str] = None,
    kkni_level: Optional[int] = None,
    limit: int = 50,
):
    """Top hard skills. Filtered by KKNI level via skill_education_demand."""
    data = _load_published_data(active_results_dir())
    summary = data["skill_education_summary"]
    if summary.empty:
        return {"items": [], "total": 0, "note": "skill_education_summary.csv is missing"}

    levels_filter: Optional[set[int]] = None
    if kkni_level is not None:
        try:
            levels_filter = {int(kkni_level)}
        except (TypeError, ValueError):
            levels_filter = None
    elif stage:
        lvls = _stage_to_kkni_levels(stage)
        if lvls:
            levels_filter = set(lvls)

    df = summary.copy()
    if levels_filter is not None:
        df = df[df["dominant_kkni"].isin(levels_filter)]
    df = df.sort_values("n_jobs_total", ascending=False).head(max(1, min(limit, 500)))

    items = []
    for _, row in df.iterrows():
        try:
            distribution = json.loads(row.get("education_distribution_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            distribution = {}
        items.append({
            "skill": row.get("skill"),
            "n_jobs_total": int(row.get("n_jobs_total", 0) or 0),
            "dominant_kkni": int(row.get("dominant_kkni"))
                if pd.notna(row.get("dominant_kkni")) else None,
            "education_distribution": {int(k): int(v) for k, v in distribution.items()},
        })
    return {"items": items, "total": len(items)}


# --------------------------------------------------------------------------- #
# /api/coverage/analyze — anonymous curriculum upload
# --------------------------------------------------------------------------- #

_CURRICULUM_TEXT_FIELDS = (
    "phrase", "phrases", "objective", "objectives",
    "content", "name", "title", "description", "topic",
)


def _normalize_phrase(text: str) -> str:
    """Cheap normalization: lowercase, collapse non-alphanum to single space."""
    if not isinstance(text, str):
        return ""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _parse_curriculum(content: bytes, filename: str) -> List[str]:
    """Extract a flat list of curriculum phrases from a CSV or JSON upload.

    Strategy:
        - JSON: walk all string values; if it's a dict with "phrases", use them.
        - CSV: collect all text from columns named like phrase/objective/content.
    """
    name = (filename or "").lower()
    if name.endswith(".json") or name.endswith(".jsonl"):
        try:
            obj = json.loads(content.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        return _walk_json_for_phrases(obj)

    # default: CSV
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")
    cols = [c for c in df.columns if str(c).strip().lower() in _CURRICULUM_TEXT_FIELDS]
    if not cols:
        # Fall back to the *first* text column
        for c in df.columns:
            if df[c].dtype == object:
                cols = [c]
                break
    phrases: List[str] = []
    for c in cols:
        for v in df[c].dropna().tolist():
            for token in re.split(r"[;,\n]+", str(v)):
                token = token.strip()
                if token:
                    phrases.append(token)
    return phrases


def _walk_json_for_phrases(obj: Any, out: Optional[List[str]] = None) -> List[str]:
    if out is None:
        out = []
    if isinstance(obj, str):
        if obj.strip():
            out.append(obj.strip())
    elif isinstance(obj, list):
        for item in obj:
            _walk_json_for_phrases(item, out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                key = str(k).lower()
                if key in _CURRICULUM_TEXT_FIELDS and v.strip():
                    out.append(v.strip())
            else:
                _walk_json_for_phrases(v, out)
    return out


def _coverage_report(curriculum_phrases: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
    """Compare curriculum against the published run's competencies + recommended skills."""
    norm_curriculum = {_normalize_phrase(p) for p in curriculum_phrases if p}
    norm_curriculum.discard("")

    comps = data.get("competencies") or []
    rec_df = data.get("recommendations")

    covered_competencies: List[Dict[str, Any]] = []
    uncovered_competencies: List[Dict[str, Any]] = []
    for c in comps:
        skills = [str(s).strip() for s in (c.get("related_skills") or []) if str(s).strip()]
        comp_norm = {_normalize_phrase(s) for s in skills}
        comp_norm.discard("")
        # Coverage = any related skill of the competency appears in curriculum,
        # OR the competency title/description shares a word with curriculum.
        title_norm = _normalize_phrase(c.get("title", ""))
        any_skill_in_curriculum = any(
            any(t in cur or cur in t for cur in norm_curriculum) for t in comp_norm
        ) if norm_curriculum and comp_norm else False
        title_match = any(
            (title_norm and (title_norm in cur or cur in title_norm))
            for cur in norm_curriculum
        )
        is_covered = bool(any_skill_in_curriculum or title_match)
        target = covered_competencies if is_covered else uncovered_competencies
        target.append({
            "id": c.get("id"),
            "title": c.get("title", ""),
            "kkni_level": c.get("kkni_level"),
            "future_domain": c.get("batch_domain", "") or c.get("future_domain", ""),
        })

    n_total = len(comps) or 1
    coverage_pct = round(100.0 * len(covered_competencies) / n_total, 1)

    # Top recommended hard skills not present in curriculum
    missing_skills: List[Dict[str, Any]] = []
    if isinstance(rec_df, pd.DataFrame) and not rec_df.empty and "skill" in rec_df.columns:
        score_col = "priority_score" if "priority_score" in rec_df.columns else None
        for _, row in rec_df.head(50).iterrows():
            skill = str(row.get("skill", "")).strip()
            if not skill:
                continue
            if _normalize_phrase(skill) in norm_curriculum:
                continue
            missing_skills.append({
                "skill": skill,
                "priority_score": float(row.get(score_col, 0) or 0) if score_col else None,
                "future_domain": row.get("best_future_domain", "") or row.get("domain", ""),
            })
            if len(missing_skills) >= 20:
                break

    return {
        "coverage_pct": coverage_pct,
        "n_competencies_total": len(comps),
        "n_competencies_covered": len(covered_competencies),
        "n_competencies_uncovered": len(uncovered_competencies),
        "covered_competencies": covered_competencies[:50],
        "uncovered_competencies": uncovered_competencies[:50],
        "missing_high_priority_skills": missing_skills,
        "n_curriculum_phrases": len(norm_curriculum),
    }


@router.post("/coverage/analyze")
async def coverage_analyze(
    request: Request,
    file: UploadFile = File(...),
    save_name: str = Form(""),
):
    """Anonymous: receive a curriculum upload, return a coverage report.

    If the user is logged in (session has 'user') and ``save_name`` is non-empty,
    the analysis is also persisted to ``coverage_analyses`` for later retrieval.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    phrases = _parse_curriculum(raw, file.filename or "")
    if not phrases:
        raise HTTPException(
            status_code=400,
            detail=(
                "Could not extract curriculum phrases. Use a CSV with a column named "
                "phrase/objective/content/topic, or a JSON file with phrase fields."
            ),
        )

    data = _load_published_data(active_results_dir())
    report = _coverage_report(phrases, data)

    saved_id: Optional[int] = None
    user = request.session.get("user") if hasattr(request, "session") else None
    if user and save_name.strip():
        from dashboard.db import exec_sql
        active_run = get_active_published_run()
        run_id = active_run["id"] if active_run else None
        saved_id = exec_sql(
            """
            INSERT INTO coverage_analyses(user_id, name, curriculum_blob, report_blob, published_run_id)
            VALUES(?, ?, ?, ?, ?)
            """,
            (
                int(user.get("id")),
                save_name.strip()[:200],
                json.dumps(phrases, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False),
                run_id,
            ),
        )

    return {
        "report": report,
        "saved_id": saved_id,
        "filename": file.filename,
    }


@router.get("/me/coverage")
def list_my_coverage(request: Request):
    user = request.session.get("user") if hasattr(request, "session") else None
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    from dashboard.db import q_all
    rows = q_all(
        "SELECT id, name, created_at, published_run_id FROM coverage_analyses "
        "WHERE user_id = ? ORDER BY id DESC LIMIT 100",
        (int(user.get("id")),),
    )
    return {"items": [dict(r) for r in rows]}


@router.get("/me/coverage/{analysis_id}")
def get_my_coverage(request: Request, analysis_id: int):
    user = request.session.get("user") if hasattr(request, "session") else None
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    from dashboard.db import q_one
    row = q_one(
        "SELECT * FROM coverage_analyses WHERE id = ? AND user_id = ?",
        (analysis_id, int(user.get("id"))),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    rec = dict(row)
    try:
        rec["report"] = json.loads(rec.pop("report_blob", "{}"))
    except (json.JSONDecodeError, TypeError):
        rec["report"] = {}
    try:
        rec["curriculum"] = json.loads(rec.pop("curriculum_blob", "[]"))
    except (json.JSONDecodeError, TypeError):
        rec["curriculum"] = []
    return rec
