from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from dashboard.db import exec_sql, hash_password, init_db, q_all, q_one, verify_password
from pipeline_orchestrator import (
    DEFAULT_RESULTS,
    department_paths,
    ensure_department_dirs,
    run_department_pipeline,
    run_department_phase2,
    run_load_more_samples,
)


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
templates = Jinja2Templates(directory=str(APP_ROOT / "templates"))

app = FastAPI(title="Admin & School Dashboard")
app.state.secret_key = os.environ.get("DASHBOARD_SECRET_KEY", "dev-only-insecure-key")
if app.state.secret_key == "dev-only-insecure-key":
    print("[WARN] Using default session secret. Set DASHBOARD_SECRET_KEY env var for production.")
app.add_middleware(SessionMiddleware, secret_key=app.state.secret_key)
app.mount("/dashboard/static", StaticFiles(directory=str(APP_ROOT / "static")), name="dashboard_static")


def _user(request: Request) -> Optional[dict]:
    return request.session.get("user")


def _require_user(request: Request) -> dict:
    user = _user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in")
    return user


def _require_role(request: Request, role: str) -> dict:
    user = _require_user(request)
    if user.get("role") != role:
        raise HTTPException(status_code=403, detail="Insufficient role")
    return user


def _dept_dirs(school_id: int, department_id: int) -> Dict[str, Path]:
    p = department_paths(school_id, department_id)
    ensure_department_dirs(p)
    return {
        "uploads": p.uploads,
        "preprocessing": p.preprocessing,
        "results": p.results,
        "feedback_store": p.feedback_store,
    }


def _phase2_completed(school_id: int, department_id: int) -> bool:
    """True if Phase 2 has been run for this department (calibrated_threshold exists)."""
    dirs = _dept_dirs(school_id, department_id)
    return (dirs["feedback_store"] / "calibrated_threshold.json").exists()


def _phase2_available(school_id: int, department_id: int) -> bool:
    """True if Phase 2 can be run (Phase 1 done, export for review exists)."""
    dirs = _dept_dirs(school_id, department_id)
    return (dirs["results"] / "expert_review_skills.csv").exists()


def _load_spektrum() -> dict:
    """Load Spektrum Keahlian taxonomy (Kepmen 244/M/2024) for admin UI."""
    for base in ("data", "DATA"):
        path = PROJECT_ROOT / base / "spektrum_keahlian" / "spektrum_keahlian.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"bidang": []}


def _resolve_results_dir(school_id: int, department_id: int) -> Path:
    dirs = _dept_dirs(school_id, department_id)
    dept_results = dirs["results"]
    if any(dept_results.glob("*.csv")) or any(dept_results.glob("*.json")):
        return dept_results
    return DEFAULT_RESULTS


def _load_table(path: Path, limit: int = 100) -> List[dict]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path).head(limit)
        return df.to_dict(orient="records")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "competencies" in data:
            return data["competencies"][:limit]
        if isinstance(data, list):
            return data[:limit]
        return [data]
    return []


def _majority(values: List[str]) -> str:
    vals = [str(v).strip().lower() for v in values if str(v).strip()]
    if not vals:
        return ""
    counts: Dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]


def _resolve_feedback_dir(school_id: int, department_id: int, result_dir: Path) -> Path:
    dirs = _dept_dirs(school_id, department_id)
    dept_fb = dirs["feedback_store"]
    if any(dept_fb.glob("*feedback*")) or any(dept_fb.glob("*verified*")) or any(dept_fb.glob("*assessments*")):
        return dept_fb
    # fallback to global feedback_store for default/demo data
    return PROJECT_ROOT / "feedback_store"


def _review_status_maps(result_dir: Path, feedback_dir: Path) -> Dict[str, Dict[str, str]]:
    skill_status: Dict[str, str] = {}
    knowledge_status: Dict[str, str] = {}
    competency_status: Dict[str, str] = {}

    # Skills: review_id -> skill from template, majority vote from skill_feedback.csv
    skill_tpl = result_dir / "expert_review_skills.csv"
    skill_fb = feedback_dir / "skill_feedback.csv"
    if skill_tpl.exists() and skill_fb.exists():
        tpl = pd.read_csv(skill_tpl)
        fb = pd.read_csv(skill_fb)
        if {"review_id", "skill"}.issubset(tpl.columns) and {"review_id", "human_valid"}.issubset(fb.columns):
            merged = fb.merge(tpl[["review_id", "skill"]], on="review_id", how="left")
            for skill, grp in merged.groupby("skill"):
                skill = str(skill).strip().lower()
                if not skill:
                    continue
                hv = _majority(grp["human_valid"].astype(str).tolist())
                if hv in {"valid", "invalid"}:
                    skill_status[skill] = "verified" if hv == "valid" else "invalid"

    # Knowledge: review_id -> knowledge
    know_tpl = result_dir / "expert_review_knowledge.csv"
    know_fb = feedback_dir / "knowledge_feedback.csv"
    if know_tpl.exists() and know_fb.exists():
        tpl = pd.read_csv(know_tpl)
        fb = pd.read_csv(know_fb)
        if {"review_id", "knowledge"}.issubset(tpl.columns) and {"review_id", "human_valid"}.issubset(fb.columns):
            merged = fb.merge(tpl[["review_id", "knowledge"]], on="review_id", how="left")
            for knowledge, grp in merged.groupby("knowledge"):
                knowledge = str(knowledge).strip().lower()
                if not knowledge:
                    continue
                hv = _majority(grp["human_valid"].astype(str).tolist())
                if hv in {"valid", "invalid"}:
                    knowledge_status[knowledge] = "verified" if hv == "valid" else "invalid"

    # Competencies: assessments json preferred, fallback competency_feedback.csv
    assess_path = feedback_dir / "competency_assessments.json"
    if assess_path.exists():
        try:
            data = json.loads(assess_path.read_text(encoding="utf-8"))
            for cid, val in data.items():
                rel = str(val.get("relevant", "")).strip().lower()
                if rel:
                    competency_status[str(cid)] = "verified"
        except Exception:
            pass
    comp_fb = feedback_dir / "competency_feedback.csv"
    if comp_fb.exists():
        fb = pd.read_csv(comp_fb)
        if {"competency_id", "human_quality", "human_relevant"}.issubset(fb.columns):
            for cid, grp in fb.groupby("competency_id"):
                cid = str(cid).strip()
                if not cid:
                    continue
                hq = _majority(grp["human_quality"].astype(str).tolist())
                hr = _majority(grp["human_relevant"].astype(str).tolist())
                if hq or hr:
                    competency_status[cid] = "verified"

    return {
        "skills": skill_status,
        "knowledge": knowledge_status,
        "competencies": competency_status,
    }


def _review_coverage(result_dir: Path, feedback_dir: Path) -> Dict[str, Dict[str, int]]:
    cov = {
        "skills": {"reviewed": 0, "total": 0},
        "knowledge": {"reviewed": 0, "total": 0},
        "competencies": {"reviewed": 0, "total": 0},
    }

    skill_tpl = result_dir / "expert_review_skills.csv"
    skill_fb = feedback_dir / "skill_feedback.csv"
    if skill_tpl.exists():
        tpl = pd.read_csv(skill_tpl)
        if "review_id" in tpl.columns:
            cov["skills"]["total"] = int(tpl["review_id"].nunique())
            if skill_fb.exists():
                fb = pd.read_csv(skill_fb)
                if {"review_id", "human_valid", "human_type", "human_bloom", "human_notes"}.intersection(set(fb.columns)):
                    cols = [c for c in ["human_valid", "human_type", "human_bloom", "human_notes"] if c in fb.columns]
                    if cols and "review_id" in fb.columns:
                        f2 = fb[["review_id"] + cols].copy()
                        reviewed = f2.apply(
                            lambda r: any(str(r.get(c, "")).strip() for c in cols),
                            axis=1,
                        )
                        cov["skills"]["reviewed"] = int(f2.loc[reviewed, "review_id"].astype(str).nunique())

    know_tpl = result_dir / "expert_review_knowledge.csv"
    know_fb = feedback_dir / "knowledge_feedback.csv"
    if know_tpl.exists():
        tpl = pd.read_csv(know_tpl)
        if "review_id" in tpl.columns:
            cov["knowledge"]["total"] = int(tpl["review_id"].nunique())
            if know_fb.exists():
                fb = pd.read_csv(know_fb)
                cols = [c for c in ["human_valid", "human_notes"] if c in fb.columns]
                if cols and "review_id" in fb.columns:
                    f2 = fb[["review_id"] + cols].copy()
                    reviewed = f2.apply(
                        lambda r: any(str(r.get(c, "")).strip() for c in cols),
                        axis=1,
                    )
                    cov["knowledge"]["reviewed"] = int(f2.loc[reviewed, "review_id"].astype(str).nunique())

    comp_tpl = result_dir / "expert_review_competencies.csv"
    comp_fb = feedback_dir / "competency_feedback.csv"
    if comp_tpl.exists():
        tpl = pd.read_csv(comp_tpl)
        if "competency_id" in tpl.columns:
            cov["competencies"]["total"] = int(tpl["competency_id"].astype(str).nunique())
            if comp_fb.exists():
                fb = pd.read_csv(comp_fb)
                cols = [c for c in ["human_quality", "human_relevant", "human_notes"] if c in fb.columns]
                if cols and "competency_id" in fb.columns:
                    f2 = fb[["competency_id"] + cols].copy()
                    reviewed = f2.apply(
                        lambda r: any(str(r.get(c, "")).strip() for c in cols),
                        axis=1,
                    )
                    cov["competencies"]["reviewed"] = int(f2.loc[reviewed, "competency_id"].astype(str).nunique())
    return cov


def _load_inter_rater_report(feedback_dir: Path) -> Dict[str, dict]:
    path = feedback_dir / "inter_rater_report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _knowledge_future_weight_map(result_dir: Path) -> Dict[str, float]:
    path = result_dir / "future_skill_weights_dummy.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "knowledge" not in df.columns or "future_weight" not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        k = str(r.get("knowledge", "")).strip().lower()
        k_norm = _normalize_knowledge_key(k)
        if not k:
            continue
        fw = _to_float(r.get("future_weight"))
        out[k] = max(out.get(k, 0.0), fw)
        if k_norm:
            out[k_norm] = max(out.get(k_norm, 0.0), fw)
    return out


def _merge_template_with_reviewer_feedback(
    template_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    id_col: str,
    reviewer_id: str,
    feedback_cols: List[str],
) -> pd.DataFrame:
    if template_df.empty:
        return template_df
    df = template_df.copy()
    for c in feedback_cols:
        if c not in df.columns:
            df[c] = ""
        else:
            df[c] = ""
    if feedback_df.empty or "reviewer_id" not in feedback_df.columns:
        return df
    rev = feedback_df[feedback_df["reviewer_id"].astype(str) == str(reviewer_id)].copy()
    if rev.empty or id_col not in rev.columns:
        return df
    for _, row in rev.iterrows():
        rid = str(row.get(id_col, "")).strip()
        if not rid:
            continue
        mask = df[id_col].astype(str) == rid
        for c in feedback_cols:
            if c in rev.columns and c in df.columns:
                val = row.get(c, "")
                if pd.notna(val):
                    df.loc[mask, c] = str(val)
    return df


def _upsert_feedback_csv(
    feedback_dir: Path,
    name: str,
    id_col: str,
    row_id: str,
    reviewer_id: str,
    updates: Dict[str, str],
) -> None:
    path = feedback_dir / name
    if path.exists():
        df = pd.read_csv(path)
    else:
        cols = [id_col, "reviewer_id"] + list(updates.keys())
        df = pd.DataFrame(columns=cols)
    for c in [id_col, "reviewer_id"] + list(updates.keys()):
        if c not in df.columns:
            df[c] = ""
    mask = (df[id_col].astype(str) == str(row_id)) & (df["reviewer_id"].astype(str) == str(reviewer_id))
    if mask.any():
        for k, v in updates.items():
            df.loc[mask, k] = v
    else:
        row = {id_col: row_id, "reviewer_id": reviewer_id}
        row.update(updates)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _load_feedback_csv(feedback_dir: Path, name: str) -> pd.DataFrame:
    path = feedback_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_float(value) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _prioritize_recommendations(
    rows: List[dict],
    skill_status: Optional[Dict[str, str]] = None,
    ranking_mode: str = "human_adjusted",
    top_n: Optional[int] = 20,
) -> List[dict]:
    if not rows:
        return []
    skill_status = skill_status or {}

    def _scores(row: dict):
        p = _to_float(row.get("priority_score"))
        skill_name = str(row.get("skill", "")).strip().lower()
        status = skill_status.get(skill_name, "not_verified")
        status_boost = 0.05 if status == "verified" else (-0.15 if status == "invalid" else 0.0)
        model_score = p
        human_score = p + status_boost
        return model_score, human_score

    def _key(row: dict):
        model_score, human_score = _scores(row)
        # Secondary sort: lower rank means higher priority
        r = _to_float(row.get("rank")) if row.get("rank") is not None else 1e9
        # Tertiary sort: demand
        d = _to_float(row.get("demand_freq"))
        primary = human_score if ranking_mode == "human_adjusted" else model_score
        return (-primary, r, -d)

    sorted_rows = []
    for row in sorted(rows, key=_key):
        item = dict(row)
        skill_name = str(item.get("skill", "")).strip().lower()
        item["verification_status"] = skill_status.get(skill_name, "not_verified")
        model_score, human_score = _scores(row)
        item["_model_score"] = round(model_score, 4)
        item["_human_adjusted_score"] = round(human_score, 4)
        # Data provenance: why this item is recommended
        parts = []
        df = row.get("demand_freq")
        if df is not None and str(df).strip():
            parts.append(f"Demand: {df} jobs")
        tl = str(row.get("trend_label", "")).strip()
        if tl:
            parts.append(f"Trend: {tl}")
        fw = row.get("future_domain") or row.get("best_future_domain") or row.get("future_weight")
        if fw is not None and str(fw).strip():
            parts.append(f"Future: {fw}")
        src = str(row.get("source", "")).strip()
        if src:
            parts.append(f"Source: {src}")
        item["_why_provenance"] = " | ".join(parts) if parts else ""
        sorted_rows.append(item)
    if top_n is None or top_n <= 0:
        result = sorted_rows
    else:
        result = sorted_rows[:top_n]
    # Recompute rank as display order (1, 2, 3...) so it matches the sorted order
    for i, row in enumerate(result, start=1):
        row["rank"] = i
    return result


def _prioritize_competencies(
    comps: List[dict],
    recs: List[dict],
    competency_status: Optional[Dict[str, str]] = None,
    competency_assessments: Optional[Dict[str, dict]] = None,
    ranking_mode: str = "human_adjusted",
    top_n: Optional[int] = 20,
) -> List[dict]:
    if not comps:
        return []
    competency_status = competency_status or {}
    competency_assessments = competency_assessments or {}

    # Build skill -> recommendation priority map for competency ranking.
    skill_priority = {}
    for row in recs:
        skill = str(row.get("skill", "")).strip().lower()
        if not skill:
            continue
        if ranking_mode == "human_adjusted":
            score = _to_float(row.get("_human_adjusted_score")) or _to_float(row.get("priority_score"))
        else:
            score = _to_float(row.get("_model_score")) or _to_float(row.get("priority_score"))
        if skill not in skill_priority or score > skill_priority[skill]:
            skill_priority[skill] = score

    ranked = []
    for c in comps:
        related = c.get("related_skills", []) or []
        if not isinstance(related, list):
            related = []
        vals = [skill_priority.get(str(s).strip().lower(), 0.0) for s in related]
        # Competency priority = max related-skill priority + post-review quality/relevance adjustments.
        c_score = max(vals) if vals else 0.0
        cid = str(c.get("id", "")).strip()
        assessment = competency_assessments.get(cid, {})
        q = str(assessment.get("quality", "")).strip().lower()
        q_map = {"1": 0.2, "2": 0.4, "3": 0.6, "4": 0.8, "5": 1.0, "poor": 0.2, "fair": 0.4, "good": 0.7, "excellent": 1.0}
        q_score = q_map.get(q, 0.0)
        rel = str(assessment.get("relevant", "")).strip().lower()
        rel_adj = 0.2 if rel == "yes" else (0.08 if rel == "partial" else (-0.2 if rel == "no" else 0.0))
        verify_adj = 0.12 if competency_status.get(cid) == "verified" else 0.0
        model_score = c_score
        human_score = (0.7 * c_score) + (0.3 * q_score) + rel_adj + verify_adj
        c_score = human_score if ranking_mode == "human_adjusted" else model_score
        item = dict(c)
        item["_priority_score"] = round(c_score, 4)
        item["_model_score"] = round(model_score, 4)
        item["_human_adjusted_score"] = round(human_score, 4)
        item["verification_status"] = competency_status.get(cid, "not_verified")
        # Data provenance: why this competency is recommended
        rel_skills = related[:5] if len(related) > 5 else related
        prov = f"From related skills: {', '.join(rel_skills)}" if rel_skills else "Generated from verified skills + future weights + LLM"
        item["_why_provenance"] = prov
        ranked.append(item)

    ranked.sort(key=lambda x: _to_float(x.get("_priority_score")), reverse=True)
    if top_n is None or top_n <= 0:
        result = ranked
    else:
        result = ranked[:top_n]
    # Add display rank (1, 2, 3...) to match sorted order
    for i, row in enumerate(result, start=1):
        row["rank"] = i
    return result


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 4}


def _normalize_knowledge_key(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip().lower()
    # Normalize punctuation/noise so repeated variants are grouped.
    t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _aggregate_knowledge_rows(rows: List[dict]) -> List[dict]:
    if not rows:
        return []
    grouped: Dict[str, dict] = {}
    for r in rows:
        raw = str(r.get("knowledge", "")).strip()
        key = _normalize_knowledge_key(raw)
        if not key:
            continue
        conf = _to_float(r.get("confidence_score", r.get("mean_confidence")))
        job_id = str(r.get("job_id", "")).strip()
        src = str(r.get("source", "")).strip()
        if key not in grouped:
            grouped[key] = {
                "knowledge": raw,
                "_conf_sum": 0.0,
                "_conf_n": 0,
                "_jobs": set(),
                "_sources": set(),
                "occurrence_count": 0,
            }
        g = grouped[key]
        g["occurrence_count"] += 1
        g["_conf_sum"] += conf
        g["_conf_n"] += 1
        if job_id:
            g["_jobs"].add(job_id)
        if src:
            g["_sources"].add(src)

    out: List[dict] = []
    for key, g in grouped.items():
        n = max(int(g["_conf_n"]), 1)
        mean_conf = g["_conf_sum"] / n
        item = {
            "knowledge": g["knowledge"],
            "confidence_score": round(mean_conf, 4),
            "occurrence_count": int(g["occurrence_count"]),
            "job_count": len(g["_jobs"]),
            "sources": ", ".join(sorted(g["_sources"])) if g["_sources"] else "",
            "_knowledge_key": key,
        }
        out.append(item)
    out.sort(key=lambda x: (-int(x.get("occurrence_count", 0)), -_to_float(x.get("confidence_score"))))
    return out


def _prioritize_knowledge_rows(
    rows: List[dict],
    knowledge_status: Optional[Dict[str, str]] = None,
    future_weight_map: Optional[Dict[str, float]] = None,
    ranking_mode: str = "human_adjusted",
    top_n: Optional[int] = 30,
) -> List[dict]:
    if not rows:
        return []
    knowledge_status = knowledge_status or {}
    future_weight_map = future_weight_map or {}
    max_fw = max(future_weight_map.values()) if future_weight_map else 0.0
    max_occ = max([_to_float(r.get("occurrence_count", r.get("freq", 1))) for r in rows] or [1.0])
    out = []
    for r in rows:
        item = dict(r)
        k_text = str(item.get("knowledge", "")).strip()
        k = k_text.lower()
        k_norm = _normalize_knowledge_key(k_text)
        conf = _to_float(item.get("confidence_score"))
        fw = _to_float(future_weight_map.get(k, future_weight_map.get(k_norm, 0.0)))
        fw_norm = (fw / max_fw) if max_fw > 0 else 0.0
        status = knowledge_status.get(k, knowledge_status.get(k_norm, "not_verified"))
        occ = _to_float(item.get("occurrence_count", item.get("freq", 1)))
        occ_norm = (occ / max_occ) if max_occ > 0 else 0.0
        status_adj = 0.12 if status == "verified" else (-0.18 if status == "invalid" else 0.0)
        model_score = (0.45 * conf) + (0.25 * fw_norm) + (0.30 * occ_norm)
        human_score = model_score + status_adj
        score = human_score if ranking_mode == "human_adjusted" else model_score
        item["future_weight"] = round(fw, 4)
        item["occurrence_count"] = int(occ)
        item["verification_status"] = status
        item["_ranking_score"] = round(score, 4)
        item["_model_score"] = round(model_score, 4)
        item["_human_adjusted_score"] = round(human_score, 4)
        parts = [f"Occurrence: {int(occ)}", f"Confidence: {round(conf, 2)}"]
        if fw > 0:
            parts.append(f"Future weight: {round(fw, 2)}")
        item["_why_provenance"] = " | ".join(parts)
        out.append(item)
    out.sort(key=lambda x: _to_float(x.get("_ranking_score")), reverse=True)
    if top_n is None or top_n <= 0:
        result = out
    else:
        result = out[:top_n]
    # Add display rank (1, 2, 3...) to match sorted order
    for i, row in enumerate(result, start=1):
        row["rank"] = i
    return result


def _enrich_competencies_with_knowledge(comps: List[dict], knowledge_rows: List[dict]) -> List[dict]:
    """
    Attach related_knowledge to each competency by lightweight lexical overlap.
    """
    if not comps:
        return []

    knowledge_items = []
    for k in knowledge_rows:
        text = str(k.get("knowledge", "")).strip()
        if not text:
            continue
        knowledge_items.append(
            {
                "text": text,
                "tokens": _tokenize(text),
                "score": _to_float(k.get("confidence_score")),
            }
        )

    enriched = []
    for c in comps:
        related_skills = c.get("related_skills", []) or []
        if not isinstance(related_skills, list):
            related_skills = []
        skill_tokens = set()
        for s in related_skills:
            skill_tokens |= _tokenize(str(s))

        # Also include title/description context.
        skill_tokens |= _tokenize(str(c.get("title", "")))
        skill_tokens |= _tokenize(str(c.get("description", "")))

        matches = []
        for k in knowledge_items:
            overlap = len(skill_tokens & k["tokens"])
            if overlap > 0:
                matches.append((overlap, k["score"], k["text"]))
        matches.sort(key=lambda x: (-x[0], -x[1], x[2]))

        item = dict(c)
        item["related_knowledge"] = [m[2] for m in matches[:8]]
        enriched.append(item)
    return enriched


def _explanations() -> Dict[str, Dict[str, str]]:
    """Registry of explanations for scores, statuses, and concepts (XAI transparency)."""
    return {
        "priority_score": {
            "title": "Priority Score",
            "what": "Combines three signals: job-market demand (40%), time-trend (30%), and future relevance (30%). Coverage is for insights only, not prioritization—existing curriculum may be outdated.",
            "formula": "0.40×demand + 0.30×trend + 0.30×future",
            "limitations": "Weights are fixed; ablation study can show per-signal impact.",
        },
        "_model_score": {
            "title": "Model Score",
            "what": "Pipeline score before human feedback. For skills: priority_score. For knowledge: 0.45×confidence + 0.25×future_weight + 0.30×occurrence. For competencies: max of related-skill priorities.",
            "formula": "",
            "limitations": "Does not reflect expert review.",
        },
        "_human_adjusted_score": {
            "title": "Human-Adjusted Score",
            "what": "Model score plus expert feedback: +0.05 if verified, -0.15 if invalid, 0 if not yet reviewed. Used when ranking mode is human_adjusted.",
            "formula": "model_score + status_boost (verified: +0.05, invalid: -0.15)",
            "limitations": "Requires human review; unverified items get neutral treatment.",
        },
        "verification_status": {
            "title": "Verification Status",
            "what": "verified = expert marked valid; invalid = expert marked invalid; not_verified = not yet reviewed. With multiple reviewers, majority vote applies.",
            "formula": "",
            "limitations": "Not all items are in the review sample; not_verified items may still be valid.",
        },
        "ranking_mode": {
            "title": "Ranking Mode",
            "what": "model_only: use pipeline scores only; human_adjusted: boost verified items, penalize invalid ones. Choose human_adjusted when you have reviewed samples.",
            "formula": "",
            "limitations": "",
        },
        "future_weight": {
            "title": "Future Weight",
            "what": "Maps skills/knowledge to forecast domains (WEF, O*NET, McKinsey) via embeddings. Combines similarity to domain examples with domain growth trend.",
            "formula": "similarity × trend_score",
            "limitations": "Embedding-based; mapping_margin indicates confidence (large = confident).",
        },
        "human_valid": {
            "title": "Validity (Your Review)",
            "what": "Mark valid if the skill/knowledge is correct and relevant; invalid if wrong or not a real skill/knowledge item.",
            "formula": "",
            "limitations": "",
        },
        "human_type": {
            "title": "Skill Type (Your Review)",
            "what": "Hard = technical; Soft = behavioral; Both = hybrid. Overrides model classification.",
            "formula": "",
            "limitations": "",
        },
        "human_bloom": {
            "title": "Bloom Level (Your Review)",
            "what": "Cognitive level: Remember, Understand, Apply, Analyze, Evaluate, Create, or N/A. Overrides model classification.",
            "formula": "",
            "limitations": "",
        },
        "human_quality": {
            "title": "Quality (Your Review)",
            "what": "Numbers and text are equivalent: 1=poor, 2=fair, 3=good, 4/5=excellent. Choose either; both are used the same in ranking.",
            "formula": "",
            "limitations": "",
        },
        "human_relevant": {
            "title": "Relevance (Your Review)",
            "what": "yes = relevant to curriculum; partial = partly; no = not relevant. Affects competency ranking.",
            "formula": "",
            "limitations": "",
        },
        "human_skill_focus": {
            "title": "Skill Focus (Your Review)",
            "what": "Hard = technical competency; Soft = interpersonal/behavioral; Both = hybrid. Derived from related skills; correct if needed.",
            "formula": "",
            "limitations": "",
        },
        "confidence_score": {
            "title": "Confidence Score",
            "what": "Model confidence (0–1) that this extraction is correct. Higher = more confident.",
            "formula": "",
            "limitations": "",
        },
        "verification_level": {
            "title": "Verification Level",
            "what": "Auto-assigned: Verified_HIGH/MEDIUM (above threshold), Not_Verified_LOW (below). Uses calibrated threshold when available.",
            "formula": "",
            "limitations": "",
        },
        "trend_label": {
            "title": "Trend Label",
            "what": "Emerging = increasing in job postings over time; Declining = decreasing; Stable = no significant change; Unknown = no trend data. Requires job dates and sufficient time span.",
            "formula": "",
            "limitations": "",
        },
        "source": {
            "title": "Extraction Source",
            "what": "BERT+LLM = hybrid (both agreed); BERT-only or LLM-only = single model. Hybrid typically more reliable.",
            "formula": "",
            "limitations": "",
        },
    }


def _plot_explanations() -> Dict[str, Dict[str, str]]:
    return {
        "skills_knowledge_total_per_model.png": {
            "title": "Total Skills & Knowledge by Model",
            "what": "Compares how many skill and knowledge items each extraction model produced.",
            "how": "Higher bars mean more extracted items. Compare Hybrid vs single models to see extraction coverage.",
        },
        "skills_knowledge_mean_per_job.png": {
            "title": "Average Skills/Knowledge per Job",
            "what": "Shows average extracted items per posting for each model.",
            "how": "Use this to understand extraction density. Extremely high values may indicate over-extraction.",
        },
        "bloom_distribution_hard_skills.png": {
            "title": "Bloom Distribution (Hard Skills)",
            "what": "Distribution of hard skills across Bloom cognitive levels.",
            "how": "Balanced upper levels (Analyze/Evaluate/Create) indicate stronger advanced competency demand.",
        },
        "heatmap_curriculum_bloom_hard_skills.png": {
            "title": "Curriculum × Bloom Heatmap",
            "what": "Maps demanded hard skills to curriculum components and Bloom levels.",
            "how": "Darker cells mean higher demand concentration. Sparse areas can indicate curriculum gaps.",
        },
        "clusters_top_hard_skills.png": {
            "title": "Top Hard Skill Clusters",
            "what": "Groups the most frequent hard skills into semantic clusters.",
            "how": "Large/central groups represent common competency themes for curriculum prioritization.",
        },
        "clusters_top_soft_skills.png": {
            "title": "Top Soft Skill Clusters",
            "what": "Groups top soft skills into behavioral competency themes.",
            "how": "Use clusters to define transversal competencies (communication, teamwork, leadership, etc.).",
        },
        "clusters_top_knowledge.png": {
            "title": "Top Knowledge Clusters",
            "what": "Groups the most frequent knowledge items.",
            "how": "Cluster themes help identify core knowledge domains that should be taught first.",
        },
        "clusters_demanded_not_covered.png": {
            "title": "Demanded but Not Covered (Insight)",
            "what": "Highlights demanded skills that do not map strongly to current curriculum components. For insights only.",
            "how": "Use to identify curriculum gaps. These inform curriculum design; prioritization uses demand, trend, and future weight.",
        },
        "coverage_distribution_hybrid.png": {
            "title": "Coverage Distribution (Insight)",
            "what": "Distribution of how much existing curriculum covers extracted skills per job. For insights only—not used for prioritization.",
            "how": "Right-shifted distribution means better alignment. Use to assess curriculum gaps; schools design better curriculum from recommendations.",
        },
        "top_future_weight_knowledge.png": {
            "title": "Top Future-Weighted Knowledge",
            "what": "Ranks knowledge items by future relevance weight.",
            "how": "Higher values imply stronger future-work importance; prioritize these in course design.",
        },
        "top_future_weight_skills.png": {
            "title": "Top Future-Weighted Skills",
            "what": "Ranks skills by future relevance weight (domain forecast alignment).",
            "how": "Higher values imply stronger future-work importance; prioritize these in curriculum.",
        },
        "emerging_skills_coverage.png": {
            "title": "Emerging Skills Coverage (Insight)",
            "what": "Shows which emerging skills are covered vs not covered by existing curriculum.",
            "how": "Green = covered; orange = not covered. Use for insights only; prioritization uses demand, trend, and future weight.",
        },
    }


def _plot_info_for_filename(filename: str) -> Dict[str, str]:
    meta = _plot_explanations().get(filename, {})
    return {
        "title": meta.get("title", filename),
        "what": meta.get("what", "This plot summarizes a pipeline metric for this department."),
        "how": meta.get("how", "Read axes/labels first, then compare relative magnitudes across categories."),
    }


def _load_competency_assessments(feedback_dir: Path) -> Dict[str, dict]:
    path = feedback_dir / "competency_assessments.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/")
def root_redirect():
    return RedirectResponse("/dashboard/login", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_home(request: Request):
    user = _user(request)
    if user:
        if user.get("role") == "admin":
            return RedirectResponse("/dashboard/admin/schools", status_code=302)
        return RedirectResponse("/dashboard/school/runs", status_code=302)
    return RedirectResponse("/dashboard/login", status_code=302)


@app.get("/dashboard/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@app.post("/dashboard/login", response_class=HTMLResponse)
def login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
    row = q_one("SELECT id, email, role, school_id, password_hash FROM users WHERE email=?", (email.strip(),))
    if not row or not verify_password(password, row["password_hash"]):
        return templates.TemplateResponse(
            "login.html", {"request": request, "error": "Invalid credentials"}
        )
    request.session["user"] = {
        "id": row["id"],
        "email": row["email"],
        "role": row["role"],
        "school_id": row["school_id"],
    }
    return RedirectResponse("/dashboard", status_code=302)


@app.get("/dashboard/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/dashboard/login", status_code=302)


@app.get("/dashboard/admin/schools", response_class=HTMLResponse)
def admin_schools(request: Request):
    _require_role(request, "admin")
    schools = q_all("SELECT * FROM schools ORDER BY id DESC")
    departments = q_all(
        """
        SELECT d.*, s.name AS school_name
        FROM departments d
        JOIN schools s ON s.id = d.school_id
        ORDER BY d.id DESC
        """
    )
    spektrum = _load_spektrum()
    return templates.TemplateResponse(
        "admin/schools.html",
        {"request": request, "schools": schools, "departments": departments, "spektrum": spektrum, "user": _user(request)},
    )


@app.post("/dashboard/admin/schools/create")
def admin_create_school(request: Request, school_name: str = Form(...)):
    _require_role(request, "admin")
    exec_sql("INSERT INTO schools(name) VALUES(?)", (school_name.strip(),))
    return RedirectResponse("/dashboard/admin/schools", status_code=302)


@app.post("/dashboard/admin/departments/create")
def admin_create_department(
    request: Request,
    school_id: int = Form(...),
    department_name: str = Form(...),
    vocational_field: str = Form(...),
    spektrum_code: Optional[str] = Form(None),
):
    _require_role(request, "admin")
    vf = vocational_field.strip()
    sc = spektrum_code.strip() if spektrum_code and str(spektrum_code).strip() else None
    dept_id = exec_sql(
        "INSERT INTO departments(school_id, name, vocational_field, spektrum_code) VALUES(?, ?, ?, ?)",
        (school_id, department_name.strip(), vf, sc),
    )
    _dept_dirs(school_id, dept_id)
    return RedirectResponse("/dashboard/admin/schools", status_code=302)


@app.get("/dashboard/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    _require_role(request, "admin")
    users = q_all(
        """
        SELECT u.id, u.email, u.role, u.created_at, s.name AS school_name
        FROM users u
        LEFT JOIN schools s ON s.id = u.school_id
        ORDER BY u.id DESC
        """
    )
    schools = q_all("SELECT * FROM schools ORDER BY name")
    return templates.TemplateResponse(
        "admin/users.html",
        {"request": request, "users": users, "schools": schools, "user": _user(request)},
    )


@app.post("/dashboard/admin/users/create")
def admin_create_user(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    school_id: Optional[int] = Form(None),
):
    _require_role(request, "admin")
    if role == "school" and not school_id:
        raise HTTPException(status_code=400, detail="school_id is required for school users")
    exec_sql(
        "INSERT INTO users(email, password_hash, role, school_id) VALUES(?, ?, ?, ?)",
        (email.strip(), hash_password(password), role, school_id),
    )
    return RedirectResponse("/dashboard/admin/users", status_code=302)


@app.get("/dashboard/admin/runs", response_class=HTMLResponse)
def admin_runs(request: Request):
    _require_role(request, "admin")
    runs = q_all(
        """
        SELECT r.*, d.name AS department_name, s.name AS school_name
        FROM runs r
        JOIN departments d ON d.id = r.department_id
        JOIN schools s ON s.id = d.school_id
        ORDER BY r.id DESC
        """
    )
    irr_rows = []
    depts = q_all(
        """
        SELECT d.id, d.school_id, d.name AS department_name, s.name AS school_name
        FROM departments d
        JOIN schools s ON s.id = d.school_id
        ORDER BY s.name, d.name
        """
    )
    for d in depts:
        rd = _resolve_results_dir(d["school_id"], d["id"])
        fd = _resolve_feedback_dir(d["school_id"], d["id"], rd)
        rep = _load_inter_rater_report(fd)
        if not rep:
            continue
        for src, vals in rep.items():
            irr_rows.append(
                {
                    "school_name": d["school_name"],
                    "department_name": d["department_name"],
                    "source": src,
                    "shared_items": vals.get("shared_items", 0),
                    "cohens_kappa": vals.get("cohens_kappa", 0),
                    "pct_agreement": vals.get("pct_agreement", 0),
                }
            )
    return templates.TemplateResponse(
        "admin/runs.html",
        {"request": request, "runs": runs, "irr_rows": irr_rows, "user": _user(request)},
    )


def _school_departments(user: dict) -> List:
    return q_all("SELECT * FROM departments WHERE school_id=? ORDER BY name", (user["school_id"],))


@app.get("/dashboard/school/runs", response_class=HTMLResponse)
def school_runs(request: Request, department_id: Optional[int] = None):
    user = _require_role(request, "school")
    departments = _school_departments(user)
    if not departments:
        return templates.TemplateResponse(
            "school/runs.html",
            {
                "request": request,
                "runs": [],
                "departments": [],
                "selected_department_id": None,
                "banner": "Ask admin to create departments for your school first.",
                "user": user,
            },
        )

    selected = department_id or departments[0]["id"]
    runs_raw = q_all("SELECT * FROM runs WHERE department_id=? ORDER BY id DESC", (selected,))
    runs = []
    for r in runs_raw:
        row = dict(r)
        try:
            cfg = json.loads(r["config_snapshot"] or "{}")
            row["phase"] = cfg.get("phase", "phase1")
        except Exception:
            row["phase"] = "phase1"
        runs.append(row)
    phase2_available = _phase2_available(user["school_id"], selected)
    has_running = any(str(r.get("status", "")).lower() == "running" for r in runs)
    return templates.TemplateResponse(
        "school/runs.html",
        {
            "request": request,
            "runs": runs,
            "departments": departments,
            "selected_department_id": selected,
            "phase2_available": phase2_available,
            "has_running": has_running,
            "banner": "",
            "user": user,
        },
    )


def _run_worker(run_id: int, school_id: int, department_id: int, sample_size: int) -> None:
    try:
        dept = q_one(
            "SELECT vocational_field, spektrum_code FROM departments WHERE id=? AND school_id=?",
            (department_id, school_id),
        )
        vocational_field = dept["vocational_field"] if dept else None
        spektrum_code = dept["spektrum_code"] if dept and "spektrum_code" in dept.keys() else None
        payload = run_department_pipeline(
            school_id=school_id,
            department_id=department_id,
            sample_size=sample_size,
            vocational_field=vocational_field,
            spektrum_code=spektrum_code,
        )
        exec_sql(
            "UPDATE runs SET status=?, message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            ("completed", json.dumps(payload), run_id),
        )
    except Exception as exc:
        exec_sql(
            "UPDATE runs SET status=?, message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            ("failed", str(exc), run_id),
        )


def _run_phase2_worker(run_id: int, school_id: int, department_id: int) -> None:
    try:
        dept = q_one(
            "SELECT vocational_field, spektrum_code FROM departments WHERE id=? AND school_id=?",
            (department_id, school_id),
        )
        vocational_field = dept["vocational_field"] if dept else None
        spektrum_code = dept["spektrum_code"] if dept and "spektrum_code" in dept.keys() else None
        payload = run_department_phase2(
            school_id=school_id,
            department_id=department_id,
            vocational_field=vocational_field,
            spektrum_code=spektrum_code,
        )
        exec_sql(
            "UPDATE runs SET status=?, message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            ("completed", json.dumps(payload), run_id),
        )
    except Exception as exc:
        exec_sql(
            "UPDATE runs SET status=?, message=?, completed_at=CURRENT_TIMESTAMP WHERE id=?",
            ("failed", str(exc), run_id),
        )


@app.get("/dashboard/school/runs/status")
def school_runs_status(request: Request, department_id: int):
    """JSON endpoint for polling run status (used by progress indicator)."""
    user = _require_role(request, "school")
    dept = q_one("SELECT * FROM departments WHERE id=? AND school_id=?", (department_id, user["school_id"]))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    rows = q_all("SELECT id, status, message, completed_at FROM runs WHERE department_id=? ORDER BY id DESC", (department_id,))
    runs = [
        {"id": r["id"], "status": r["status"], "message": r["message"] or "", "completed_at": r["completed_at"]}
        for r in rows
    ]
    return {"runs": runs}


@app.post("/dashboard/school/runs/start")
def school_start_run(request: Request, department_id: int = Form(...), sample_size: int = Form(1000)):
    user = _require_role(request, "school")
    dept = q_one("SELECT * FROM departments WHERE id=? AND school_id=?", (department_id, user["school_id"]))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    run_id = exec_sql(
        "INSERT INTO runs(department_id, status, config_snapshot) VALUES(?, 'running', ?)",
        (department_id, json.dumps({"phase": "phase1", "sample_size": sample_size})),
    )
    t = threading.Thread(
        target=_run_worker,
        args=(run_id, user["school_id"], department_id, sample_size),
        daemon=True,
    )
    t.start()
    return RedirectResponse(f"/dashboard/school/runs?department_id={department_id}", status_code=302)


@app.post("/dashboard/school/runs/start_phase2")
def school_start_phase2(request: Request, department_id: int = Form(...)):
    user = _require_role(request, "school")
    dept = q_one("SELECT * FROM departments WHERE id=? AND school_id=?", (department_id, user["school_id"]))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    run_id = exec_sql(
        "INSERT INTO runs(department_id, status, config_snapshot) VALUES(?, 'running', ?)",
        (department_id, json.dumps({"phase": "phase2"})),
    )
    t = threading.Thread(
        target=_run_phase2_worker,
        args=(run_id, user["school_id"], department_id),
        daemon=True,
    )
    t.start()
    return RedirectResponse(f"/dashboard/school/runs?department_id={department_id}", status_code=302)


@app.get("/dashboard/school/upload", response_class=HTMLResponse)
def school_upload_page(
    request: Request,
    department_id: Optional[int] = None,
    uploaded: Optional[str] = None,
    rows: Optional[int] = None,
):
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    success_banner = ""
    if uploaded == "jobs":
        success_banner = f"Jobs CSV uploaded successfully. {rows or 0} rows loaded." if rows is not None else "Jobs CSV uploaded successfully."
    elif uploaded == "curriculum":
        success_banner = "Curriculum file uploaded successfully."
    return templates.TemplateResponse(
        "school/upload.html",
        {
            "request": request,
            "departments": departments,
            "selected_department_id": selected,
            "success_banner": success_banner,
            "user": user,
        },
    )


@app.post("/dashboard/school/upload/jobs")
async def school_upload_jobs(request: Request, department_id: int = Form(...), file: UploadFile = File(...)):
    user = _require_role(request, "school")
    dept = q_one("SELECT * FROM departments WHERE id=? AND school_id=?", (department_id, user["school_id"]))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    dirs = _dept_dirs(user["school_id"], department_id)
    out = dirs["uploads"] / f"jobs_{file.filename}"
    out.write_bytes(await file.read())
    row_count = 0
    try:
        row_count = len(pd.read_csv(out))
    except Exception:
        row_count = 0
    exec_sql(
        "INSERT INTO job_uploads(department_id, file_path, row_count) VALUES(?, ?, ?)",
        (department_id, str(out), row_count),
    )
    return RedirectResponse(
        f"/dashboard/school/upload?department_id={department_id}&uploaded=jobs&rows={row_count}",
        status_code=302,
    )


@app.post("/dashboard/school/upload/curriculum")
async def school_upload_curriculum(request: Request, department_id: int = Form(...), file: UploadFile = File(...)):
    user = _require_role(request, "school")
    dept = q_one("SELECT * FROM departments WHERE id=? AND school_id=?", (department_id, user["school_id"]))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    dirs = _dept_dirs(user["school_id"], department_id)
    out = dirs["uploads"] / f"curriculum_{file.filename}"
    out.write_bytes(await file.read())
    exec_sql(
        "INSERT INTO curriculum_uploads(department_id, file_path, format) VALUES(?, ?, ?)",
        (department_id, str(out), out.suffix.lower().lstrip(".")),
    )
    return RedirectResponse(
        f"/dashboard/school/upload?department_id={department_id}&uploaded=curriculum",
        status_code=302,
    )


@app.get("/dashboard/school/results", response_class=HTMLResponse)
def school_results(
    request: Request,
    department_id: Optional[int] = None,
    aggregate_field: int = 0,
    ranking_mode: str = "human_adjusted",
    skills_n: int = 20,
    skills_custom: Optional[str] = None,
    knowledge_n: int = 30,
    knowledge_custom: Optional[str] = None,
    competencies_n: int = 20,
    competencies_custom: Optional[str] = None,
):
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    if selected is None:
        raise HTTPException(status_code=400, detail="No department available")
    result_dir = _resolve_results_dir(user["school_id"], selected)
    feedback_dir = _resolve_feedback_dir(user["school_id"], selected, result_dir)
    ranking_mode = str(ranking_mode or "human_adjusted").strip().lower()
    if ranking_mode not in {"human_adjusted", "model_only"}:
        ranking_mode = "human_adjusted"
    peer_rows = []

    # Optional peer aggregation for same vocational field across schools.
    if int(aggregate_field) == 1:
        dept = q_one("SELECT * FROM departments WHERE id=?", (selected,))
        if dept:
            peer_rows = q_all(
                """
                SELECT d.id, d.school_id, d.name AS department_name, d.vocational_field, s.name AS school_name
                FROM departments d
                JOIN schools s ON s.id = d.school_id
                WHERE lower(trim(d.vocational_field)) = lower(trim(?))
                ORDER BY s.name, d.name
                """,
                (dept["vocational_field"],),
            )
            result_dirs = []
            feedback_dirs = []
            for pr in peer_rows:
                rd = _resolve_results_dir(pr["school_id"], pr["id"])
                fd = _resolve_feedback_dir(pr["school_id"], pr["id"], rd)
                result_dirs.append(rd)
                feedback_dirs.append(fd)
        else:
            result_dirs = [result_dir]
            feedback_dirs = [feedback_dir]
            peer_rows = []
    else:
        result_dirs = [result_dir]
        feedback_dirs = [feedback_dir]
        peer_rows = q_all(
            """
            SELECT d.id, d.school_id, d.name AS department_name, d.vocational_field, s.name AS school_name
            FROM departments d
            JOIN schools s ON s.id = d.school_id
            WHERE d.id = ?
            """,
            (selected,),
        )

    def _concat_rows(file_name: str, limit_each: int = 1000) -> List[dict]:
        rows: List[dict] = []
        for rd in result_dirs:
            rows.extend(_load_table(rd / file_name, limit=limit_each))
        return rows

    # Merge review statuses across dirs (any verified beats invalid; invalid beats not_verified).
    merged_skill_status: Dict[str, str] = {}
    merged_knowledge_status: Dict[str, str] = {}
    merged_comp_status: Dict[str, str] = {}
    merged_assessments: Dict[str, dict] = {}
    merged_future_weights: Dict[str, float] = {}
    coverage_totals = {
        "skills": {"reviewed": 0, "total": 0},
        "knowledge": {"reviewed": 0, "total": 0},
        "competencies": {"reviewed": 0, "total": 0},
    }
    contributor_meta = []
    seen_feedback_dirs = set()
    irr_reports = []
    for idx, (rd, fd) in enumerate(zip(result_dirs, feedback_dirs)):
        pr = peer_rows[idx] if idx < len(peer_rows) else None
        maps = _review_status_maps(rd, fd)
        for k, v in maps["skills"].items():
            prev = merged_skill_status.get(k, "not_verified")
            if prev != "verified":
                merged_skill_status[k] = "verified" if v == "verified" else ("invalid" if prev == "not_verified" else prev)
        for k, v in maps["knowledge"].items():
            prev = merged_knowledge_status.get(k, "not_verified")
            if prev != "verified":
                merged_knowledge_status[k] = "verified" if v == "verified" else ("invalid" if prev == "not_verified" else prev)
            k_norm = _normalize_knowledge_key(k)
            if k_norm:
                prev_norm = merged_knowledge_status.get(k_norm, "not_verified")
                if prev_norm != "verified":
                    merged_knowledge_status[k_norm] = (
                        "verified" if v == "verified" else ("invalid" if prev_norm == "not_verified" else prev_norm)
                    )
        for k, v in maps["competencies"].items():
            prev = merged_comp_status.get(k, "not_verified")
            if prev != "verified":
                merged_comp_status[k] = "verified" if v == "verified" else prev
        merged_assessments.update(_load_competency_assessments(fd))
        fw = _knowledge_future_weight_map(rd)
        for k, v in fw.items():
            merged_future_weights[k] = max(merged_future_weights.get(k, 0.0), v)
        cov = _review_coverage(rd, fd)
        for k in ["skills", "knowledge", "competencies"]:
            coverage_totals[k]["reviewed"] += int(cov[k]["reviewed"])
            coverage_totals[k]["total"] += int(cov[k]["total"])

        if str(fd) not in seen_feedback_dirs:
            irr = _load_inter_rater_report(fd)
            if irr:
                irr_reports.append({"scope": str(fd), "report": irr})
            seen_feedback_dirs.add(str(fd))

        run_stats = q_one(
            "SELECT COUNT(*) AS run_count, MIN(created_at) AS first_run, MAX(completed_at) AS last_run FROM runs WHERE department_id=?",
            (pr["id"],),
        ) if pr else None
        upload_stats = q_one(
            "SELECT COUNT(*) AS upload_count, COALESCE(SUM(row_count), 0) AS upload_rows, MIN(created_at) AS first_upload, MAX(created_at) AS last_upload FROM job_uploads WHERE department_id=?",
            (pr["id"],),
        ) if pr else None
        contributor_meta.append(
            {
                "school_name": pr["school_name"] if pr else "-",
                "department_name": pr["department_name"] if pr else "-",
                "vocational_field": pr["vocational_field"] if pr else "-",
                "uses_default_results": rd == DEFAULT_RESULTS,
                "run_count": int(run_stats["run_count"]) if run_stats else 0,
                "first_run": (run_stats["first_run"] if run_stats else None) or "-",
                "last_run": (run_stats["last_run"] if run_stats else None) or "-",
                "upload_count": int(upload_stats["upload_count"]) if upload_stats else 0,
                "upload_rows": int(upload_stats["upload_rows"]) if upload_stats else 0,
            }
        )

    def _parse_custom(s: Optional[str]) -> Optional[int]:
        if s is None or not str(s).strip():
            return None
        try:
            return int(s)
        except ValueError:
            return None

    skills_custom_int = _parse_custom(skills_custom)
    knowledge_custom_int = _parse_custom(knowledge_custom)
    competencies_custom_int = _parse_custom(competencies_custom)

    skills_limit = skills_custom_int if (skills_custom_int is not None and skills_custom_int > 0) else skills_n
    knowledge_limit = knowledge_custom_int if (knowledge_custom_int is not None and knowledge_custom_int > 0) else knowledge_n
    competencies_limit = (
        competencies_custom_int if (competencies_custom_int is not None and competencies_custom_int > 0) else competencies_n
    )

    # Safety clamp for UI-provided limits.
    skills_limit = max(1, min(int(skills_limit), 500))
    knowledge_limit = max(1, min(int(knowledge_limit), 1000))
    competencies_limit = max(1, min(int(competencies_limit), 500))

    recs_raw = _concat_rows("recommendations.csv", limit_each=1500)
    recs = _prioritize_recommendations(
        recs_raw,
        skill_status=merged_skill_status,
        ranking_mode=ranking_mode,
        top_n=None,
    )
    hard_recs = [r for r in recs if str(r.get("skill_type", "")).strip().lower() in {"hard", "both"}]
    soft_recs = [r for r in recs if str(r.get("skill_type", "")).strip().lower() in {"soft", "both"}]
    hard_recs = hard_recs[:skills_limit]
    soft_recs = soft_recs[:skills_limit]

    knowledge_raw = _concat_rows("advanced_knowledge.csv", limit_each=5000)
    knowledge_raw = _aggregate_knowledge_rows(knowledge_raw)
    top_knowledge = _prioritize_knowledge_rows(
        knowledge_raw,
        knowledge_status=merged_knowledge_status,
        future_weight_map=merged_future_weights,
        ranking_mode=ranking_mode,
        top_n=knowledge_limit,
    )

    comps_raw = _concat_rows("competency_proposals.json", limit_each=1500)
    comps = _prioritize_competencies(
        comps_raw,
        recs,
        competency_status=merged_comp_status,
        competency_assessments=merged_assessments,
        ranking_mode=ranking_mode,
        top_n=competencies_limit,
    )
    comps = _enrich_competencies_with_knowledge(comps, top_knowledge)
    phase2_completed = _phase2_completed(user["school_id"], selected)
    banner = ""
    if result_dir == DEFAULT_RESULTS:
        banner = "Showing default demo results. Upload and run your department data for personalized results."
    if int(aggregate_field) == 1:
        banner = (banner + " " if banner else "") + "Cross-school aggregation mode is ON for the same vocational field."
    if phase2_completed:
        banner = (banner + " " if banner else "") + "Final results (Phase 2)."
    else:
        banner = (banner + " " if banner else "") + "Preview (Phase 1). Run Phase 2 after review to see final results."
    if ranking_mode == "model_only":
        banner = (banner + " " if banner else "") + "Ranking mode: MODEL ONLY (human review shown but not used for ordering)."
    else:
        banner = (banner + " " if banner else "") + "Ranking mode: HUMAN ADJUSTED (human-in-the-loop evidence influences ordering)."

    irr_overview = []
    for bundle in irr_reports:
        for src, vals in bundle["report"].items():
            irr_overview.append(
                {
                    "source": src,
                    "shared_items": vals.get("shared_items", 0),
                    "cohens_kappa": vals.get("cohens_kappa", 0),
                    "pct_agreement": vals.get("pct_agreement", 0),
                }
            )
    return templates.TemplateResponse(
        "school/results.html",
        {
            "request": request,
            "departments": departments,
            "selected_department_id": selected,
            "aggregate_field": int(aggregate_field),
            "ranking_mode": ranking_mode,
            "skills_n": skills_n,
            "skills_custom": skills_custom or "",
            "knowledge_n": knowledge_n,
            "knowledge_custom": knowledge_custom or "",
            "competencies_n": competencies_n,
            "competencies_custom": competencies_custom or "",
            "recommendations": recs,
            "hard_skills": hard_recs,
            "soft_skills": soft_recs,
            "knowledges": top_knowledge,
            "competencies": comps,
            "coverage_totals": coverage_totals,
            "contributors": contributor_meta,
            "irr_overview": irr_overview,
            "banner": banner,
            "phase2_completed": phase2_completed,
            "phase2_available": _phase2_available(user["school_id"], selected),
            "explanations": _explanations(),
            "user": user,
        },
    )


@app.get("/dashboard/school/insights", response_class=HTMLResponse)
def school_insights(request: Request, department_id: Optional[int] = None):
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    if selected is None:
        raise HTTPException(status_code=400, detail="No department available")
    result_dir = _resolve_results_dir(user["school_id"], selected)
    fig_dir = result_dir / "figures"
    image_paths = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    image_cards = []
    for p in image_paths:
        info = _plot_info_for_filename(p.name)
        image_cards.append(
            {
                "name": p.name,
                "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "title": info["title"],
                "what": info["what"],
                "how": info["how"],
            }
        )
    banner = ""
    if result_dir == DEFAULT_RESULTS:
        banner = "Showing default demo insights. Run your department pipeline to replace these with your own."
    return templates.TemplateResponse(
        "school/insights.html",
        {
            "request": request,
            "departments": departments,
            "selected_department_id": selected,
            "images": image_cards,
            "banner": banner,
            "user": user,
        },
    )


@app.get("/dashboard/school/methodology", response_class=HTMLResponse)
def school_methodology(request: Request):
    """How it works / transparency page for explainable AI."""
    user = _require_role(request, "school")
    explanations = _explanations()
    return templates.TemplateResponse(
        "school/methodology.html",
        {
            "request": request,
            "explanations": explanations,
            "user": user,
        },
    )


@app.get("/dashboard/school/review", response_class=HTMLResponse)
def school_review(request: Request, department_id: Optional[int] = None):
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    if selected is None:
        raise HTTPException(status_code=400, detail="No department available")
    result_dir = _resolve_results_dir(user["school_id"], selected)
    feedback_dir = _resolve_feedback_dir(user["school_id"], selected, result_dir)
    reviewer_id = user.get("email", "default")
    load_more_available = (result_dir / "expert_review_skills.csv").exists()

    skills_tpl = _load_table(result_dir / "expert_review_skills.csv", limit=500)
    knowledge_tpl = _load_table(result_dir / "expert_review_knowledge.csv", limit=500)
    comp_tpl = _load_table(result_dir / "expert_review_competencies.csv", limit=200)

    skills_df = pd.DataFrame(skills_tpl)
    knowledge_df = pd.DataFrame(knowledge_tpl)
    comp_df = pd.DataFrame(comp_tpl)

    skills_fb = _load_feedback_csv(feedback_dir, "skill_feedback.csv")
    knowledge_fb = _load_feedback_csv(feedback_dir, "knowledge_feedback.csv")
    comp_fb = _load_feedback_csv(feedback_dir, "competency_feedback.csv")

    skills_df = _merge_template_with_reviewer_feedback(
        skills_df, skills_fb, "review_id", reviewer_id, ["human_valid", "human_type", "human_bloom", "human_notes"]
    )
    knowledge_df = _merge_template_with_reviewer_feedback(
        knowledge_df, knowledge_fb, "review_id", reviewer_id, ["human_valid", "human_notes"]
    )
    comp_df = _merge_template_with_reviewer_feedback(
        comp_df, comp_fb, "competency_id", reviewer_id, ["human_quality", "human_relevant", "human_skill_focus", "human_notes"]
    )

    return templates.TemplateResponse(
        "school/review.html",
        {
            "request": request,
            "departments": departments,
            "selected_department_id": selected,
            "reviewer_id": reviewer_id,
            "skills": skills_df.fillna("").to_dict(orient="records"),
            "knowledge": knowledge_df.fillna("").to_dict(orient="records"),
            "competencies": comp_df.fillna("").to_dict(orient="records"),
            "load_more_available": load_more_available,
            "explanations": _explanations(),
            "user": user,
        },
    )


@app.post("/dashboard/school/review/save_skill")
def save_skill_review(
    request: Request,
    department_id: int = Form(...),
    review_id: str = Form(...),
    human_valid: str = Form(""),
    human_type: str = Form(""),
    human_bloom: str = Form(""),
    human_notes: str = Form(""),
):
    user = _require_role(request, "school")
    result_dir = _resolve_results_dir(user["school_id"], department_id)
    feedback_dir = _resolve_feedback_dir(user["school_id"], department_id, result_dir)
    _upsert_feedback_csv(
        feedback_dir,
        "skill_feedback.csv",
        "review_id",
        review_id,
        user.get("email", "default"),
        {
            "human_valid": human_valid,
            "human_type": human_type,
            "human_bloom": human_bloom,
            "human_notes": human_notes,
        },
    )
    return RedirectResponse(f"/dashboard/school/review?department_id={department_id}", status_code=302)


@app.post("/dashboard/school/review/save_knowledge")
def save_knowledge_review(
    request: Request,
    department_id: int = Form(...),
    review_id: str = Form(...),
    human_valid: str = Form(""),
    human_notes: str = Form(""),
):
    user = _require_role(request, "school")
    result_dir = _resolve_results_dir(user["school_id"], department_id)
    feedback_dir = _resolve_feedback_dir(user["school_id"], department_id, result_dir)
    _upsert_feedback_csv(
        feedback_dir,
        "knowledge_feedback.csv",
        "review_id",
        review_id,
        user.get("email", "default"),
        {"human_valid": human_valid, "human_notes": human_notes},
    )
    return RedirectResponse(f"/dashboard/school/review?department_id={department_id}", status_code=302)


@app.post("/dashboard/school/review/save_competency")
def save_competency_review(
    request: Request,
    department_id: int = Form(...),
    competency_id: str = Form(...),
    human_quality: str = Form(""),
    human_relevant: str = Form(""),
    human_skill_focus: str = Form(""),
    human_notes: str = Form(""),
):
    user = _require_role(request, "school")
    result_dir = _resolve_results_dir(user["school_id"], department_id)
    feedback_dir = _resolve_feedback_dir(user["school_id"], department_id, result_dir)
    _upsert_feedback_csv(
        feedback_dir,
        "competency_feedback.csv",
        "competency_id",
        competency_id,
        user.get("email", "default"),
        {
            "human_quality": human_quality,
            "human_relevant": human_relevant,
            "human_skill_focus": human_skill_focus,
            "human_notes": human_notes,
        },
    )
    return RedirectResponse(f"/dashboard/school/review?department_id={department_id}", status_code=302)


@app.post("/dashboard/school/review/api/save_skill")
async def save_skill_review_api(request: Request):
    user = _require_role(request, "school")
    payload = await request.json()
    department_id = int(payload.get("department_id"))
    review_id = str(payload.get("review_id", "")).strip()
    if not review_id:
        raise HTTPException(status_code=400, detail="review_id is required")
    result_dir = _resolve_results_dir(user["school_id"], department_id)
    feedback_dir = _resolve_feedback_dir(user["school_id"], department_id, result_dir)
    _upsert_feedback_csv(
        feedback_dir,
        "skill_feedback.csv",
        "review_id",
        review_id,
        user.get("email", "default"),
        {
            "human_valid": str(payload.get("human_valid", "")),
            "human_type": str(payload.get("human_type", "")),
            "human_bloom": str(payload.get("human_bloom", "")),
            "human_notes": str(payload.get("human_notes", "")),
        },
    )
    return {"ok": True}


@app.post("/dashboard/school/review/api/save_knowledge")
async def save_knowledge_review_api(request: Request):
    user = _require_role(request, "school")
    payload = await request.json()
    department_id = int(payload.get("department_id"))
    review_id = str(payload.get("review_id", "")).strip()
    if not review_id:
        raise HTTPException(status_code=400, detail="review_id is required")
    result_dir = _resolve_results_dir(user["school_id"], department_id)
    feedback_dir = _resolve_feedback_dir(user["school_id"], department_id, result_dir)
    _upsert_feedback_csv(
        feedback_dir,
        "knowledge_feedback.csv",
        "review_id",
        review_id,
        user.get("email", "default"),
        {
            "human_valid": str(payload.get("human_valid", "")),
            "human_notes": str(payload.get("human_notes", "")),
        },
    )
    return {"ok": True}


@app.post("/dashboard/school/review/api/save_competency")
async def save_competency_review_api(request: Request):
    user = _require_role(request, "school")
    payload = await request.json()
    department_id = int(payload.get("department_id"))
    competency_id = str(payload.get("competency_id", "")).strip()
    if not competency_id:
        raise HTTPException(status_code=400, detail="competency_id is required")
    result_dir = _resolve_results_dir(user["school_id"], department_id)
    feedback_dir = _resolve_feedback_dir(user["school_id"], department_id, result_dir)
    _upsert_feedback_csv(
        feedback_dir,
        "competency_feedback.csv",
        "competency_id",
        competency_id,
        user.get("email", "default"),
        {
            "human_quality": str(payload.get("human_quality", "")),
            "human_relevant": str(payload.get("human_relevant", "")),
            "human_skill_focus": str(payload.get("human_skill_focus", "")),
            "human_notes": str(payload.get("human_notes", "")),
        },
    )
    return {"ok": True}


@app.post("/dashboard/school/review/load_more")
def school_review_load_more(request: Request, department_id: int = Form(...)):
    user = _require_role(request, "school")
    dept = q_one("SELECT * FROM departments WHERE id=? AND school_id=?", (department_id, user["school_id"]))
    if not dept:
        raise HTTPException(status_code=404, detail="Department not found")
    try:
        payload = run_load_more_samples(user["school_id"], department_id)
        if payload.get("status") == "error":
            raise HTTPException(status_code=400, detail=payload.get("reason", "Load more failed"))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return RedirectResponse(f"/dashboard/school/review?department_id={department_id}", status_code=302)


# ---------------------------------------------------------------------------
# Item 9: Trend Sparklines API  — GET /dashboard/api/sparklines
# ---------------------------------------------------------------------------

@app.get("/dashboard/api/sparklines")
def api_sparklines(request: Request, department_id: Optional[int] = None, top_n: int = 15):
    """Return monthly frequency data for top skills (for sparkline rendering)."""
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    if selected is None:
        return {"skills": []}

    result_dir = _resolve_results_dir(user["school_id"], selected)

    # Load skills with dates
    skills_data: List[dict] = []
    for fname in ["advanced_skills_with_dates.csv", "advanced_skills_normalized.csv"]:
        p = result_dir / fname
        if p.exists():
            try:
                df = pd.read_csv(p)
                if "skill" in df.columns and "job_date" in df.columns:
                    df["job_date"] = pd.to_datetime(df["job_date"], errors="coerce")
                    df = df.dropna(subset=["job_date"])
                    df["_month"] = df["job_date"].dt.to_period("M").astype(str)
                    skill_col = "canonical_skill" if "canonical_skill" in df.columns else "skill"
                    # Top skills by count
                    top_skills = df[skill_col].value_counts().head(top_n).index.tolist()
                    for sk in top_skills:
                        monthly = (
                            df[df[skill_col] == sk]
                            .groupby("_month").size()
                            .reset_index(name="freq")
                            .sort_values("_month")
                        )
                        skills_data.append({
                            "skill": sk,
                            "months": monthly["_month"].tolist(),
                            "counts": monthly["freq"].tolist(),
                        })
                    break
            except Exception:
                pass

    # Fallback: use trends CSV (slope data only, no sparkline)
    if not skills_data:
        trends_path = result_dir / "skill_time_trends.csv"
        if trends_path.exists():
            try:
                tdf = pd.read_csv(trends_path).head(top_n)
                for _, row in tdf.iterrows():
                    slope = float(row.get("slope", 0))
                    skills_data.append({
                        "skill": str(row.get("skill", "")),
                        "months": [],
                        "counts": [],
                        "slope": round(slope, 4),
                        "trend_label": str(row.get("trend_label", "Stable")),
                    })
            except Exception:
                pass

    return {"skills": skills_data}


# ---------------------------------------------------------------------------
# Item 2: Printable PDF Report  — GET /dashboard/school/report
# ---------------------------------------------------------------------------

@app.get("/dashboard/school/report", response_class=HTMLResponse)
def school_report(
    request: Request,
    department_id: Optional[int] = None,
    top_n: int = 20,
):
    """Render a printer-optimised report page for download / Save as PDF."""
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    if selected is None:
        raise HTTPException(status_code=400, detail="No department available")

    dept = q_one(
        "SELECT d.*, s.name AS school_name FROM departments d JOIN schools s ON s.id=d.school_id WHERE d.id=?",
        (selected,),
    )
    result_dir = _resolve_results_dir(user["school_id"], selected)
    feedback_dir = _resolve_feedback_dir(user["school_id"], selected, result_dir)

    status_maps = _review_status_maps(result_dir, feedback_dir)
    fw_map = _knowledge_future_weight_map(result_dir)
    assessments: Dict[str, dict] = {}

    # Load recommendations
    recs_raw = _load_table(result_dir / "recommendations.csv", limit=top_n * 2)
    recs = _prioritize_recommendations(
        recs_raw, status_maps["skills"], top_n=top_n, ranking_mode="human_adjusted"
    )

    # Load knowledge
    knowledge_raw = _load_table(result_dir / "advanced_knowledge.csv", limit=500)
    knowledge = _prioritize_knowledge_rows(
        _aggregate_knowledge_rows(knowledge_raw, fw_map), top_n=top_n
    )

    # Load competencies
    comp_raw = _load_table(result_dir / "competency_proposals.json", limit=top_n * 2)
    competencies = _prioritize_competencies(
        comp_raw, recs, assessments, top_n=top_n
    )

    # Load run metadata for provenance
    run_meta: dict = {}
    meta_path = result_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            run_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    return templates.TemplateResponse(
        "school/report.html",
        {
            "request": request,
            "user": user,
            "dept": dict(dept) if dept else {},
            "departments": departments,
            "selected_dept_id": selected,
            "recs": recs,
            "knowledge": knowledge,
            "competencies": competencies,
            "run_meta": run_meta,
            "top_n": top_n,
            "report_date": __import__("datetime").date.today().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Item 3: Explainability API  — GET /dashboard/api/explain_score
# Returns the score breakdown for a single recommendation row.
# ---------------------------------------------------------------------------

@app.get("/dashboard/api/explain_score")
def api_explain_score(
    request: Request,
    skill: str,
    department_id: Optional[int] = None,
):
    """Return demand/trend/future score breakdown for a skill."""
    user = _require_role(request, "school")
    departments = _school_departments(user)
    selected = department_id or (departments[0]["id"] if departments else None)
    if selected is None:
        return {"error": "no department"}

    result_dir = _resolve_results_dir(user["school_id"], selected)

    # Load recommendations and find the skill
    recs_path = result_dir / "recommendations.csv"
    if not recs_path.exists():
        return {"error": "recommendations.csv not found"}

    try:
        df = pd.read_csv(recs_path)
        match = df[df["skill"].str.lower() == skill.strip().lower()]
        if match.empty:
            return {"error": "skill not found", "skill": skill}
        row = match.iloc[0].to_dict()

        # Build explainability payload
        breakdown = {
            "skill": str(row.get("skill", skill)),
            "priority_score": round(_to_float(row.get("priority_score", 0)), 4),
            "score_components": {
                "demand": {
                    "value": round(_to_float(row.get("demand_norm", row.get("demand_freq", 0))), 4),
                    "weight": 0.40,
                    "contribution": round(_to_float(row.get("demand_norm", 0)) * 0.40, 4),
                    "raw_freq": int(_to_float(row.get("demand_freq", 0))),
                    "label": "How often this skill appears in job postings",
                },
                "trend": {
                    "value": round(_to_float(row.get("trend_score_norm", row.get("slope", 0))), 4),
                    "weight": 0.30,
                    "contribution": round(_to_float(row.get("trend_score_norm", 0)) * 0.30, 4),
                    "trend_label": str(row.get("trend_label", "Stable")),
                    "q_value": round(_to_float(row.get("trend_q_value", 1.0)), 4),
                    "label": "Whether demand is growing or declining over time",
                },
                "future": {
                    "value": round(_to_float(row.get("future_weight", 0)), 4),
                    "weight": 0.30,
                    "contribution": round(_to_float(row.get("future_weight", 0)) * 0.30, 4),
                    "domain": str(row.get("future_domain", row.get("best_future_domain", "—"))),
                    "mapping_margin": round(_to_float(row.get("mapping_margin", 0)), 4),
                    "label": "Alignment with WEF/O*NET future-of-work domains",
                },
            },
            "verification": str(row.get("verification_level", "not_verified")),
            "bloom_level": str(row.get("bloom", "—")),
            "skill_type": str(row.get("type", "—")),
        }
        return breakdown
    except Exception as exc:
        return {"error": str(exc)}


@app.get("/dashboard/file/{path:path}")
def serve_project_file(path: str):
    safe = (PROJECT_ROOT / path).resolve()
    if not str(safe).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden path")
    if not safe.exists():
        raise HTTPException(status_code=404, detail="Not found")
    from fastapi.responses import FileResponse

    return FileResponse(str(safe))

