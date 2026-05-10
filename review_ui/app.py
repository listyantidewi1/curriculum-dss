"""
Review UI - FastAPI web app for human-in-the-loop feedback.

Multi-reviewer support: each reviewer reviews the same set. Feedback is stored
per (item_id, reviewer_id) in feedback_store/ so reviewers don't overwrite each other.

Run: uvicorn review_ui.app:app --reload
Usage: Open with ?reviewer_id=alice (or r1, r2, etc.) so each reviewer has their own session.
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Resolve paths relative to project root (ensure absolute paths)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    import config
    OUTPUT_DIR = Path(config.OUTPUT_DIR).resolve()
    FEEDBACK_DIR = (Path(config.PROJECT_ROOT) / "feedback_store").resolve()
except ImportError:
    OUTPUT_DIR = (PROJECT_ROOT / "results").resolve()
    FEEDBACK_DIR = (PROJECT_ROOT / "feedback_store").resolve()

FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

# Log paths on startup for debugging (visible in uvicorn console)
_skills_path = OUTPUT_DIR / "expert_review_skills.csv"
import sys
print(f"[Review UI] OUTPUT_DIR={OUTPUT_DIR}", file=sys.stderr)
print(f"[Review UI] expert_review_skills.csv exists={_skills_path.exists()}, path={_skills_path}", file=sys.stderr)

app = FastAPI(title="Skill Extraction Review UI")


# --- Pydantic models ---
class SkillFeedback(BaseModel):
    review_id: str
    human_valid: str = ""
    human_type: str = ""
    human_notes: str = ""
    reviewer_id: str = ""


class KnowledgeFeedback(BaseModel):
    review_id: str
    human_valid: str = ""
    human_notes: str = ""
    reviewer_id: str = ""


class CompetencyFeedback(BaseModel):
    competency_id: str
    human_quality: str = ""
    human_relevant: str = ""
    human_skill_focus: str = ""
    human_notes: str = ""
    reviewer_id: str = ""


# --- Data loading ---
def _load_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_feedback_csv(name: str) -> pd.DataFrame:
    path = FEEDBACK_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_feedback_csv(df: pd.DataFrame, name: str) -> None:
    path = FEEDBACK_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _upsert_feedback(feedback_path: Path, id_col: str, row_id: str, reviewer_id: str, updates: dict) -> None:
    """Upsert a row in feedback CSV by (id_col, reviewer_id)."""
    all_cols = [id_col, "reviewer_id"] + list(updates.keys())
    if feedback_path.exists():
        df = pd.read_csv(feedback_path)
    else:
        df = pd.DataFrame(columns=all_cols)
    for c in all_cols:
        if c not in df.columns:
            df[c] = ""
    mask = (df[id_col].astype(str) == str(row_id)) & (df["reviewer_id"].astype(str) == str(reviewer_id))
    if mask.any():
        for k, v in updates.items():
            if k in df.columns:
                df.loc[mask, k] = v
    else:
        new_row = {id_col: row_id, "reviewer_id": reviewer_id}
        for k, v in updates.items():
            new_row[k] = v
        for c in df.columns:
            if c not in new_row:
                new_row[c] = ""
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(feedback_path, index=False, encoding="utf-8-sig")


def _merge_template_with_feedback(template_df: pd.DataFrame, feedback_df: pd.DataFrame,
                                  id_col: str, reviewer_id: str, feedback_cols: list) -> pd.DataFrame:
    """Merge template items with this reviewer's feedback only. Template feedback cols are cleared
    so each reviewer sees only their own feedback."""
    if template_df.empty:
        return template_df
    df = template_df.copy()
    # Clear feedback columns - feedback comes only from feedback_store per reviewer
    for c in feedback_cols:
        df[c] = ""
    if not feedback_df.empty and reviewer_id:
        rev_fb = feedback_df[feedback_df["reviewer_id"].astype(str) == str(reviewer_id)]
        if not rev_fb.empty and id_col in rev_fb.columns:
            fb_map = rev_fb.set_index(id_col)
            for idx, row in df.iterrows():
                rid = str(row.get(id_col, ""))
                if rid in fb_map.index:
                    for c in feedback_cols:
                        if c in fb_map.columns:
                            val = fb_map.loc[rid, c]
                            if pd.notna(val) and str(val).strip():
                                df.at[idx, c] = val
    return df


# --- API routes ---
@app.get("/api/config")
def get_config():
    skills_path = OUTPUT_DIR / "expert_review_skills.csv"
    return {
        "output_dir": str(OUTPUT_DIR),
        "feedback_dir": str(FEEDBACK_DIR),
        "skills_file_exists": skills_path.exists(),
        "skills_file_path": str(skills_path),
    }


@app.get("/api/skills")
def get_skills(reviewer_id: str = Query("", alias="reviewer_id")):
    template = _load_csv(OUTPUT_DIR, "expert_review_skills.csv")
    if template.empty:
        skills_path = OUTPUT_DIR / "expert_review_skills.csv"
        hint = f"expert_review_skills.csv not found at {skills_path}" if not skills_path.exists() else "File exists but is empty"
        return {"items": [], "total": 0, "hint": hint}
    feedback = _load_feedback_csv("skill_feedback.csv")
    rid = reviewer_id.strip() or "default"
    df = _merge_template_with_feedback(
        template, feedback, "review_id", rid,
        ["human_valid", "human_type", "human_notes"]
    )
    df = df.fillna("")
    items = df.to_dict(orient="records")
    for r in items:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = ""
            elif hasattr(v, "item"):
                r[k] = str(v)
    return {"items": items, "total": len(items), "reviewer_id": rid}


@app.get("/api/knowledge")
def get_knowledge(reviewer_id: str = Query("", alias="reviewer_id")):
    template = _load_csv(OUTPUT_DIR, "expert_review_knowledge.csv")
    if template.empty:
        return {"items": [], "total": 0}
    feedback = _load_feedback_csv("knowledge_feedback.csv")
    rid = reviewer_id.strip() or "default"
    df = _merge_template_with_feedback(
        template, feedback, "review_id", rid,
        ["human_valid", "human_notes"]
    )
    df = df.fillna("")
    items = df.to_dict(orient="records")
    for r in items:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = ""
            elif hasattr(v, "item"):
                r[k] = str(v)
    return {"items": items, "total": len(items), "reviewer_id": rid}


@app.get("/api/competencies")
def get_competencies(reviewer_id: str = Query("", alias="reviewer_id")):
    template = _load_csv(OUTPUT_DIR, "expert_review_competencies.csv")
    if template.empty:
        return {"items": [], "total": 0}
    feedback = _load_feedback_csv("competency_feedback.csv")
    rid = reviewer_id.strip() or "default"
    df = _merge_template_with_feedback(
        template, feedback, "competency_id", rid,
        ["human_quality", "human_relevant", "human_skill_focus", "human_notes"]
    )
    df = df.fillna("")
    items = df.to_dict(orient="records")
    for r in items:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = ""
            elif hasattr(v, "item"):
                r[k] = str(v)
    return {"items": items, "total": len(items), "reviewer_id": rid}


@app.post("/api/save_skill_feedback")
def save_skill_feedback(fb: SkillFeedback):
    template = _load_csv(OUTPUT_DIR, "expert_review_skills.csv")
    if template.empty or "review_id" not in template.columns:
        raise HTTPException(status_code=400, detail="No skills to update")
    if str(fb.review_id) not in template["review_id"].astype(str).values:
        raise HTTPException(status_code=404, detail="review_id not found")
    rid = (fb.reviewer_id or "default").strip()
    path = FEEDBACK_DIR / "skill_feedback.csv"
    _upsert_feedback(path, "review_id", fb.review_id, rid, {
        "human_valid": fb.human_valid,
        "human_type": fb.human_type,
        "human_notes": fb.human_notes,
    })
    return {"ok": True}


@app.post("/api/save_knowledge_feedback")
def save_knowledge_feedback(fb: KnowledgeFeedback):
    template = _load_csv(OUTPUT_DIR, "expert_review_knowledge.csv")
    if template.empty or "review_id" not in template.columns:
        raise HTTPException(status_code=400, detail="No knowledge to update")
    if str(fb.review_id) not in template["review_id"].astype(str).values:
        raise HTTPException(status_code=404, detail="review_id not found")
    rid = (fb.reviewer_id or "default").strip()
    path = FEEDBACK_DIR / "knowledge_feedback.csv"
    _upsert_feedback(path, "review_id", fb.review_id, rid, {
        "human_valid": fb.human_valid,
        "human_notes": fb.human_notes,
    })
    return {"ok": True}


@app.post("/api/save_competency_feedback")
def save_competency_feedback(fb: CompetencyFeedback):
    template = _load_csv(OUTPUT_DIR, "expert_review_competencies.csv")
    if template.empty or "competency_id" not in template.columns:
        raise HTTPException(status_code=400, detail="No competencies to update")
    if str(fb.competency_id) not in template["competency_id"].astype(str).values:
        raise HTTPException(status_code=404, detail="competency_id not found")
    rid = (fb.reviewer_id or "default").strip()
    path = FEEDBACK_DIR / "competency_feedback.csv"
    _upsert_feedback(path, "competency_id", fb.competency_id, rid, {
        "human_quality": fb.human_quality,
        "human_relevant": fb.human_relevant,
        "human_skill_focus": fb.human_skill_focus,
        "human_notes": fb.human_notes,
    })
    return {"ok": True}


@app.post("/api/save_all_skills")
def save_all_skills(items: list[dict]):
    template = _load_csv(OUTPUT_DIR, "expert_review_skills.csv")
    if template.empty:
        raise HTTPException(status_code=400, detail="No skills loaded")
    rid = "default"
    path = FEEDBACK_DIR / "skill_feedback.csv"
    for item in items:
        review_id = item.get("review_id")
        if not review_id:
            continue
        rid = (item.get("reviewer_id") or "default").strip()
        _upsert_feedback(path, "review_id", review_id, rid, {
            "human_valid": item.get("human_valid", ""),
            "human_type": item.get("human_type", ""),
            "human_notes": item.get("human_notes", ""),
        })
    return {"ok": True, "updated": len(items)}


@app.post("/api/save_all_knowledge")
def save_all_knowledge(items: list[dict]):
    template = _load_csv(OUTPUT_DIR, "expert_review_knowledge.csv")
    if template.empty:
        raise HTTPException(status_code=400, detail="No knowledge loaded")
    path = FEEDBACK_DIR / "knowledge_feedback.csv"
    for item in items:
        review_id = item.get("review_id")
        if not review_id:
            continue
        rid = (item.get("reviewer_id") or "default").strip()
        _upsert_feedback(path, "review_id", review_id, rid, {
            "human_valid": item.get("human_valid", ""),
            "human_notes": item.get("human_notes", ""),
        })
    return {"ok": True, "updated": len(items)}


@app.post("/api/save_all_competencies")
def save_all_competencies(items: list[dict]):
    template = _load_csv(OUTPUT_DIR, "expert_review_competencies.csv")
    if template.empty:
        raise HTTPException(status_code=400, detail="No competencies loaded")
    path = FEEDBACK_DIR / "competency_feedback.csv"
    for item in items:
        cid = item.get("competency_id")
        if not cid:
            continue
        rid = (item.get("reviewer_id") or "default").strip()
        _upsert_feedback(path, "competency_id", cid, rid, {
            "human_quality": item.get("human_quality", ""),
            "human_relevant": item.get("human_relevant", ""),
            "human_skill_focus": item.get("human_skill_focus", ""),
            "human_notes": item.get("human_notes", ""),
        })
    return {"ok": True, "updated": len(items)}


# --- Static files and HTML ---
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
    return html
