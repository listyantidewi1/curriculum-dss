"""
Gold Labeling UI - FastAPI web app for ground-truth labeling.

Multi-reviewer support: each reviewer labels the same gold set. Labels are stored
per (gold_id, labeler_id) in DATA/labels/gold_labels/ so reviewers don't overwrite
each other. Used for extraction evaluation (precision) and inter-rater reliability.

Run: uvicorn gold_labeling_ui.app:app --reload
Usage: Open with ?labeler_id=alice (or dewi, r2, etc.) for multi-reviewer sessions.

Overlap set: First 20 items in each category must be labeled by ALL reviewers
to compute Cohen's Kappa for inter-rater reliability.
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    import config
    LABELS_DIR = (Path(config.PROJECT_ROOT) / "DATA" / "labels").resolve()
except ImportError:
    LABELS_DIR = (PROJECT_ROOT / "DATA" / "labels").resolve()

GOLD_LABELS_DIR = LABELS_DIR / "gold_labels"
GOLD_LABELS_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP_N = 20  # First N items labeled by all reviewers for IRR

app = FastAPI(title="Gold Set Labeling UI")


# --- Pydantic models ---
class SkillLabel(BaseModel):
    gold_id: str
    is_correct: str = ""
    type_label: str = ""
    notes: str = ""
    labeler_id: str = ""


class KnowledgeLabel(BaseModel):
    gold_id: str
    is_correct: str = ""
    notes: str = ""
    labeler_id: str = ""


class DomainLabel(BaseModel):
    gold_id: str
    true_domain_id: str = ""
    notes: str = ""
    labeler_id: str = ""


# --- Data loading ---
def _load_gold_csv(name: str) -> pd.DataFrame:
    path = LABELS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_labels_csv(name: str) -> pd.DataFrame:
    path = GOLD_LABELS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save_labels_csv(df: pd.DataFrame, name: str) -> None:
    path = GOLD_LABELS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _upsert_label(path: Path, id_col: str, row_id: str, labeler_id: str, updates: dict) -> None:
    """Upsert a row in labels CSV by (id_col, labeler_id)."""
    all_cols = [id_col, "labeler_id"] + list(updates.keys())
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=all_cols)
    for c in all_cols:
        if c not in df.columns:
            df[c] = ""
    mask = (df[id_col].astype(str) == str(row_id)) & (df["labeler_id"].astype(str) == str(labeler_id))
    if mask.any():
        for k, v in updates.items():
            if k in df.columns:
                df.loc[mask, k] = v
    else:
        new_row = {id_col: row_id, "labeler_id": labeler_id, **updates}
        for c in df.columns:
            if c not in new_row:
                new_row[c] = ""
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _merge_template_with_labels(template: pd.DataFrame, labels: pd.DataFrame,
                                id_col: str, labeler_id: str, label_cols: list) -> pd.DataFrame:
    """Merge template items with this labeler's labels only."""
    if template.empty:
        return template
    df = template.copy()
    for c in label_cols:
        if c in df.columns:
            df[c] = ""
    if not labels.empty and labeler_id:
        rev = labels[labels["labeler_id"].astype(str) == str(labeler_id)]
        if not rev.empty and id_col in rev.columns:
            fb_map = rev.set_index(id_col)
            for idx, row in df.iterrows():
                gid = str(row.get(id_col, ""))
                if gid in fb_map.index:
                    for c in label_cols:
                        if c in fb_map.columns:
                            val = fb_map.loc[gid, c]
                            if pd.notna(val) and str(val).strip():
                                df.at[idx, c] = val
    return df


# --- API routes ---
@app.get("/api/config")
def get_config():
    return {
        "labels_dir": str(LABELS_DIR),
        "gold_labels_dir": str(GOLD_LABELS_DIR),
        "overlap_n": OVERLAP_N,
        "skills_exists": (LABELS_DIR / "gold_skills.csv").exists(),
        "knowledge_exists": (LABELS_DIR / "gold_knowledge.csv").exists(),
        "domain_exists": (LABELS_DIR / "gold_future_domain.csv").exists(),
        "future_domains_path": str(PROJECT_ROOT / "future_domains.csv"),
    }


@app.get("/api/domains")
def get_future_domains():
    """Return domain_id + future_domain name for domain labeling dropdown."""
    path = PROJECT_ROOT / "future_domains.csv"
    if not path.exists():
        return {"domains": [], "fallback": [
            {"id": "WEF01", "name": "WEF01"},
            {"id": "none", "name": "none"},
            {"id": "unclear", "name": "unclear"},
        ]}
    df = pd.read_csv(path)
    if "domain_id" not in df.columns:
        return {"domains": [{"id": "none", "name": "none"}, {"id": "unclear", "name": "unclear"}]}
    has_name = "future_domain" in df.columns
    items = []
    for _, row in df.iterrows():
        did = str(row["domain_id"])
        dname = str(row["future_domain"]) if has_name else did
        items.append({"id": did, "name": dname})
    items.append({"id": "none", "name": "none — no domain fits"})
    items.append({"id": "unclear", "name": "unclear — cannot decide"})
    return {"domains": items}


@app.get("/api/skills")
def get_skills(labeler_id: str = Query("", alias="labeler_id")):
    template = _load_gold_csv("gold_skills.csv")
    if template.empty:
        return {"items": [], "total": 0, "overlap_n": OVERLAP_N,
                "hint": "Run export_gold_set.py first to create gold_skills.csv"}
    labels = _load_labels_csv("skill_labels.csv")
    rid = labeler_id.strip() or "default"
    df = _merge_template_with_labels(
        template, labels, "gold_id", rid,
        ["is_correct", "type_label", "notes"]
    )
    df = df.fillna("")
    items = df.to_dict(orient="records")
    for r in items:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = ""
            elif hasattr(v, "item"):
                r[k] = str(v)
    return {"items": items, "total": len(items), "overlap_n": OVERLAP_N, "labeler_id": rid}


@app.get("/api/knowledge")
def get_knowledge(labeler_id: str = Query("", alias="labeler_id")):
    template = _load_gold_csv("gold_knowledge.csv")
    if template.empty:
        return {"items": [], "total": 0, "overlap_n": OVERLAP_N}
    labels = _load_labels_csv("knowledge_labels.csv")
    rid = labeler_id.strip() or "default"
    df = _merge_template_with_labels(
        template, labels, "gold_id", rid,
        ["is_correct", "notes"]
    )
    df = df.fillna("")
    items = df.to_dict(orient="records")
    for r in items:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = ""
            elif hasattr(v, "item"):
                r[k] = str(v)
    return {"items": items, "total": len(items), "overlap_n": OVERLAP_N, "labeler_id": rid}


@app.get("/api/domain")
def get_domain(labeler_id: str = Query("", alias="labeler_id")):
    template = _load_gold_csv("gold_future_domain.csv")
    if template.empty:
        return {"items": [], "total": 0, "overlap_n": OVERLAP_N}
    labels = _load_labels_csv("domain_labels.csv")
    rid = labeler_id.strip() or "default"
    df = _merge_template_with_labels(
        template, labels, "gold_id", rid,
        ["true_domain_id", "notes"]
    )
    df = df.fillna("")
    items = df.to_dict(orient="records")
    for r in items:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = ""
            elif hasattr(v, "item"):
                r[k] = str(v)
    return {"items": items, "total": len(items), "overlap_n": OVERLAP_N, "labeler_id": rid}


@app.post("/api/save_skill_label")
def save_skill_label(lbl: SkillLabel):
    template = _load_gold_csv("gold_skills.csv")
    if template.empty or "gold_id" not in template.columns:
        raise HTTPException(status_code=400, detail="No gold skills template")
    if str(lbl.gold_id) not in template["gold_id"].astype(str).values:
        raise HTTPException(status_code=404, detail="gold_id not found")
    rid = (lbl.labeler_id or "default").strip()
    path = GOLD_LABELS_DIR / "skill_labels.csv"
    _upsert_label(path, "gold_id", lbl.gold_id, rid, {
        "is_correct": lbl.is_correct.strip().lower(),
        "type_label": lbl.type_label.strip(),
        "notes": lbl.notes.strip(),
    })
    return {"ok": True}


@app.post("/api/save_knowledge_label")
def save_knowledge_label(lbl: KnowledgeLabel):
    template = _load_gold_csv("gold_knowledge.csv")
    if template.empty or "gold_id" not in template.columns:
        raise HTTPException(status_code=400, detail="No gold knowledge template")
    if str(lbl.gold_id) not in template["gold_id"].astype(str).values:
        raise HTTPException(status_code=404, detail="gold_id not found")
    rid = (lbl.labeler_id or "default").strip()
    path = GOLD_LABELS_DIR / "knowledge_labels.csv"
    _upsert_label(path, "gold_id", lbl.gold_id, rid, {
        "is_correct": lbl.is_correct.strip().lower(),
        "notes": lbl.notes.strip(),
    })
    return {"ok": True}


@app.post("/api/save_domain_label")
def save_domain_label(lbl: DomainLabel):
    template = _load_gold_csv("gold_future_domain.csv")
    if template.empty or "gold_id" not in template.columns:
        raise HTTPException(status_code=400, detail="No gold domain template")
    if str(lbl.gold_id) not in template["gold_id"].astype(str).values:
        raise HTTPException(status_code=404, detail="gold_id not found")
    rid = (lbl.labeler_id or "default").strip()
    path = GOLD_LABELS_DIR / "domain_labels.csv"
    _upsert_label(path, "gold_id", lbl.gold_id, rid, {
        "true_domain_id": lbl.true_domain_id.strip(),
        "notes": lbl.notes.strip(),
    })
    return {"ok": True}


# --- Static and HTML ---
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
    return html
