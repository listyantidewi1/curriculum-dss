"""
generate_competencies.py

Takes high/medium-verified skills from verified_skills.csv and asks an LLM
(through OpenRouter) to propose competency statements / curriculum components.

Domain-based batching (default): Groups skills by best_future_domain before LLM
calls so each batch contains thematically related skills. Uses normalized-key
lookup for future_skill_weights; confidence thresholds (similarity, mapping_margin)
for high vs low confidence; on-the-fly embedding lookup for unmapped skills;
merge of small batches when domains are strongly similar. Use --no-batch-by-domain
for legacy sequential chunking.

Output:
    competency_proposals.json  (list of JSON objects with competencies)

Use --llm_concurrency N (default: 5) for parallel LLM calls across batches.
Same inputs, model, temperature; only wall-clock time improves. Results
are merged in batch_id order to preserve reproducibility.
"""

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from openai import OpenAI

import config  # uses config.OUTPUT_DIR
from pipeline import AdvancedPipelineConfig
import kkni

COMPETENCY_REQUIRED_FIELDS = {"id", "title", "description", "related_skills", "future_relevance"}
# Soft-skills fields are required after the 2026 reframe but treated as soft-required
# during a transition period: missing values trigger a warning, not a drop.
COMPETENCY_SOFT_SKILLS_FIELDS = {"soft_skills_required", "soft_skills_description"}
# Hard cap on competencies per batch. The prompt asks for at most this many; if the
# LLM ignores it, we keep the most-anchored ones (largest related_skills list).
COMPETENCIES_PER_BATCH_CAP = 12
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 2  # seconds


def _filter_to_hard_skills(df, include_soft: bool):
    """Restrict competency input to Hard / Both skill types.

    No-op if include_soft is True or if the type column is absent. Soft-typed
    rows are excluded so each competency is anchored to hard-skill content;
    soft-skill requirements are surfaced per-competency via soft_skills_required
    (see _load_top_soft_skills).
    """
    if include_soft:
        return df
    if "type" not in df.columns:
        print("[WARN] Skill 'type' column missing; cannot filter to hard skills. Using all rows.")
        return df
    types = df["type"].astype(str).str.strip().str.lower()
    keep = types.isin({"hard", "both"})
    n_before = len(df)
    df_filt = df[keep].copy()
    n_dropped = n_before - len(df_filt)
    if n_dropped > 0:
        print(
            f"[INFO] Filtered competency input to hard/both skills: "
            f"kept {len(df_filt)}, dropped {n_dropped} soft-typed rows "
            f"(soft skills surface per-competency via soft_skills_required)."
        )
    return df_filt


def _load_top_soft_skills(output_dir, top_n: int = 30):
    """Return the top-N most frequent Soft-typed skills extracted from the data.

    Used as prompt context when generating competencies so the LLM can draw
    from observed soft-skill demand. Falls back to an empty list if the source
    file or 'type' column is missing — the LLM is then free to invent.
    """
    from pathlib import Path as _P
    out_dir = _P(output_dir)
    candidate_files = [
        out_dir / "advanced_skills_human_filtered.csv",
        out_dir / "advanced_skills.csv",
    ]
    src = next((p for p in candidate_files if p.exists()), None)
    if src is None:
        print("[INFO] No advanced_skills*.csv found for soft-skills context; LLM will infer soft skills.")
        return []
    try:
        sdf = pd.read_csv(src)
    except Exception as e:
        print(f"[WARN] Could not read {src.name} for soft-skills context: {e}")
        return []
    if "type" not in sdf.columns or "skill" not in sdf.columns:
        return []
    soft_mask = sdf["type"].astype(str).str.strip().str.lower() == "soft"
    soft_df = sdf[soft_mask]
    if soft_df.empty:
        return []
    counts = soft_df.groupby("skill").size().sort_values(ascending=False)
    top = counts.head(top_n).index.tolist()
    print(f"[INFO] Loaded top {len(top)} extracted soft skills as competency prompt context (from {src.name}).")
    return [str(s).strip() for s in top if str(s).strip()]


def load_skill_time_trends(
    output_dir: Path,
    trends_file: str = "skill_time_trends.csv",
) -> Dict[str, Dict]:
    """
    Load skill -> {trend_label, slope, p_value} from skill_time_trends.csv
    (empirical emerging/declining from job posting frequency over time).
    Returns empty dict if file missing.
    """
    path = output_dir / trends_file
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "skill" not in df.columns or "trend_label" not in df.columns:
        return {}
    df = df.drop_duplicates(subset=["skill"], keep="first")
    return {
        str(row["skill"]).strip(): {
            "trend_label": str(row["trend_label"]),
            "slope": float(row.get("slope", 0) or 0),
            "p_value": float(row.get("p_value", 1) or 1),
        }
        for _, row in df.iterrows()
    }


def load_skill_future_weights(
    output_dir: Path,
    weights_file: str = "future_skill_weights.csv",
) -> Dict[str, Dict]:
    """
    Load skill -> {best_future_domain, trend_label, future_weight, similarity, mapping_margin}
    from future_skill_weights.csv. Returns empty dict if file missing.

    Uses normalized-key lookup: exact skill first, then _normalize_for_grouping(skill)
    to reduce coverage gap from case/punctuation variants.
    """
    path = output_dir / weights_file
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"skill", "best_future_domain", "trend_label", "future_weight"}
    if required - set(df.columns):
        return {}
    # Keep first occurrence per skill (file is sorted by future_weight desc)
    df = df.drop_duplicates(subset=["skill"], keep="first")
    from domain_batching import normalize_for_grouping

    result: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        exact_key = str(row["skill"]).strip()
        data = {
            "best_future_domain": str(row["best_future_domain"]),
            "trend_label": str(row["trend_label"]),
            "future_weight": float(row["future_weight"]),
            "similarity": float(row.get("similarity", row.get("top1_similarity", 0)) or 0),
            "mapping_margin": float(row.get("mapping_margin", 0) or 0),
        }
        result[exact_key] = data
        norm_key = normalize_for_grouping(exact_key)
        if norm_key and norm_key not in result:
            result[norm_key] = data
    return result


def get_skill_future_data(skill: str, skill_future_weights: Dict[str, Dict]) -> Optional[Dict]:
    """
    Lookup skill in future weights: exact match first, then normalized key.
    """
    if not skill_future_weights:
        return None
    s = str(skill).strip()
    if s in skill_future_weights:
        return skill_future_weights[s]
    from domain_batching import normalize_for_grouping
    norm_key = normalize_for_grouping(s)
    if norm_key and norm_key in skill_future_weights:
        return skill_future_weights[norm_key]
    return None


def build_future_context(output_dir: Path,
                         future_file: str = "future_skill_weights_dummy.csv",
                         top_k_domains: int = 5,
                         top_k_knowledge: int = 15) -> str:
    """
    Build a short text summary of future-critical domains and knowledge items
    from future_skill_weights_dummy.csv to guide competency generation.
    """
    fw_path = output_dir / future_file
    if not fw_path.exists():
        print(f"[WARN] Future weights file not found at {fw_path}. "
              f"Continuing without future context.")
        return ""

    df = pd.read_csv(fw_path)

    required_cols = {"knowledge", "best_future_domain",
                     "trend_label", "future_weight"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Future weights file missing columns {missing}. "
              f"Continuing without future context.")
        return ""

    # Aggregate by domain
    domain_stats = (
        df.groupby("best_future_domain")["future_weight"]
        .mean()
        .reset_index()
        .sort_values("future_weight", ascending=False)
        .head(top_k_domains)
    )

    # Top knowledge items overall
    top_kw = (
        df.sort_values("future_weight", ascending=False)
        .head(top_k_knowledge)[["knowledge", "best_future_domain", "future_weight"]]
    )

    # Build human-readable context text
    lines = []
    lines.append("Future-critical domains (average future_weight):")
    for _, row in domain_stats.iterrows():
        lines.append(
            f"- {row['best_future_domain']} "
            f"(avg future_weight={row['future_weight']:.2f})"
        )

    lines.append("")
    lines.append("Example high future-weight knowledge items:")
    for _, row in top_kw.iterrows():
        lines.append(
            f"- {row['knowledge']} "
            f"(domain={row['best_future_domain']}, "
            f"future_weight={row['future_weight']:.2f})"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Helper: load OpenRouter / OpenAI client the same way as in pipeline
# ----------------------------------------------------------------------

def load_openrouter_client() -> OpenAI:
    base_url = "https://openrouter.ai/api/v1"

    # 1) env var, if available
    api_key = os.getenv("OPENROUTER_API_KEY")

    # 2) fallback: api_keys/OpenRouter.txt
    if not api_key:
        key_path = Path("api_keys") / "OpenRouter.txt"
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise RuntimeError(
                f"OpenRouter API key not found. Set OPENROUTER_API_KEY "
                f"or create {key_path}"
            )

    if not api_key:
        raise RuntimeError("OpenRouter API key is empty.")

    return OpenAI(api_key=api_key, base_url=base_url)


# ----------------------------------------------------------------------
# Competency deduplication (group similar, track occurrence)
# ----------------------------------------------------------------------


def _normalize_competency_key(text: str) -> str:
    """Normalize competency title for grouping (matches skill/knowledge normalization)."""
    if not text or not isinstance(text, str):
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _jaccard_skills(a_skills: List[str], b_skills: List[str]) -> float:
    """Jaccard similarity between two related_skills lists (normalized by lowercasing)."""
    sa = {str(s).strip().lower() for s in (a_skills or []) if str(s).strip()}
    sb = {str(s).strip().lower() for s in (b_skills or []) if str(s).strip()}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _finalize_kkni(comp: Dict, skill_bloom_map: Optional[Dict[str, str]]) -> None:
    """Compute the deterministic KKNI floor from related_skills' Bloom levels,
    clamp the LLM-proposed kkni_level to [floor, floor+1], and write back the
    canonical KKNI fields. In-place. Idempotent."""
    if not isinstance(comp, dict):
        return
    related = comp.get("related_skills") or []
    bloom_levels = []
    if skill_bloom_map:
        for s in related:
            key = str(s).strip()
            if not key:
                continue
            b = skill_bloom_map.get(key) or skill_bloom_map.get(key.lower())
            if b:
                bloom_levels.append(str(b).strip())
    floor = kkni.kkni_floor_for_competency(bloom_levels)
    proposed = comp.get("kkni_level")
    final_level = kkni.clamp_llm_kkni(proposed, floor)
    comp["kkni_level"] = final_level
    comp["kkni_floor"] = floor
    comp["kkni_descriptor"] = kkni.kkni_descriptor(final_level)


def _absorb_soft_skills(target: Dict, source: Dict) -> None:
    """Merge source's soft_skills_required (set union) and soft_skills_description
    (keep target's if present, else use source's) into target — in place."""
    t_soft = list(target.get("soft_skills_required") or [])
    s_soft = list(source.get("soft_skills_required") or [])
    seen = set()
    merged: List[str] = []
    for s in t_soft + s_soft:
        key = str(s).strip().lower()
        if key and key not in seen:
            seen.add(key)
            merged.append(str(s).strip())
    target["soft_skills_required"] = merged
    if not str(target.get("soft_skills_description") or "").strip():
        target["soft_skills_description"] = str(source.get("soft_skills_description") or "").strip()


def _merge_by_skills_overlap(
    comps: List[Dict], threshold: float = 0.5
) -> List[Dict]:
    """
    Merge competencies with high Jaccard overlap in related_skills.
    Greedy: process by occurrence_count desc; merge into first match.
    Preserves importance: occurrence_count is summed when merging.
    Soft-skill fields are absorbed (set union for the list, first non-empty for the description).
    """
    if not comps or threshold <= 0:
        return comps
    comps = sorted(
        comps,
        key=lambda c: (-int(c.get("occurrence_count", 1)), c.get("title", "")),
    )
    result: List[Dict] = []
    for c in comps:
        skills = set(
            str(s).strip().lower()
            for s in (c.get("related_skills") or [])
            if str(s).strip()
        )
        occ = int(c.get("occurrence_count", 1))
        merged = False
        for r in result:
            r_skills = set(
                str(s).strip().lower()
                for s in (r.get("related_skills") or [])
                if str(s).strip()
            )
            j = _jaccard_skills(list(skills), list(r_skills))
            if j >= threshold:
                r["occurrence_count"] = int(r.get("occurrence_count", 1)) + occ
                combined = r_skills | skills
                r["related_skills"] = sorted(combined)
                _absorb_soft_skills(r, c)
                merged = True
                break
        if not merged:
            result.append(c.copy())
    return result


def _merge_by_semantic_title(
    comps: List[Dict],
    embedder,
    title_threshold: float = 0.85,
    skills_jaccard_threshold: float = 0.40,
) -> List[Dict]:
    """Merge competencies whose titles are semantically near-duplicates AND share
    enough related skills. Catches synonymous competencies (e.g. "Data Analysis"
    vs "Analyze Data") that exact normalized-title matching misses.

    Two competencies merge when:
        SBERT cosine(title_a, title_b) >= title_threshold
        AND Jaccard(related_skills_a, related_skills_b) >= skills_jaccard_threshold
    """
    if not comps or embedder is None or len(comps) < 2:
        return comps
    titles = [str(c.get("title", "")).strip() for c in comps]
    if not all(titles):
        return comps  # don't bother on incomplete data
    try:
        import numpy as np
        emb = embedder.encode(titles, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:
        print(f"[WARN] Semantic title dedup skipped ({e}).")
        return comps

    n = len(comps)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[max(px, py)] = min(px, py)  # keep earliest as canonical

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(emb[i], emb[j]))
            if sim < title_threshold:
                continue
            jac = _jaccard_skills(comps[i].get("related_skills") or [], comps[j].get("related_skills") or [])
            if jac >= skills_jaccard_threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    result: List[Dict] = []
    n_merged = 0
    for root, members in clusters.items():
        members.sort()
        canonical = comps[members[0]].copy()
        if len(members) == 1:
            result.append(canonical)
            continue
        n_merged += len(members) - 1
        related: set = set(str(s).strip() for s in (canonical.get("related_skills") or []) if str(s).strip())
        occ = int(canonical.get("occurrence_count", 1))
        for idx in members[1:]:
            other = comps[idx]
            for s in other.get("related_skills") or []:
                if str(s).strip():
                    related.add(str(s).strip())
            occ += int(other.get("occurrence_count", 1))
            _absorb_soft_skills(canonical, other)
        canonical["related_skills"] = sorted(related)
        canonical["occurrence_count"] = occ
        result.append(canonical)
    if n_merged:
        print(f"[INFO] Semantic-title dedup merged {n_merged} near-duplicate competencies.")
    return result


def _deduplicate_competencies(
    comps: List[Dict],
    merge_overlap_threshold: float = 0.0,
    semantic_embedder=None,
    semantic_title_threshold: float = 0.85,
    semantic_skills_jaccard: float = 0.40,
) -> List[Dict]:
    """
    Three-stage dedup:
      1. Normalized-title exact match (cheap).
      2. Semantic title near-duplicate match (SBERT cosine ≥ semantic_title_threshold
         AND skills Jaccard ≥ semantic_skills_jaccard) — only when semantic_embedder
         is provided.
      3. Optional related_skills Jaccard overlap merge (merge_overlap_threshold > 0).
    Soft-skill fields are absorbed (set union for the list, first non-empty for the description).
    """
    if not comps:
        return comps
    by_key: Dict[str, List[Dict]] = defaultdict(list)
    for c in comps:
        title = c.get("title", "") or c.get("description", "")[:100]
        key = _normalize_competency_key(title)
        if not key:
            by_key[f"_unnamed_{len(by_key)}"].append(c)
        else:
            by_key[key].append(c)

    out = []
    for key, group in by_key.items():
        canonical = group[0].copy()
        canonical["occurrence_count"] = len(group)
        if len(group) > 1:
            related = set()
            for g in group:
                for s in g.get("related_skills") or []:
                    related.add(str(s).strip())
                _absorb_soft_skills(canonical, g)
            canonical["related_skills"] = sorted(related)
        out.append(canonical)

    if semantic_embedder is not None:
        out = _merge_by_semantic_title(
            out,
            semantic_embedder,
            title_threshold=semantic_title_threshold,
            skills_jaccard_threshold=semantic_skills_jaccard,
        )

    if merge_overlap_threshold > 0:
        out = _merge_by_skills_overlap(out, threshold=merge_overlap_threshold)
    return out


# ----------------------------------------------------------------------
# Few-shot examples from human assessments
# ----------------------------------------------------------------------

def load_few_shot_examples(
    output_dir: Path,
    feedback_dir: Path,
    proposals_file: str = "competency_proposals.json",
    assessments_file: str = "competency_assessments.json",
    min_quality: int = 4,
    max_examples: int = 3,
) -> List[Dict]:
    """
    Load high-quality competencies (human_quality >= min_quality, human_relevant=yes)
    as few-shot examples for prompt tuning.
    """
    proposals_path = output_dir / proposals_file
    assessments_path = feedback_dir / assessments_file
    if not proposals_path.exists() or not assessments_path.exists():
        return []

    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    comps = proposals.get("competencies", [])
    assessments = json.loads(assessments_path.read_text(encoding="utf-8"))

    examples = []
    for c in comps:
        cid = c.get("id", "")
        bid = c.get("batch_id", 0)
        key = f"{cid}_b{bid}" if bid else cid
        a = assessments.get(key, {})
        q = str(a.get("quality", "")).strip()
        r = str(a.get("relevant", "")).lower().strip()
        q_num = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}.get(q, 0)
        if q_num >= min_quality and r == "yes":
            examples.append(c)
            if len(examples) >= max_examples:
                break
    return examples


# ----------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------

def build_prompt(
    skills: List[str],
    future_context: str = "",
    few_shot_examples: Optional[List[Dict]] = None,
    skill_future_weights: Optional[Dict[str, Dict]] = None,
    skill_time_trends: Optional[Dict[str, Dict]] = None,
    skill_bloom_map: Optional[Dict[str, str]] = None,
    curriculum_context: str = "",
    vocational_domain: str = "",
    top_soft_skills: Optional[List[str]] = None,
) -> str:
    # Annotate skills with Bloom, future_weight, and/or empirical trend when available
    def format_skill(s: str) -> str:
        s_stripped = s.strip()
        parts = []
        if skill_bloom_map and s_stripped in skill_bloom_map:
            parts.append(f"bloom={skill_bloom_map[s_stripped]}")
        w = get_skill_future_data(s_stripped, skill_future_weights or {})
        if w:
            fw = w.get("future_weight", 0)
            domain = w.get("best_future_domain", "")
            trend = w.get("trend_label", "")
            parts.append(f"future_weight={fw:.2f}, domain={domain}, forecast_trend={trend}")
        if skill_time_trends and s_stripped in skill_time_trends:
            t = skill_time_trends[s_stripped]
            emp = t.get("trend_label", "")
            if emp and emp != "Stable":
                parts.append(f"empirical_trend={emp}")
        if parts:
            return f"- {s} ({'; '.join(parts)})"
        return f"- {s}"

    bullet_list = "\n".join(format_skill(s) for s in skills)

    future_block = ""
    if future_context.strip():
        future_block = f"""
Additional context about future-critical domains and technologies:

{future_context}
"""

    few_shot_block = ""
    if few_shot_examples:
        ex_str = json.dumps(
            [{"id": e.get("id"), "title": e.get("title"), "description": e.get("description"),
              "related_skills": e.get("related_skills", []), "future_relevance": e.get("future_relevance", "")}
             for e in few_shot_examples],
            indent=2,
            ensure_ascii=False,
        )
        few_shot_block = f"""
Here are examples of HIGH-QUALITY competencies (use similar style and structure):

{ex_str}

"""

    future_weighting_block = ""
    if skill_future_weights or skill_time_trends:
        future_weighting_block = """
10. FUTURE WEIGHTING (important): Skills may be annotated with:
   - future_weight, domain, forecast_trend: from domain forecasts (Strong_Growth, Moderate_Growth, Decline).
   - empirical_trend: from actual job posting frequency over time (Emerging, Declining).
   - PRIORITIZE competencies that cover skills with HIGH future_weight and/or empirical_trend=Emerging.
   - Give emerging skills more prominence in competency titles and descriptions.
   - DEPRIORITIZE competencies built mainly around skills with NEGATIVE future_weight or empirical_trend=Declining.
   - Do not create competencies that focus primarily on declining skills.

"""
    domain_line = (
        f" in the {vocational_domain} domain"
        if vocational_domain
        else " in the Software & Game Development domain"
    )
    curriculum_block = ""
    if curriculum_context and curriculum_context.strip():
        curriculum_block = f"""

Existing curriculum components (align where relevant):
{curriculum_context}
"""

    soft_skills_block = ""
    if top_soft_skills:
        soft_list = ", ".join(top_soft_skills)
        soft_skills_block = f"""

Top soft skills observed in the same job postings (use these as primary candidates
when filling soft_skills_required; you MAY add others if obviously required):
{soft_list}
"""

    # KKNI reference. The LLM is asked to choose a kkni_level constrained
    # to a Bloom-derived floor (computed per competency, post-hoc). Showing
    # the full 1-9 table here gives it the context it needs.
    kkni_block = "\n" + kkni.kkni_reference_for_prompt() + "\n"

    return f"""
You are an expert in competency-based education and vocational curriculum design.

You are given a list of VERIFIED HARD job skills extracted from real job
postings{domain_line}. Each competency you produce is a HARD-SKILL competency;
soft skills are surfaced separately, per competency, via the soft_skills_required
and soft_skills_description fields.
{curriculum_block}{soft_skills_block}

Your task is to group related hard skills and generate COMPETENCY STATEMENTS that are
suitable for use in an upper-secondary / vocational curriculum (e.g., Indonesian SMK).

Please follow these rules:

1. Output MUST be valid JSON only (no explanation text).
2. The root should be a JSON object with key "competencies" whose value is a list.
3. Each competency is an object with:
   - "id": a short identifier (e.g., "C1", "C2", ...).
   - "title": a concise competency title.
   - "description": ONE single sentence that is clear, measurable, and operational.
     Start with a Bloom-level verb (Design, Analyze, Evaluate, Create, etc.); state the object/outcome;
     include how (measurable, operational criteria).
     Example: "Design scalable, high-performance software components by breaking down complex
     requirements into smaller, manageable subsystems and choosing appropriate technologies."
     Focus on learner-demonstrable outcomes, not business impact.
   - "related_skills": list of HARD-skill phrases from the input that this competency covers.
     Do NOT add any skill to related_skills that was not in the input list above.
     Only reference skills that appear in the list. Paraphrasing or inventing skills is not allowed.
   - "future_relevance": a short note (1–2 sentences) on why this competency
     matters for the future of work (based on the context if available).
   - "soft_skills_required": a list of 3 to 6 soft-skill phrases that a learner needs to
     perform this competency effectively (e.g. ["Communication", "Teamwork", "Critical Thinking"]).
     PREFER soft skills from the "Top soft skills observed" list above when they fit; you may
     add others if they are obviously required for this competency. Do not duplicate hard skills here.
   - "soft_skills_description": ONE concise sentence (≤ 30 words) that explains how those soft
     skills support performance of this competency. This sentence is shown verbatim in the
     school-facing curriculum report, so write it in plain, human-readable language.
   - "kkni_level": INTEGER 1-9 indicating the appropriate KKNI qualification level for this
     competency. Use the KKNI reference table below. The level you pick MUST be consistent
     with the highest Bloom level among the related_skills:
       Remember/Understand → KKNI 1-2 (operator level, SMP/SMA);
       Apply              → KKNI 3-4 (skilled operator / junior teknisi, SMK/D1/D2);
       Analyze            → KKNI 5-6 (teknisi analis, D3/D4/S1);
       Evaluate           → KKNI 6-7 (senior analis / profesi);
       Create             → KKNI 7-9 (ahli / magister / doktor).
     Pick the level that best fits the integrative scope of the competency. The post-validator
     will clamp out-of-band values, so stay within the band shown for the highest Bloom.
4. BLOOM ALIGNMENT: For each competency, use the HIGHEST Bloom taxonomy level among its
   related skills (e.g. if skills have Apply, Understand, Analyze, write at Analyze level).
   Bloom order: Remember < Understand < Apply < Analyze < Evaluate < Create.
5. COMPETENCY COUNT: Produce AT MOST 12 competencies per batch. Coalesce overlapping
   themes into a single broader competency rather than splitting them. Quality over
   quantity — fewer well-anchored competencies are better than many shallow ones.
6. Avoid creating competencies that strongly overlap with others in this batch; prefer distinct themes.
7. Prefer higher-level, integrative competencies, not trivial one-skill items.
8. CURRICULUM LANGUAGE: Write competencies as if for a Ministry of Education curriculum document.
   Use learning-outcome language (what the learner can demonstrate or produce).
   AVOID HR/job-description phrasing: "actionable insights", "strategic decisions", "inform
   stakeholders", "drive value", "business impact", "competitive advantage", "meet business needs".
   USE INSTEAD: "identify patterns", "communicate findings", "apply methods", "demonstrate
   understanding", "present recommendations", "evaluate alternatives", "create artifacts".
   PRESERVE all substantive technical and cognitive requirements (methods, tools, processes).
   Competencies must be ASSESSABLE (e.g., by test, project, or portfolio).
9. Give slightly higher priority and more detail to skills and themes that appear
   in future-critical domains (AI, data, cloud, security, human–AI collaboration),
   if such context is provided.
{future_weighting_block}
{few_shot_block}
{future_block}
{kkni_block}

Here are the verified hard skills:

{bullet_list}
"""


# ----------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------

def _validate_competency(c: dict) -> bool:
    """Check that a competency dict has all required fields.

    Soft-skill fields (soft_skills_required, soft_skills_description) are
    soft-required: missing values are normalized to empty list / empty
    string in-place rather than failing validation. This keeps older
    cached results loadable while we transition the schema.
    """
    if not (
        isinstance(c, dict)
        and all(k in c for k in COMPETENCY_REQUIRED_FIELDS)
        and isinstance(c.get("related_skills"), list)
    ):
        return False
    # Normalize soft-skill fields if missing or wrong type — do not drop the competency.
    if "soft_skills_required" not in c or not isinstance(c.get("soft_skills_required"), list):
        c["soft_skills_required"] = []
    else:
        c["soft_skills_required"] = [str(s).strip() for s in c["soft_skills_required"] if str(s).strip()]
    if "soft_skills_description" not in c or not isinstance(c.get("soft_skills_description"), str):
        c["soft_skills_description"] = ""
    # KKNI level is normalized post-clamp by _finalize_kkni; here we only ensure
    # the key exists so older cached competencies remain loadable.
    if "kkni_level" not in c:
        c["kkni_level"] = None
    return True


def _parse_llm_json(content: str) -> Optional[Dict]:
    """Parse LLM response JSON with fallback strategies."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n```$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    last_brace = max(content.rfind("}"), content.rfind("]"))
    if last_brace != -1:
        try:
            return json.loads(content[: last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def call_llm_for_competencies(client: OpenAI,
                              skills: List[str],
                              model_name: str,
                              future_context: str = "",
                              few_shot_examples: Optional[List[Dict]] = None,
                              skill_future_weights: Optional[Dict[str, Dict]] = None,
                              skill_time_trends: Optional[Dict[str, Dict]] = None,
                              skill_bloom_map: Optional[Dict[str, str]] = None,
                              curriculum_context: str = "",
                              vocational_domain: str = "",
                              top_soft_skills: Optional[List[str]] = None,
                              temperature: float = 0.0) -> Dict:
    prompt = build_prompt(
        skills,
        future_context=future_context,
        few_shot_examples=few_shot_examples,
        skill_future_weights=skill_future_weights,
        skill_time_trends=skill_time_trends,
        skill_bloom_map=skill_bloom_map,
        curriculum_context=curriculum_context,
        vocational_domain=vocational_domain,
        top_soft_skills=top_soft_skills,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise curriculum designer drafting competencies for Ministry of Education curriculum documents. "
                "Use learning-outcome language (what learners can demonstrate). "
                "Avoid HR or job-description phrasing. "
                "Always respond with VALID JSON only. "
                "Do NOT include any markdown fences or commentary."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=4000,
            )
            content = resp.choices[0].message.content or ""
            data = _parse_llm_json(content)

            if data is not None:
                comps = data.get("competencies", [])
                valid = [c for c in comps if _validate_competency(c)]
                invalid_count = len(comps) - len(valid)
                if invalid_count > 0:
                    print(f"[WARN] Dropped {invalid_count} competencies "
                          f"with missing fields (attempt {attempt + 1})")
                # Post-validation: filter related_skills to only those in input (anti-hallucination)
                input_skills_set = {str(s).strip().lower() for s in skills}
                for c in valid:
                    rs = c.get("related_skills") or []
                    filtered = [s for s in rs if str(s).strip().lower() in input_skills_set]
                    if len(filtered) < len(rs):
                        print(f"[WARN] Filtered {len(rs) - len(filtered)} non-input skills "
                              f"from competency {c.get('id', '?')}")
                    c["related_skills"] = filtered

                # KKNI: clamp LLM-proposed kkni_level to [floor, floor+1] where
                # floor is derived from the highest Bloom in related_skills.
                for c in valid:
                    _finalize_kkni(c, skill_bloom_map)

                # Hard cap: at most COMPETENCIES_PER_BATCH_CAP per batch.
                # If the LLM ignored rule 5 and produced more, keep the most-anchored
                # ones (largest related_skills list = best supported by input).
                if len(valid) > COMPETENCIES_PER_BATCH_CAP:
                    before = len(valid)
                    valid = sorted(
                        valid,
                        key=lambda c: len(c.get("related_skills") or []),
                        reverse=True,
                    )[:COMPETENCIES_PER_BATCH_CAP]
                    print(
                        f"[WARN] Batch produced {before} competencies; "
                        f"capping to {COMPETENCIES_PER_BATCH_CAP} (kept the most-anchored)."
                    )
                data["competencies"] = valid
                return data

            last_error = "JSON parse failed"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            print(f"[WARN] Attempt {attempt + 1} failed ({last_error}), "
                  f"retrying in {wait}s...")
            time.sleep(wait)

    raw = content if 'content' in locals() else f"Error: {last_error}"
    with open("last_competency_raw_response.txt", "w", encoding="utf-8") as f:
        f.write(raw)

    print(f"[WARN] All {1 + MAX_RETRIES} attempts failed. "
          f"Raw content saved to last_competency_raw_response.txt.")
    return {"competencies": []}



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate competency proposals from verified skills using an LLM."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing verified_skills.csv (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--verified_file",
        type=str,
        default="verified_skills.csv",
        help="Input verified skills file (default: verified_skills.csv)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="competency_proposals.json",
        help="Output JSON file (default: competency_proposals.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="Model name on OpenRouter (default: deepseek/deepseek-v3.2)",
    )
    parser.add_argument(
        "--max_skills_per_call",
        type=int,
        default=30,
        help="Number of skills per LLM call (default: 30)",
    )
    parser.add_argument(
        "--human_verified_only",
        action="store_true",
        help="Use only human-verified skills from feedback_store/human_verified_skills.csv",
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Use all skills (with Bloom/type corrections) from advanced_skills_human_filtered.csv. "
             "Tags each competency with all_skills_human_verified. Some competencies may be from unverified skills.",
    )
    parser.add_argument(
        "--feedback_dir",
        type=str,
        default=None,
        help="Feedback store directory (default: PROJECT_ROOT/feedback_store)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature for reproducibility (0=deterministic, default: 0)",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable competency deduplication (group similar titles, aggregate occurrence)",
    )
    parser.add_argument(
        "--merge-overlap-threshold",
        type=float,
        default=0.5,
        help="Jaccard threshold for merging competencies by related_skills overlap (0=off, default: 0.5)",
    )
    parser.add_argument(
        "--no-batch-by-domain",
        action="store_true",
        help="Disable domain-based batching; use sequential chunking (legacy).",
    )
    parser.add_argument(
        "--domain-similarity-threshold",
        type=float,
        default=0.45,
        help="Below this similarity, treat domain assignment as low-confidence -> Uncertain batch (default: 0.45)",
    )
    parser.add_argument(
        "--domain-margin-threshold",
        type=float,
        default=0.05,
        help="Below this mapping_margin, treat as low-confidence -> Uncertain batch (default: 0.05)",
    )
    parser.add_argument(
        "--domain-order",
        type=str,
        default="mean_future_weight",
        choices=["mean_future_weight", "trend_first"],
        help="Order domain batches: mean_future_weight (default) or trend_first",
    )
    parser.add_argument(
        "--no-merge-small-domain-batches",
        action="store_true",
        help="Disable merging of small domain batches when domains are strongly similar.",
    )
    parser.add_argument(
        "--merge-domain-similarity-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for merging domain batches (default: 0.7)",
    )
    parser.add_argument(
        "--future-domains-file",
        type=str,
        default=None,
        help="Path to future_domains.csv (default: PROJECT_ROOT/future_domains.csv)",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        default=None,
        help="Path to curriculum file (CSV/JSON) for domain-specific prompts. Uses curriculum_loader.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Vocational domain/field (e.g., RPL, Pengembangan Gim, Akuntansi) to tailor prompt.",
    )
    parser.add_argument(
        "--llm_concurrency",
        type=int,
        default=5,
        help="Parallel LLM workers for competency generation (1=sequential, default: 5)",
    )
    parser.add_argument(
        "--include-soft-in-competencies",
        action="store_true",
        help="Include Soft-typed skills in competency input (legacy behavior). "
             "Default after 2026 reframe is hard-only: soft skills are surfaced per-competency "
             "via the soft_skills_required field instead of becoming competencies themselves.",
    )
    parser.add_argument(
        "--soft-skills-context-top-n",
        type=int,
        default=30,
        help="Number of top extracted Soft skills to include as prompt context for the LLM "
             "to draw from when filling soft_skills_required (default: 30).",
    )

    args = parser.parse_args()
    args.deduplicate = not args.no_deduplicate
    args.batch_by_domain = not args.no_batch_by_domain
    args.merge_small_domain_batches = not args.no_merge_small_domain_batches
    if args.human_verified_only and args.comprehensive:
        raise ValueError("Cannot use both --human_verified_only and --comprehensive. Choose one.")

    out_dir = Path(args.output_dir)
    output_path = out_dir / args.output_json
    feedback_dir = Path(args.feedback_dir) if args.feedback_dir else Path(config.PROJECT_ROOT) / "feedback_store"

    # Build future-of-work context from future_skill_weights_dummy.csv (if available)
    future_context = build_future_context(out_dir)
    if future_context:
        print("[INFO] Future context loaded and will be injected into LLM prompt.")
    else:
        print("[INFO] No future context available; generating competencies from skills only.")

    # Load curriculum for domain-specific context (if provided)
    curriculum_context = ""
    if args.curriculum:
        curr_path = Path(args.curriculum)
        if curr_path.exists():
            try:
                from curriculum_loader import load_curriculum_file
                components = load_curriculum_file(curr_path)
                if components:
                    lines = []
                    for c in components[:20]:  # Limit to avoid token overflow
                        comp_name = c.get("component_name", "") or c.get("component_id", "")
                        bloom = c.get("bloom_level", "")
                        phrases = c.get("phrases", [])
                        phrases_str = ", ".join(str(p) for p in (phrases[:5] if isinstance(phrases, list) else []))
                        lines.append(f"  - {comp_name} (Bloom: {bloom}): {phrases_str}")
                    curriculum_context = "\n".join(lines)
                    print(f"[INFO] Loaded {len(components)} curriculum components from {curr_path.name}")
            except Exception as e:
                print(f"[WARN] Could not load curriculum from {args.curriculum}: {e}")
        else:
            print(f"[WARN] Curriculum file not found: {args.curriculum}")

    vocational_domain = (args.domain or "").strip() or None
    if vocational_domain:
        print(f"[INFO] Vocational domain for prompts: {vocational_domain}")

    # Load human-verified skill set (for tagging in comprehensive mode)
    human_verified_skills: set = set()
    human_path = feedback_dir / "human_verified_skills.csv"
    if human_path.exists():
        hv_df = pd.read_csv(human_path)
        if "skill" in hv_df.columns:
            human_verified_skills = {
                str(s).strip().lower() for s in hv_df["skill"].dropna().unique()
            }
            print(f"[INFO] Loaded {len(human_verified_skills)} human-verified skills for tagging")

    if args.human_verified_only:
        human_path = feedback_dir / "human_verified_skills.csv"
        if not human_path.exists():
            raise FileNotFoundError(
                f"human_verified_skills.csv not found at {human_path}. "
                "Run export_for_review, review in web app, then import_feedback first."
            )
        print(f"[INFO] Using human-verified skills from {human_path}")
        df = pd.read_csv(human_path)
        if "skill" not in df.columns:
            raise ValueError("human_verified_skills.csv must contain 'skill' column")
        df = _filter_to_hard_skills(df, args.include_soft_in_competencies)
        grp = df.groupby("skill").size().reset_index()
        grp.columns = ["skill", "freq"]
        skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()
    elif args.comprehensive:
        comp_path = out_dir / "advanced_skills_human_filtered.csv"
        if not comp_path.exists():
            comp_path = out_dir / "advanced_skills.csv"
        if not comp_path.exists():
            raise FileNotFoundError(
                f"Neither advanced_skills_human_filtered.csv nor advanced_skills.csv found in {out_dir}. "
                "Run apply_feedback first (without --filter_only) to apply Bloom/type corrections."
            )
        print(f"[INFO] Comprehensive mode: using all skills from {comp_path.name}")
        df = pd.read_csv(comp_path)
        if "skill" not in df.columns:
            raise ValueError(f"{comp_path.name} must contain 'skill' column")
        df = _filter_to_hard_skills(df, args.include_soft_in_competencies)
        grp = df.groupby("skill").size().reset_index()
        grp.columns = ["skill", "freq"]
        skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()
    else:
        verified_path = out_dir / args.verified_file
        if not verified_path.exists():
            raise FileNotFoundError(f"verified_skills.csv not found: {verified_path}")
        print(f"[INFO] Reading verified skills from {verified_path}")
        df = pd.read_csv(verified_path)
        if "is_verified" not in df.columns:
            raise ValueError("verified_skills.csv must contain 'is_verified' column")
        df_v = df[df["is_verified"] == True].copy()
        if df_v.empty:
            raise RuntimeError("No verified skills found (is_verified == True).")
        df_v = _filter_to_hard_skills(df_v, args.include_soft_in_competencies)
        if df_v.empty:
            raise RuntimeError(
                "No hard/both skills remaining after filtering. "
                "Pass --include-soft-in-competencies to include soft skills, "
                "or check that the type column has Hard/Both values."
            )
        grp = df_v.groupby("skill").size().reset_index()
        grp.columns = ["skill", "freq"]
        skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()

    print(f"[INFO] Total unique skills for competency generation: {len(skills_sorted)}")

    # Build skill->Bloom map (mode or first non-N/A per skill)
    skill_bloom_map = {}
    if "bloom" in df.columns and "skill" in df.columns:
        for skill in skills_sorted:
            s = str(skill).strip()
            vals = df[df["skill"].astype(str).str.strip() == s]["bloom"].dropna().astype(str).str.strip()
            vals = vals[vals.str.lower() != "n/a"]
            if len(vals) > 0:
                try:
                    mode_val = vals.mode().iloc[0]
                except (IndexError, KeyError):
                    mode_val = vals.iloc[0]
                skill_bloom_map[s] = str(mode_val)
        if skill_bloom_map:
            print(f"[INFO] Loaded Bloom levels for {len(skill_bloom_map)} skills")
    else:
        print("[INFO] Bloom column not available; Bloom alignment optional in prompt")

    # Build skill->type map (for downranking vague single-word hard skills)
    skill_type_map: Dict[str, str] = {}
    if "skill" in df.columns and "type" in df.columns:
        for _, row in df.drop_duplicates(subset=["skill"]).iterrows():
            sk = str(row.get("skill", "")).strip().lower()
            if sk:
                skill_type_map[sk] = str(row.get("type", "")).strip()

    # Load skill future weights (for annotation and reordering)
    skill_future_weights = load_skill_future_weights(out_dir)
    if skill_future_weights:
        print(f"[INFO] Loaded future weights for {len(skill_future_weights)} skills")

    # Load empirical time trends (emerging/declining from job posting frequency)
    skill_time_trends = load_skill_time_trends(out_dir)
    if skill_time_trends:
        print(f"[INFO] Loaded empirical trends for {len(skill_time_trends)} skills")

    # Load top extracted soft skills as competency-prompt context
    # (LLM may pick from these for soft_skills_required, but is allowed to add others).
    top_soft_skills = _load_top_soft_skills(out_dir, top_n=args.soft_skills_context_top_n)

    # Reorder skills: future_weight first, then empirical Emerging > Stable > Declining.
    # Downrank single-word hard skills (vague) so they appear later.
    def sort_key(s: str) -> tuple:
        s_stripped = s.strip()
        fw = -999.0
        w = get_skill_future_data(s_stripped, skill_future_weights or {})
        if w:
            fw = w.get("future_weight", -999.0)
        emp_rank = 0  # 2=Emerging, 1=Stable, 0=Declining or unknown
        if skill_time_trends and s_stripped in skill_time_trends:
            lbl = skill_time_trends[s_stripped].get("trend_label", "")
            emp_rank = {"Emerging": 2, "Stable": 1, "Declining": 0}.get(lbl, 0)
        is_vague = (
            skill_type_map.get(s_stripped.lower(), "").lower() == "hard"
            and len(s_stripped.split()) < 2
        )
        vague_rank = 0 if is_vague else 1  # vague sorts later
        return (fw, emp_rank, vague_rank)

    skills_sorted = sorted(skills_sorted, key=sort_key, reverse=True)
    if skill_future_weights or skill_time_trends:
        print("[INFO] Skills reordered by future_weight and empirical trend (emerging first)")

    # Load few-shot examples from human assessments (if available)
    few_shot = load_few_shot_examples(out_dir, feedback_dir, max_examples=3)
    if few_shot:
        print(f"[INFO] Using {len(few_shot)} few-shot examples from human assessments")

    client = load_openrouter_client()

    all_competencies = []
    batch_id = 1

    # Build batches: domain-based or sequential (legacy)
    if args.batch_by_domain:
        from domain_batching import (
            assign_domain_for_skill,
            build_domain_batches,
            compute_domain_embeddings,
            load_future_domains,
            merge_similar_domain_batches,
        )
        from sentence_transformers import SentenceTransformer

        skill_to_domain: Dict[str, str] = {}
        skill_to_domain_data: Dict[str, Dict] = {}

        domains_path = Path(
            args.future_domains_file
            if args.future_domains_file
            else config.PROJECT_ROOT / "future_domains.csv"
        )
        domains_df = load_future_domains(domains_path)
        embedder = None
        if not domains_df.empty:
            embedder = SentenceTransformer(AdvancedPipelineConfig.EMBEDDING_MODEL)

        for s in skills_sorted:
            s_stripped = s.strip()
            w = get_skill_future_data(s_stripped, skill_future_weights or {})
            if w:
                sim = w.get("similarity", 0) or 0
                margin = w.get("mapping_margin", 0) or 0
                high_conf = (
                    sim >= args.domain_similarity_threshold
                    and margin >= args.domain_margin_threshold
                )
                if high_conf:
                    domain = w.get("best_future_domain", "Unmapped")
                    skill_to_domain[s_stripped] = domain
                    skill_to_domain_data[s_stripped] = w
                else:
                    skill_to_domain[s_stripped] = "Uncertain"
                    skill_to_domain_data[s_stripped] = w
            else:
                # On-the-fly embedding lookup
                if embedder is not None and not domains_df.empty:
                    domain, fw, trend = assign_domain_for_skill(s, domains_df, embedder)
                    skill_to_domain[s_stripped] = domain
                    skill_to_domain_data[s_stripped] = {
                        "best_future_domain": domain,
                        "trend_label": trend,
                        "future_weight": fw,
                        "similarity": 0,
                        "mapping_margin": 0,
                    }
                else:
                    skill_to_domain[s_stripped] = "Unmapped"
                    skill_to_domain_data[s_stripped] = {
                        "best_future_domain": "Unmapped",
                        "trend_label": "",
                        "future_weight": -999,
                        "similarity": 0,
                        "mapping_margin": 0,
                    }

        batches = build_domain_batches(
            skills_sorted,
            skill_to_domain,
            skill_to_domain_data,
            args.max_skills_per_call,
            domain_order=args.domain_order,
            embedder=embedder,
        )

        if args.merge_small_domain_batches and embedder is not None and not domains_df.empty:
            domain_embeddings = compute_domain_embeddings(domains_df, embedder)
            if domain_embeddings:
                batches = merge_similar_domain_batches(
                    batches,
                    domain_embeddings,
                    args.merge_domain_similarity_threshold,
                    max_per_batch=args.max_skills_per_call,
                )

        domain_counts = {}
        for domain, chunk in batches:
            domain_counts[domain] = domain_counts.get(domain, 0) + len(chunk)
        print(f"[INFO] Domain-based batching: {len(batches)} batches, "
              f"domains: {list(domain_counts.keys())[:10]}...")

        concurrency = getattr(args, "llm_concurrency", 5) or 5
        if concurrency > 1:
            print(f"[INFO] Using {concurrency} concurrent LLM workers for competency generation")
            batch_tasks = [
                (bid, domain, chunk)
                for bid, (domain, chunk) in enumerate(batches, start=1)
            ]

            def _process_batch(task: Tuple[int, str, List[str]]) -> Tuple[int, Optional[str], List[Dict]]:
                bid, domain, chunk = task
                print(f"[INFO] Calling LLM for batch {bid} (domain={domain}, {len(chunk)} skills)...")
                data = call_llm_for_competencies(
                    client,
                    chunk,
                    args.model,
                    future_context=future_context,
                    few_shot_examples=few_shot or [],
                    skill_future_weights=skill_future_weights or None,
                    skill_time_trends=skill_time_trends or None,
                    skill_bloom_map=skill_bloom_map or None,
                    curriculum_context=curriculum_context,
                    vocational_domain=vocational_domain or "",
                    top_soft_skills=top_soft_skills,
                    temperature=args.temperature,
                )
                comps = data.get("competencies", [])
                for c in comps:
                    c.setdefault("batch_id", bid)
                    c.setdefault("batch_domain", domain)
                return (bid, domain, comps)

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_process_batch, t): t for t in batch_tasks}
                results_by_id = {}
                for future in as_completed(futures):
                    bid, domain, comps = future.result()
                    results_by_id[bid] = comps
            for bid in sorted(results_by_id):
                all_competencies.extend(results_by_id[bid])
        else:
            for domain, chunk in batches:
                print(f"[INFO] Calling LLM for batch {batch_id} (domain={domain}, {len(chunk)} skills)...")
                data = call_llm_for_competencies(
                    client,
                    chunk,
                    args.model,
                    future_context=future_context,
                    few_shot_examples=few_shot or [],
                    skill_future_weights=skill_future_weights or None,
                    skill_time_trends=skill_time_trends or None,
                    skill_bloom_map=skill_bloom_map or None,
                    curriculum_context=curriculum_context,
                    vocational_domain=vocational_domain or "",
                    top_soft_skills=top_soft_skills,
                    temperature=args.temperature,
                )
                comps = data.get("competencies", [])
                for c in comps:
                    c.setdefault("batch_id", batch_id)
                    c.setdefault("batch_domain", domain)
                all_competencies.extend(comps)
                batch_id += 1
    else:
        # Legacy sequential chunking
        concurrency = getattr(args, "llm_concurrency", 5) or 5
        if concurrency > 1:
            print(f"[INFO] Using {concurrency} concurrent LLM workers for competency generation")
            legacy_batches = [
                (bid, chunk)
                for bid, chunk in enumerate(
                    [
                        skills_sorted[i : i + args.max_skills_per_call]
                        for i in range(0, len(skills_sorted), args.max_skills_per_call)
                    ],
                    start=1,
                )
            ]

            def _process_legacy_batch(task: Tuple[int, List[str]]) -> Tuple[int, List[Dict]]:
                bid, chunk = task
                print(f"[INFO] Calling LLM for batch {bid} ({len(chunk)} skills)...")
                data = call_llm_for_competencies(
                    client,
                    chunk,
                    args.model,
                    future_context=future_context,
                    few_shot_examples=few_shot or [],
                    skill_future_weights=skill_future_weights or None,
                    skill_time_trends=skill_time_trends or None,
                    skill_bloom_map=skill_bloom_map or None,
                    curriculum_context=curriculum_context,
                    vocational_domain=vocational_domain or "",
                    top_soft_skills=top_soft_skills,
                    temperature=args.temperature,
                )
                comps = data.get("competencies", [])
                for c in comps:
                    c.setdefault("batch_id", bid)
                return (bid, comps)

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(_process_legacy_batch, t): t for t in legacy_batches}
                results_by_id = {}
                for future in as_completed(futures):
                    bid, comps = future.result()
                    results_by_id[bid] = comps
            for bid in sorted(results_by_id):
                all_competencies.extend(results_by_id[bid])
        else:
            for i in range(0, len(skills_sorted), args.max_skills_per_call):
                chunk = skills_sorted[i : i + args.max_skills_per_call]
                print(f"[INFO] Calling LLM for batch {batch_id} ({len(chunk)} skills)...")
                data = call_llm_for_competencies(
                    client,
                    chunk,
                    args.model,
                    future_context=future_context,
                    few_shot_examples=few_shot or [],
                    skill_future_weights=skill_future_weights or None,
                    skill_time_trends=skill_time_trends or None,
                    skill_bloom_map=skill_bloom_map or None,
                    curriculum_context=curriculum_context,
                    vocational_domain=vocational_domain or "",
                    top_soft_skills=top_soft_skills,
                    temperature=args.temperature,
                )
                comps = data.get("competencies", [])
                for c in comps:
                    c.setdefault("batch_id", batch_id)
                all_competencies.extend(comps)
                batch_id += 1

    # Tag competencies with human verification status (comprehensive mode)
    if args.comprehensive and human_verified_skills:
        for c in all_competencies:
            related = c.get("related_skills") or []
            if not related:
                c["all_skills_human_verified"] = False
            else:
                related_lower = [str(s).strip().lower() for s in related]
                c["all_skills_human_verified"] = all(
                    r in human_verified_skills for r in related_lower
                )
    elif args.comprehensive:
        for c in all_competencies:
            c["all_skills_human_verified"] = False  # no human feedback loaded

    print(f"[INFO] Total competencies generated (before dedup): {len(all_competencies)}")

    # Deduplicate similar competencies; aggregate occurrence_count
    if args.deduplicate and all_competencies:
        thresh = args.merge_overlap_threshold
        all_competencies = _deduplicate_competencies(
            all_competencies,
            merge_overlap_threshold=thresh,
            semantic_embedder=embedder,
        )
        print(f"[INFO] Competencies after grouping: {len(all_competencies)}")
    else:
        for c in all_competencies:
            c.setdefault("occurrence_count", 1)

    # Build output with metadata for reproducibility and comprehensive mode
    output_obj = {"competencies": all_competencies, "llm_temperature": args.temperature}
    if args.comprehensive:
        verified_count = sum(1 for c in all_competencies if c.get("all_skills_human_verified"))
        output_obj["generation_mode"] = "comprehensive"
        output_obj["note"] = (
            "Some competencies in this file are generated from skills not yet human-verified. "
            "Check the 'all_skills_human_verified' field per competency. "
            f"{verified_count}/{len(all_competencies)} competencies have all related skills human-verified."
        )

    output_path.write_text(
        json.dumps(output_obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved competency proposals to {output_path}")


if __name__ == "__main__":
    main()
