"""
dashboard_v2/app.py

Phase 2.6 — minimal public UI for the v2 competency pipeline.

Streamlit single-page app that:
  * lists all competencies from a v2 run directory (competencies.json)
  * supports filter by KKNI level, future_weight, grounding score, and search
  * shows full provenance chain (source sentences + jobs) per competency
  * exposes the per-competency rationale with a "Read more" toggle
  * exposes the cluster's batch_reasoning with a "View reasoning" toggle
  * shows education-level demand distribution (Phase 2.4)

Run:
    streamlit run dashboard_v2/app.py -- --run-dir results/competency_v2_live_gpt54mini

Or, from the project root:
    streamlit run dashboard_v2/app.py

Environment variables (optional override):
    COMPETENCY_V2_RUN_DIR  — directory with competencies.json + batch_reasonings.json + clusters.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Add project root to sys.path so kkni etc. import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# CLI args
# --------------------------------------------------------------------------- #


def _resolve_run_dir() -> Path:
    # 1. CLI flag --run-dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    # streamlit may pass other args we don't care about
    args, _ = parser.parse_known_args(sys.argv[1:])

    rd = args.run_dir or os.environ.get("COMPETENCY_V2_RUN_DIR")
    if rd:
        return Path(rd).resolve()

    # 2. Auto-pick the most recent results/competency_v2_* directory that
    #    has a competencies.json file.
    candidates = []
    results_root = PROJECT_ROOT / "results"
    if results_root.exists():
        for d in results_root.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if not (name.startswith("competency_v2_pipeline_")
                    or name.startswith("competency_v2_live")):
                continue
            comp_file = d / "competencies.json"
            if comp_file.exists():
                candidates.append((d, comp_file.stat().st_mtime))
    # Prefer competency_v2_pipeline_* (full pipeline output) over competency_v2_live*
    # (single-stage / model-comparison runs) when timestamps are close — but
    # primarily order by recency. Take the newest competencies.json.
    candidates.sort(key=lambda t: t[1], reverse=True)
    if candidates:
        return candidates[0][0].resolve()
    return (PROJECT_ROOT / "results" / "competency_v2_pipeline_e2e_v2_real_sentences").resolve()


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner=False)
def load_run(run_dir: str) -> dict:
    """Load competencies + batch_reasonings + clusters from a run directory."""
    rd = Path(run_dir)
    out = {"run_dir": str(rd), "competencies": [], "batch_reasonings": [], "clusters": []}
    cp = rd / "competencies.json"
    if cp.exists():
        out["competencies"] = json.loads(cp.read_text(encoding="utf-8"))
    bp = rd / "batch_reasonings.json"
    if bp.exists():
        out["batch_reasonings"] = json.loads(bp.read_text(encoding="utf-8"))
    cl = rd / "clusters.json"
    if cl.exists():
        out["clusters"] = json.loads(cl.read_text(encoding="utf-8"))
    return out


@st.cache_data(show_spinner=False)
def load_job_titles(run_dir: str) -> dict:
    """Load {job_id: title} from this run's jobs_metadata.csv (path is recorded
    in pipeline_report.json). Returns empty dict when not available — the
    role-shape panel then falls back to showing raw job_ids.
    """
    import csv as _csv
    rd = Path(run_dir)
    pr = rd / "pipeline_report.json"
    if not pr.exists():
        return {}
    try:
        report = json.loads(pr.read_text(encoding="utf-8"))
    except Exception:
        return {}
    jm_path = report.get("jobs_metadata") or ""
    if not jm_path:
        return {}
    p = Path(jm_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    out: dict = {}
    try:
        with open(p, encoding="utf-8-sig") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                jid = (row.get("job_id") or "").strip()
                title = (row.get("title") or "").strip()
                if jid and title:
                    out[jid] = title
    except Exception:
        return {}
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


# Role-normalisation: hoisted to `role_normalization` module (Phase 2.A of v8
# sprint) so the Phase 2.2 generator can share the same buckets the dashboard
# uses. See role_normalization.py for the table.
from role_normalization import normalize_role as _normalize_role  # noqa: F401


def _stage_chip(stage: str, frac: float) -> str:
    return f"`{stage}` {frac*100:.0f}%"


def _kkni_color(level: Optional[int]) -> str:
    if level is None:
        return "#888888"
    palette = {
        3: "#1f77b4", 4: "#2ca02c", 5: "#ff7f0e", 6: "#d62728",
        7: "#9467bd", 8: "#8c564b", 9: "#e377c2",
    }
    return palette.get(int(level), "#888888")


# Brief cognitive-complexity descriptor per KKNI level (v8 sprint Phase 3).
# Mirrors the Bloom-style anchors in the generator's KKNI_BRIEF — kept short
# enough to fit in a metadata-cell caption. KKNI = cognitive complexity of
# the work itself, independent of education-stage demand.
_KKNI_LEVEL_DESCRIPTORS = {
    3: "operational; follows procedures",
    4: "applies standard methods; minor diagnostic decisions",
    5: "analyses + selects methods; supervises routine",
    6: "designs novel-within-standard; leads small teams",
    7: "complex novel problems; multidisciplinary; leadership",
    8: "research; original contribution",
    9: "research; original contribution",
}


def _kkni_descriptor(level: Optional[int]) -> str:
    if level is None:
        return ""
    try:
        return _KKNI_LEVEL_DESCRIPTORS.get(int(level), "")
    except (TypeError, ValueError):
        return ""


# v9 sprint: target-stage-fit support. Each stage maps to a KKNI range; the
# fit chip is based on the delta between the competency's KKNI and the
# stage's max KKNI level. KKNI and education-stage demand stay independent
# labels — this is a UX hint for curriculum designers, not a gate.
_TARGET_STAGE_KKNI_RANGE = {
    "SMK":      (2, 3),
    "D3":       (4, 5),
    "D4 / S1":  (6, 6),
    "S2":       (7, 8),
}


def _target_stage_fit(target_stage: str, kkni_level: Optional[int]) -> Optional[dict]:
    """Return a fit-chip descriptor for this competency given the user's
    selected target stage. Returns None when target_stage is 'Any' / kkni
    is missing."""
    if not target_stage or target_stage == "Any" or kkni_level is None:
        return None
    rng = _TARGET_STAGE_KKNI_RANGE.get(target_stage)
    if not rng:
        return None
    lo, hi = rng
    try:
        k = int(kkni_level)
    except (TypeError, ValueError):
        return None
    if lo <= k <= hi:
        return {"severity": "match", "delta": 0, "msg": f"Matches your {target_stage} target stage"}
    if k == hi + 1:
        return {"severity": "info", "delta": 1, "msg": f"Slightly above your {target_stage} target stage"}
    if k > hi + 1:
        return {"severity": "warning", "delta": k - hi,
                "msg": f"{k - hi} levels above your {target_stage} target stage — consider whether students can reach this complexity"}
    if k < lo:
        return {"severity": "below", "delta": k - lo,
                "msg": f"Below your {target_stage} target stage — may already be covered at a lower level"}
    return None


def _stages_for_kkni(level: Optional[int]) -> list:
    """Map a KKNI level back to the friendly Indonesian education stages it
    represents. Uses the canonical `kkni.STAGE_TO_KKNI` mapping reversed.
    e.g., level 5 -> ['D3'], level 6 -> ['D4', 'S1'], level 3 -> ['SMK'].
    """
    if level is None:
        return []
    try:
        from kkni import STAGE_TO_KKNI
    except Exception:
        return []
    out = []
    for stage, levels in STAGE_TO_KKNI.items():
        if int(level) in levels:
            out.append(stage)
    # Prefer the SMK-D3-D4-S1 ordering that vocational stakeholders use
    preferred_order = ["SMK", "D1", "D2", "D3", "D4", "S1", "Profesi", "S2", "S3", "SMA"]
    out.sort(key=lambda s: preferred_order.index(s) if s in preferred_order else 99)
    return out


def _grounding_color(score: float) -> str:
    if score >= 0.95: return "#1a9850"  # green
    if score >= 0.80: return "#91cf60"  # lighter green
    if score >= 0.60: return "#fc8d59"  # orange
    return "#d73027"                     # red


def _bar(fraction: float, color: str, label: str = "") -> str:
    pct = max(0, min(100, int(round(fraction * 100))))
    return f"""
<div style='background:#eee; border-radius:4px; height:8px; overflow:hidden; margin-top:2px;'>
  <div style='background:{color}; width:{pct}%; height:100%'></div>
</div>
<div style='font-size:11px; color:#666;'>{label}</div>
"""


# --------------------------------------------------------------------------- #
# Detail panel
# --------------------------------------------------------------------------- #


def _localize(comp: dict, field: str, use_indonesian: bool) -> str:
    """Return the Indonesian translation of a competency field when
    available + requested; otherwise the English original. Used by every
    rendered string in the detail view (v9 sprint language toggle)."""
    if use_indonesian:
        v = comp.get(f"{field}_id")
        if isinstance(v, str) and v:
            return v
    return comp.get(field, "") or ""


def _localize_list(comp: dict, field: str, use_indonesian: bool) -> list:
    """Same as `_localize` but for list-valued fields (related_skills,
    soft_skills_required). Returns the English original when Indonesian
    translations are missing or partial."""
    if use_indonesian:
        v = comp.get(f"{field}_id")
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    return comp.get(field, []) or []


def _render_detail(
    comp: dict,
    br_lookup: dict,
    cluster_lookup: dict,
    job_titles: dict = None,
    target_stage: str = "Any",
    use_indonesian: bool = False,
):
    title = _localize(comp, "title", use_indonesian) or "(no title)"
    st.markdown(f"## {title}")

    # Metadata row 1: Competency complexity / Labour-market demand /
    # Future weight / Source jobs. Splitting KKNI complexity from
    # education-stage demand is the v8 sprint Phase 3 methodological fix —
    # the two are INDEPENDENT labels and must not be presented as one
    # compound chip.
    cols = st.columns(4)
    kkni = comp.get("kkni_level")
    kkni_descr = _kkni_descriptor(kkni)
    cols[0].markdown(
        f"**Competency complexity**<br>"
        f"<span style='color:{_kkni_color(kkni)}; font-size:24px; font-weight:bold'>KKNI {kkni or '?'}</span>"
        f"<br><span style='font-size:11px; color:#666'>{kkni_descr or '—'}</span>"
        f"<br><span style='font-size:10px; color:#888'>source: {comp.get('kkni_level_source','?')}</span>",
        unsafe_allow_html=True,
    )
    # Labour-market demand: surface the dominant education stage(s) from
    # Phase 2.4 right next to the KKNI complexity cell so the user can see
    # both facts side-by-side. INDEPENDENT of KKNI.
    edu = comp.get("education_levels_demanded") or {}
    if edu:
        top_two = sorted(edu.items(), key=lambda x: -x[1])[:2]
        demand_line = " · ".join(f"{stage} {frac*100:.0f}%" for stage, frac in top_two)
        n_disclosed = comp.get("n_jobs_with_education", 0)
        n_total = len(comp.get("source_job_ids", []))
        demand_caption = f"from {n_disclosed}/{n_total} postings disclosing"
    else:
        demand_line = "—"
        demand_caption = "no education data"
    cols[1].markdown(
        f"**Labour-market demand**<br>"
        f"<span style='font-size:18px; font-weight:bold'>{demand_line}</span>"
        f"<br><span style='font-size:11px; color:#666'>{demand_caption}</span>"
        f"<br><span style='font-size:10px; color:#888'>independent of KKNI</span>",
        unsafe_allow_html=True,
    )
    fg = comp.get("future_weight", 0.0)
    cols[2].markdown(
        f"**Future weight**<br>"
        f"<span style='font-size:24px; font-weight:bold'>{fg:.2f}</span>"
        f"<br><span style='font-size:11px; color:#666'>{comp.get('empirical_trend','?')}</span>",
        unsafe_allow_html=True,
    )
    nj = len(comp.get("source_job_ids", []))
    demand = float(comp.get("demand_score") or 0.0)
    # v8 sprint Phase 4: surface demand as a relative chip (top X% of corpus).
    # demand_score is normalised against the run's max n_unique_jobs so 1.00
    # = the most-demanded competency in this run.
    if demand > 0:
        top_pct = max(1, int(round((1 - demand) * 100)))
        demand_caption = f"top {top_pct}% by demand"
    else:
        demand_caption = f"{len(comp.get('source_sentences',[]))} unique sentences"
    cols[3].markdown(
        f"**Source jobs**<br>"
        f"<span style='font-size:24px; font-weight:bold'>{nj}</span>"
        f"<br><span style='font-size:11px; color:#666'>{demand_caption}</span>",
        unsafe_allow_html=True,
    )

    # Grounding (now on its own line below the metadata row so we have room
    # for the skill/sentence axis breakdown without crowding the layout).
    gs = comp.get("grounding_score", comp.get("grounding_score_preview", 0.0))
    skill_g = comp.get("skill_grounding_score")
    sent_r = comp.get("sentence_relevance_score")
    if skill_g is not None and sent_r is not None and (skill_g != gs or sent_r != gs):
        breakdown = f"skill {float(skill_g):.2f} · sentence {float(sent_r):.2f}"
    else:
        breakdown = f"method: {comp.get('grounding_method','?')}"
    st.markdown(
        f"<span style='font-size:14px; color:#444'>**Grounding**: "
        f"<span style='color:{_grounding_color(gs)}; font-weight:bold'>{gs:.2f}</span> "
        f"<span style='font-size:11px; color:#666'>· {breakdown}</span></span>",
        unsafe_allow_html=True,
    )

    # Caption explaining the independence of KKNI complexity vs labour-market
    # demand. Shown once per detail page, right under the metadata row.
    st.caption(
        "ℹ️ **Competency complexity (KKNI)** describes the cognitive demand of "
        "the work itself. **Labour-market demand** describes the education stage "
        "employers currently request. These are independent labels — a D3 "
        "institution can teach S2 competencies; a KKNI-7 competency can be in "
        "demand at SMK or D3. They do not influence each other."
    )

    # v9 sprint: target-stage fit chip. Shown only when user picked a stage
    # in the sidebar. Doesn't gate anything — pure UX hint.
    fit = _target_stage_fit(target_stage, comp.get("kkni_level"))
    if fit:
        sev = fit["severity"]
        if sev == "warning":
            st.warning(f"⚠️ {fit['msg']}")
        elif sev == "info":
            st.info(f"ℹ️ {fit['msg']}")
        elif sev == "match":
            st.success(f"✅ {fit['msg']}")
        else:  # below
            st.caption(f"⬇ {fit['msg']}")

    st.markdown(f"**Description:** {_localize(comp, 'description', use_indonesian)}")

    # v9.10: SKKNI occupation matches. Each chip = one Indonesian
    # occupational standard the competency maps to (top-K, cosine >=
    # threshold). An empty list means no SKKNI ICT match — surfaced as a
    # caption so users can see the honesty.
    occ_matches = comp.get("occupation_matches") or []
    if occ_matches:
        st.markdown("### Maps to (Indonesian occupational standards)")
        chips = []
        any_community = False
        for m in occ_matches:
            name = m.get("occupation_en", "?")
            score = m.get("cosine", 0.0)
            # Mark community-defined entries with a dagger (†) so the chip
            # carries its provenance even before the user expands details.
            marker = "" if m.get("is_official", True) else " †"
            if not m.get("is_official", True):
                any_community = True
            chips.append(f"`{name}{marker}` ({score:.2f})")
        st.markdown(" · ".join(chips))
        if any_community:
            st.caption(
                "† = Community-defined occupation (not an official "
                "Kepmenaker SKKNI). Added to cover gaps in the current ICT "
                "catalogue — see file `DATA/skema_okupasi_community_additions.csv` "
                "for the full list."
            )
        with st.expander("Why these occupations? ▾"):
            for m in occ_matches:
                official_tag = "" if m.get("is_official", True) else " *(community-defined, non-official)*"
                st.markdown(
                    f"**{m.get('occupation_en','?')}**{official_tag} — "
                    f"_{m.get('occupation_id','')}_  "
                    f"(cosine {m.get('cosine', 0.0):.3f})"
                )
                desc = m.get("description_en", "")
                if desc:
                    st.caption(desc)
    else:
        st.caption(
            "ℹ️ No SKKNI ICT occupation match above the threshold — this "
            "competency may not anchor to a current Indonesian occupational "
            "standard (could be cross-occupational, leadership-level, or a "
            "skill the SKKNI catalogue doesn't yet cover, e.g. cloud "
            "architect, ML engineer, frontend specialist)."
        )

    # Title evidence concerns (Phase 2.1 Tier 2D). Flags brand/proper-noun
    # tokens in the title that don't appear in enough source sentences —
    # i.e. the title is overreaching what the cluster's evidence supports.
    concerns = comp.get("title_evidence_concerns") or []
    if concerns:
        warnings = [c for c in concerns if c.get("severity") == "warning"]
        infos = [c for c in concerns if c.get("severity") == "info"]
        if warnings:
            for c in warnings:
                st.warning(
                    f"⚠️ Title mentions **{c['term']}** but it appears in "
                    f"only **{c['literal_hits']} of {c['n_sentences']}** source sentences "
                    f"(threshold {c['min_required']}). Title may overreach evidence."
                )
        if infos:
            terms = ", ".join(f"`{c['term']}` ({c['literal_hits']}/{c['n_sentences']})" for c in infos)
            st.caption(f"ℹ️ Limited evidence for: {terms}")

    # Related skills
    st.markdown("### Related skills")
    rs = _localize_list(comp, "related_skills", use_indonesian)
    if rs:
        st.markdown("\n".join(f"- {s}" for s in rs))
    else:
        st.write("_no related skills_")

    # Soft skills
    soft = _localize_list(comp, "soft_skills_required", use_indonesian)
    if soft:
        st.markdown("### Soft skills required")
        st.markdown(", ".join(f"`{s}`" for s in soft))
        sd = _localize(comp, "soft_skills_description", use_indonesian)
        if sd:
            st.caption(sd)

    # Education-level demand (Phase 2.4)
    edu = comp.get("education_levels_demanded") or {}
    if edu:
        st.markdown("### Education-level demand")
        st.caption(
            f"From {comp.get('n_jobs_with_education', 0)} of {len(comp.get('source_job_ids', []))} "
            f"source job postings that disclosed education requirements"
        )
        for stage, frac in sorted(edu.items(), key=lambda x: -x[1]):
            st.markdown(f"`{stage}` — {frac*100:.0f}%")
            st.markdown(_bar(frac, "#4477aa"), unsafe_allow_html=True)

    # Role-shape diversity (Phase 2.1 Tier 2). Shows the histogram of source
    # job_titles bucketed into broad role families — auditing whether the
    # competency draws from a narrow role band (e.g., only DevOps engineers)
    # or spans broadly. Job posting titles are unique per posting, so we
    # normalize into role buckets via _normalize_role.
    if job_titles:
        src_jobs = comp.get("source_job_ids") or []
        role_counts: dict = {}
        unknown = 0
        sample_titles_per_role: dict = {}
        for j in src_jobs:
            t = job_titles.get(j) or ""
            if not t:
                unknown += 1
                continue
            role = _normalize_role(t)
            role_counts[role] = role_counts.get(role, 0) + 1
            sample_titles_per_role.setdefault(role, []).append(t)
        if role_counts:
            st.markdown("### Source role distribution")
            n_resolved = sum(role_counts.values())
            uniq_roles = len(role_counts)
            st.caption(
                f"{n_resolved} of {len(src_jobs)} source postings resolved to a job title, "
                f"bucketed into {uniq_roles} role families"
                + (f" — {unknown} unmatched" if unknown else "")
            )
            shown = sorted(role_counts.items(), key=lambda x: -x[1])
            max_n = shown[0][1] if shown else 1
            for role, cnt in shown:
                frac_share = cnt / n_resolved if n_resolved else 0.0
                frac_bar = cnt / max_n if max_n else 0.0
                st.markdown(f"`{role}` — {cnt} ({frac_share*100:.0f}%)")
                st.markdown(_bar(frac_bar, "#aa7744"), unsafe_allow_html=True)
            # Show 2-3 example titles per top role to keep the panel auditable
            with st.expander("Example postings per role ▾"):
                for role, _ in shown[:5]:
                    examples = sample_titles_per_role.get(role, [])[:3]
                    if examples:
                        st.markdown(f"**{role}**")
                        for ex in examples:
                            st.caption(f"• {ex}")

    # Rationale (read-more toggle)
    rationale = _localize(comp, "rationale", use_indonesian)
    st.markdown("### Why this competency? (LLM rationale)")
    if rationale:
        teaser = rationale[:200] + ("…" if len(rationale) > 200 else "")
        st.caption(teaser)
        if len(rationale) > 200:
            with st.expander("Read more ▾"):
                st.write(rationale)
    else:
        st.write("_no rationale_")

    # Batch reasoning (public via toggle, per resolved Phase 2.2 design Q#2)
    br_id = comp.get("batch_reasoning_id") or ""
    br = br_lookup.get(br_id)
    if br:
        with st.expander("How the LLM grouped these skills (batch reasoning) ▾"):
            st.caption(
                f"Model: `{br.get('model','?')}` ({br.get('provider','?')}) — "
                f"latency {br.get('latency_seconds',0):.1f}s, "
                f"competencies generated from this cluster: {br.get('n_competencies_out','?')}"
            )
            st.write(br.get("batch_reasoning") or "_(no batch_reasoning recorded)_")

    # Cluster context. Streamlit forbids nested expanders, so the inner
    # "show items" is gated by a checkbox instead.
    cid = comp.get("cluster_id") or ""
    cl = cluster_lookup.get(cid)
    if cl:
        with st.expander("Source cluster details ▾"):
            st.caption(
                f"Cluster `{cid}` (stream: {cl.get('stream','?')}, method: {cl.get('method','?')}) — "
                f"{cl.get('n_items','?')} items, cohesion {cl.get('cohesion_score','?'):.3f}"
                if isinstance(cl.get('cohesion_score'), (int, float))
                else f"Cluster `{cid}`"
            )
            st.write(f"**Heuristic label:** {cl.get('summary_label','?')}")
            if cl.get("top_terms"):
                st.write(f"**Top terms:** {', '.join(cl['top_terms'])}")
            if cl.get("items"):
                # Checkbox key includes cid + comp id so multiple competencies
                # from the same cluster get independent toggles.
                show_items = st.checkbox(
                    f"Show all {len(cl['items'])} cluster items",
                    key=f"show_items_{comp.get('id','')}_{cid}",
                )
                if show_items:
                    for it in cl["items"][:50]:
                        st.markdown(f"- {it}")
                    if len(cl["items"]) > 50:
                        st.caption(f"... and {len(cl['items']) - 50} more")

    # Provenance chain
    sents = comp.get("source_sentences") or []
    sids = comp.get("source_sentence_ids") or []
    job_ids = comp.get("source_job_ids") or []
    with st.expander(f"Provenance — {nj} jobs / {len(sents)} sentences ▾"):
        # If every "sentence" matches one of the related_skills verbatim, warn
        # the user that we're showing skill texts as a stand-in (happens with
        # legacy data that didn't preserve full sentence_text).
        related = {(s or "").strip().lower() for s in (comp.get("related_skills") or [])}
        skill_proxies = sum(
            1 for s in sents if (s or "").strip().lower() in related
        )
        if sents and skill_proxies == len(sents):
            st.caption(
                "ℹ️ This competency's source data didn't preserve full sentence "
                "text — showing the extracted skill phrases as a stand-in. Re-run "
                "the pipeline on Phase 1 output that includes `sentence_text` "
                "(e.g., from the offline Skill-LLM batch) for verbatim provenance."
            )
        # Render
        for i, s in enumerate(sents, 1):
            st.markdown(f"{i}. _{s}_")
        # And the unique job-ID list for full traceability
        if job_ids:
            st.caption(f"Source jobs: " + ", ".join(f"`{j}`" for j in job_ids[:25]) +
                       (f" + {len(job_ids) - 25} more" if len(job_ids) > 25 else ""))

    # --- User ratings (Phase 2.6 sub-deliverable) ---
    _render_rating_block(comp)


def _render_rating_block(comp: dict):
    """Light-signup-gated rating widget + public aggregate display.

    Per locked design:
      - rating does NOT affect priority_score / ranking (research signal only)
      - anonymous browse is free; rating requires light signup
      - one rating per (user, competency); overwrites on re-submit
    """
    from user_ratings import (
        VALID_ROLES,
        find_or_create_user,
        get_aggregate,
        get_user_rating,
        submit_rating,
    )

    st.markdown("### User ratings")

    comp_id = comp.get("id", "")
    agg = get_aggregate(comp_id)

    if agg.rating_count == 0:
        st.caption("No ratings yet. Be the first — your feedback helps us improve future versions.")
    else:
        cols = st.columns(3)
        cols[0].metric(
            "Average",
            f"{agg.rating_mean:.1f} / 5",
            f"σ {agg.rating_std:.2f}" if agg.rating_count > 1 else None,
        )
        cols[1].metric("Ratings", str(agg.rating_count))
        # Distribution
        dist_str = " · ".join(
            f"{k}★:{agg.distribution[k]}" for k in range(5, 0, -1)
        )
        cols[2].caption(dist_str)

        if agg.by_role:
            role_summary = " | ".join(
                f"{role}: {info['mean']:.1f} (n={info['count']})"
                for role, info in sorted(agg.by_role.items())
            )
            st.caption(f"By role: {role_summary}")

    # --- Signup / rating widget ---
    if "rating_user" not in st.session_state:
        st.session_state["rating_user"] = None

    if st.session_state["rating_user"] is None:
        with st.expander("Sign in to rate (one-time, takes 10 seconds) ▾"):
            with st.form(key=f"signup_form_{comp_id}"):
                email = st.text_input("Email", placeholder="you@example.com")
                role = st.selectbox("Your role", options=list(VALID_ROLES))
                signup_ok = st.form_submit_button("Sign in")
            if signup_ok and email:
                try:
                    user = find_or_create_user(email, role)
                    st.session_state["rating_user"] = user
                    st.rerun()
                except Exception as e:
                    st.error(f"Sign-in failed: {e}")
    else:
        user = st.session_state["rating_user"]
        existing = get_user_rating(user["user_id"], comp_id)
        default_rating = int(existing["rating"]) if existing else 4
        default_text = existing["feedback_text"] if existing else ""

        with st.form(key=f"rate_form_{comp_id}"):
            st.caption(f"Signed in as `{user['role']}`. Ratings are aggregated; identity is never shown publicly.")
            r = st.slider(
                "Your rating", min_value=1, max_value=5,
                value=default_rating, step=1,
            )
            txt = st.text_area(
                "Optional feedback (≤ 2000 chars)",
                value=default_text, max_chars=2000, height=80,
                placeholder="Anything you'd like the curriculum team to know about this competency?",
            )
            cols = st.columns([1, 1, 3])
            submit = cols[0].form_submit_button("Submit rating")
            sign_out = cols[1].form_submit_button("Sign out")
        if submit:
            try:
                submit_rating(
                    user_id=user["user_id"],
                    role=user["role"],
                    competency_id=comp_id,
                    rating=int(r),
                    feedback_text=txt,
                    pipeline_run_tag=comp.get("generator_version", ""),
                )
                st.success(
                    "Thanks! Your rating is recorded. "
                    "It will not change the live ranking but helps us improve future versions."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not save rating: {e}")
        if sign_out:
            st.session_state["rating_user"] = None
            st.rerun()


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #


def _render_pipeline_audit_page(run_dir: Path):
    """Pipeline-audit page (v8.1) — surfaces refinement moves and the full
    noise audit so curriculum designers can see which items were extracted
    but didn't reach a competency, and why.
    """
    st.markdown("# Pipeline audit")
    st.caption(
        "Transparency view: every item the extractor surfaced that did NOT "
        "reach a competency, plus every refinement move the LLM arbiter made. "
        "Helps spot extractor noise, generic-skill pollution, and "
        "cluster-membership decisions you might want to second-guess."
    )

    # Load the three audit sources
    pr_path = run_dir / "pipeline_report.json"
    na_path = run_dir / "noise_audit.json"
    it_path = run_dir / "it_relevance_audit.json"  # v9 sprint
    pipeline_report: dict = {}
    noise_audit: dict = {}
    it_audit: dict = {}
    try:
        if pr_path.exists():
            pipeline_report = json.loads(pr_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load pipeline_report.json: {e}")
    try:
        if na_path.exists():
            noise_audit = json.loads(na_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load noise_audit.json: {e}")
    try:
        if it_path.exists():
            it_audit = json.loads(it_path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"Could not load it_relevance_audit.json: {e}")

    # ---- IT-relevance gate section (v9 sprint) ----
    if it_audit:
        st.markdown("## IT-relevance gate")
        s = it_audit.get("summary", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Items in", s.get("n_items_in", 0))
        c2.metric("Items kept", s.get("n_items_kept", 0),
                  delta=f"−{s.get('n_items_in',0) - s.get('n_items_kept',0)}")
        c3.metric("Jobs IT / total",
                  f"{s.get('n_jobs_it', 0)} / {s.get('n_jobs_total', 0)}")
        c4.metric("Sentences IT / total",
                  f"{s.get('n_sentences_it', 0)} / {s.get('n_sentences_total', 0)}")

        dropped_jobs = it_audit.get("dropped_jobs", []) or []
        if dropped_jobs:
            with st.expander(f"Dropped jobs ({len(dropped_jobs)}) — non-IT postings"):
                for dj in dropped_jobs[:200]:
                    st.markdown(f"`{dj.get('job_id','?')}` — dropped **{dj.get('n_items_dropped',0)}** items")
                    excerpt = dj.get("description_excerpt", "")
                    if excerpt:
                        st.caption(excerpt)
                if len(dropped_jobs) > 200:
                    st.caption(f"... and {len(dropped_jobs) - 200} more")

        dropped_sents = it_audit.get("dropped_sentences", []) or []
        if dropped_sents:
            with st.expander(f"Dropped sentences ({len(dropped_sents)}) — boilerplate / non-IT within IT postings"):
                q = st.text_input("Search dropped sentences", "", key="it_dropped_search")
                ql = q.lower().strip()
                shown = [s_ for s_ in dropped_sents if not ql or ql in (s_.get("sentence_text","") or "").lower()]
                st.caption(f"Showing **{len(shown)}** of {len(dropped_sents)}.")
                for ds in shown[:200]:
                    st.markdown(f"- `{ds.get('sentence_id','?')}` (×{ds.get('n_items_dropped',0)}): _{ds.get('sentence_text','')}_")
                if len(shown) > 200:
                    st.caption(f"... and {len(shown) - 200} more (showing first 200)")

        st.markdown("---")

    # ---- Refinement moves section ----
    st.markdown("## Cluster refinement (LLM-as-arbiter)")
    ref = (pipeline_report.get("stages", {}).get("phase2_1_clustering") or {}).get("refinement")
    if not ref:
        st.info(
            "No refinement audit found for this run. Either refinement was "
            "disabled (default) or this is a legacy run from before v8."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clusters processed", ref.get("n_clusters_processed", 0))
        c2.metric("Items moved OUT", ref.get("total_items_moved_out", 0))
        c3.metric("Items moved IN",  ref.get("total_items_moved_in", 0))
        c4.metric("LLM calls", ref.get("total_llm_calls", 0))
        c5, c6 = st.columns(2)
        c5.metric(
            "Mean cohesion before",
            f"{ref.get('cohesion_mean_before', 0):.3f}",
        )
        c6.metric(
            "Mean cohesion after",
            f"{ref.get('cohesion_mean_after', 0):.3f}",
            delta=f"{ref.get('cohesion_mean_after', 0) - ref.get('cohesion_mean_before', 0):+.3f}",
        )

        changed = [pc for pc in ref.get("per_cluster", []) if (pc.get("n_items_moved_out", 0) + pc.get("n_items_moved_in", 0)) > 0]
        if changed:
            st.markdown(f"### Clusters with moves ({len(changed)} of {ref.get('n_clusters_processed', 0)})")
            for pc in changed:
                with st.expander(
                    f"cluster_idx={pc['cluster_idx']} · out={pc['n_items_moved_out']} · in={pc['n_items_moved_in']} · "
                    f"cohesion {pc['initial_cohesion']:.3f} → {pc['final_cohesion']:.3f}"
                ):
                    for m in pc.get("move_decisions", []):
                        kind = m.get("kind", "?")
                        verdict = m.get("verdict", "?")
                        text = m.get("text", "")
                        reason = m.get("reason", "")
                        icon = "⬅" if verdict == "move_out" else "➡" if verdict == "join" else "·"
                        st.markdown(f"{icon} **{verdict}** `{text}` — _{reason}_")
        else:
            st.caption("No clusters were modified — every item stayed where HDBSCAN placed it.")

    st.markdown("---")

    # ---- Noise audit section ----
    st.markdown("## Items that did NOT reach a competency")
    if not noise_audit:
        st.info(
            "No noise_audit.json found for this run. v8.1+ runs include it; "
            "legacy runs do not."
        )
        return

    n_total = noise_audit.get("n_items_dropped_total", 0)
    by_stage: dict = noise_audit.get("by_stage", {}) or {}
    st.caption(
        f"**{n_total} items** dropped in total across all Phase 2.1 stages. "
        "Each item below has the canonical text + the stage at which it was "
        "dropped + the reason."
    )

    # Stage summary chart (text bars)
    if by_stage:
        max_n = max(by_stage.values()) or 1
        st.markdown("### Drops by stage")
        for stage, n in sorted(by_stage.items(), key=lambda x: -x[1]):
            frac = n / max_n
            st.markdown(f"`{stage}` — **{n}**")
            st.markdown(_bar(frac, "#888"), unsafe_allow_html=True)

    # Per-stage filter + searchable item list
    items = noise_audit.get("items", []) or []
    if not items:
        return
    st.markdown("### Browse dropped items")
    stages = sorted(set(it.get("stage", "") for it in items))
    chosen_stages = st.multiselect(
        "Filter by stage", options=stages, default=stages,
    )
    q = st.text_input("Search by item text", "")
    q_lower = q.lower().strip()

    filtered = [
        it for it in items
        if it.get("stage") in chosen_stages
        and (not q_lower or q_lower in (it.get("text", "") or "").lower())
    ]
    st.caption(f"Showing **{len(filtered)} of {len(items)}** items.")

    # Group by stage, show items in expanders to avoid one huge wall of text
    by_stage_items: dict = {}
    for it in filtered:
        by_stage_items.setdefault(it.get("stage", ""), []).append(it)
    for stage in sorted(by_stage_items.keys()):
        stage_items = by_stage_items[stage]
        with st.expander(f"`{stage}` — {len(stage_items)} items"):
            for it in stage_items[:200]:  # cap per stage to keep page responsive
                txt = it.get("text", "")
                reason = it.get("reason", "")
                njobs = it.get("n_unique_jobs", 0)
                extra = it.get("extra") or {}
                meta_bits = []
                if njobs:
                    meta_bits.append(f"in {njobs} jobs")
                if extra:
                    meta_bits.extend(f"{k}={v}" for k, v in list(extra.items())[:3])
                meta = " · ".join(meta_bits)
                st.markdown(f"- **{txt}** — _{reason}_" + (f" ({meta})" if meta else ""))
            if len(stage_items) > 200:
                st.caption(f"... and {len(stage_items) - 200} more (showing first 200)")


def _render_coverage_page(comps):
    """Curriculum upload + coverage report — Phase 2.6 sub-deliverable."""
    st.markdown("# Curriculum coverage analysis")
    st.caption(
        "Upload your existing curriculum (PDF or plain text). The system will "
        "match each competency against the curriculum content and show which "
        "ones are already covered, partially covered, or missing — with "
        "priority weighting by future-of-work demand."
    )

    upload = st.file_uploader(
        "Upload curriculum (PDF or .txt)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=False,
    )
    pasted = st.text_area(
        "...or paste curriculum text directly",
        height=200,
        placeholder="Paste the curriculum / syllabus / module list here. Each paragraph or numbered item is treated as one section.",
    )

    if not upload and not pasted.strip():
        st.info("Upload a curriculum or paste text to see the coverage analysis.")
        return

    from curriculum_coverage import (
        compute_coverage,
        parse_curriculum_pdf,
        parse_curriculum_text,
        translate_sections_to_english,
    )

    auto_translate = st.checkbox(
        "Auto-translate Indonesian sections to English (LLM, ~1 sec/section)",
        value=True,
        help="Detects each section's language and translates Indonesian ones to English "
             "before matching against the competency database. Skipped sections are kept as-is.",
    )

    with st.spinner("Parsing curriculum..."):
        if upload is not None:
            data_bytes = upload.read()
            if upload.name.lower().endswith(".pdf"):
                sections = parse_curriculum_pdf(data_bytes)
            else:
                sections = parse_curriculum_text(data_bytes.decode("utf-8", errors="replace"))
        else:
            sections = parse_curriculum_text(pasted)

    if not sections:
        st.error(
            "Couldn't extract any sections from this curriculum. Try plain text "
            "with numbered headings or paragraph breaks."
        )
        return

    if auto_translate:
        with st.spinner("Detecting language + translating Indonesian sections..."):
            sections, n_translated = translate_sections_to_english(sections)
        if n_translated:
            st.info(f"Translated {n_translated} Indonesian section(s) to English.")

    st.success(f"Parsed {len(sections)} curriculum sections.")
    with st.expander(f"Preview parsed sections ({len(sections)})"):
        for s in sections[:20]:
            st.markdown(f"**{s.title}** _(id: {s.section_id})_")
            st.caption(s.text[:300] + ("..." if len(s.text) > 300 else ""))
        if len(sections) > 20:
            st.caption(f"... and {len(sections) - 20} more sections")

    with st.spinner("Embedding and matching... (~10 s)"):
        annotations, report = compute_coverage(comps, sections)

    # --- Summary KPIs ---
    cols = st.columns(4)
    cols[0].metric("Well covered (≥ 0.65)", report.n_well_covered)
    cols[1].metric("Partially covered (0.45–0.65)", report.n_partially_covered)
    cols[2].metric("Missing (< 0.45)", report.n_missing)
    cols[3].metric("Weighted coverage", f"{report.weighted_mean_coverage:.2f}")

    # --- Top priority gaps (the "you should add" list) ---
    st.markdown("### Top priority-weighted gaps")
    st.caption(
        "Competencies that are missing or weakly covered AND have high future-of-work demand. "
        "Adding these has the highest impact on curriculum relevance."
    )
    annot_by_id = {a.competency_id: a for a in annotations}
    for a in report.top_gaps:
        c = next((c for c in comps if c.get("id") == a.competency_id), None)
        if c is None:
            continue
        label_color = {
            "missing": "#d73027",
            "partially_covered": "#fc8d59",
            "well_covered": "#1a9850",
        }.get(a.coverage_label, "#888")
        st.markdown(
            f"**{c['title']}** — "
            f"<span style='color:{label_color}'>{a.coverage_label}</span> "
            f"(coverage {a.coverage_score:.2f}, future weight {c.get('future_weight',0):.2f}, "
            f"priority-weighted gap {a.priority_weighted_gap:.2f})",
            unsafe_allow_html=True,
        )
        if a.best_matching_section_title:
            st.caption(f"Best (weak) match: _{a.best_matching_section_title}_")

    # --- Full coverage table ---
    st.markdown("### Full coverage breakdown")
    rows = []
    for c in comps:
        a = annot_by_id.get(c.get("id"))
        if a is None:
            continue
        rows.append({
            "Competency": c.get("title", ""),
            "KKNI": c.get("kkni_level"),
            "Future weight": round(float(c.get("future_weight", 0.0)), 2),
            "Coverage": round(a.coverage_score, 2),
            "Label": a.coverage_label,
            "Best match": a.best_matching_section_title or "—",
            "Priority gap": round(a.priority_weighted_gap, 2),
        })
    rows.sort(key=lambda r: -r["Priority gap"])
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(
        page_title="Competency Recommendations (v2)",
        page_icon="📚",
        layout="wide",
    )

    run_dir = _resolve_run_dir()
    data = load_run(str(run_dir))
    comps = data["competencies"]
    brs = {br["id"]: br for br in data["batch_reasonings"]}
    clusters = {c["id"]: c for c in data["clusters"]}
    # job_id -> title for the role-shape panel. Empty dict when this run's
    # jobs_metadata isn't reachable (older e2e runs); the panel hides itself.
    job_titles = load_job_titles(str(run_dir))

    # Top-level page switcher
    page = st.sidebar.radio(
        "Page",
        options=["Browse competencies", "Curriculum coverage", "Pipeline audit"],
        label_visibility="visible",
    )
    if page == "Pipeline audit":
        _render_pipeline_audit_page(run_dir)
        return
    if page == "Curriculum coverage":
        _render_coverage_page(comps)
        return

    # --------- Sidebar ---------
    # v9 sprint: language toggle. Switches between English originals and
    # Indonesian translations (when present in the run, populated by
    # `--translate-output-to-id`). Translations live in `{field}_id` keys.
    lang = st.sidebar.radio(
        "Language / Bahasa",
        options=["English", "Bahasa Indonesia"],
        index=0,
        horizontal=True,
    )
    has_id_translations = any(c.get("title_id") for c in comps[:5])
    if lang == "Bahasa Indonesia" and not has_id_translations:
        st.sidebar.warning(
            "This run wasn't built with `--translate-output-to-id`, so "
            "Indonesian translations aren't available. Showing English."
        )
        lang = "English"
    use_indonesian = (lang == "Bahasa Indonesia")

    st.sidebar.markdown("## Filters")
    st.sidebar.caption(f"Reading from: `{run_dir.name}`")
    st.sidebar.caption(f"{len(comps)} competencies, {len(brs)} LLM calls")

    # KKNI level filter
    all_levels = sorted({c.get("kkni_level") for c in comps if c.get("kkni_level")})
    chosen_levels = st.sidebar.multiselect(
        "KKNI level", options=all_levels, default=all_levels,
    )

    # v9 sprint: target curriculum stage. When the user picks a stage, every
    # competency gets a fit chip on its detail page (matches / above / below).
    # "Any" leaves the chips off — competency complexity stands alone.
    target_stage = st.sidebar.selectbox(
        "Target curriculum stage",
        options=["Any", "SMK", "D3", "D4 / S1", "S2"],
        index=0,
        help="When set, each competency gets a fit chip showing whether its "
             "cognitive complexity matches a curriculum at this stage. KKNI "
             "and education-stage demand remain independent labels — this is "
             "purely a UX hint.",
    )

    # v9.10: SKKNI occupation filter. The competencies' `occupation_matches`
    # contain top-K SKKNI occupations they map to; the filter narrows the
    # list to ones that map to any of the user's chosen occupations.
    all_occupations = sorted({
        m.get("occupation_en", "")
        for c in comps
        for m in (c.get("occupation_matches") or [])
        if m.get("occupation_en")
    })
    chosen_occupations = []
    if all_occupations:
        chosen_occupations = st.sidebar.multiselect(
            "Filter by SKKNI occupation",
            options=all_occupations,
            default=[],
            help="Show only competencies that map to one of these Indonesian "
                 "occupational standards (skema okupasi). When empty, all "
                 "competencies are shown.",
        )

    # Future-weight filter
    min_fw = st.sidebar.slider("Min future weight", 0.0, 1.0, 0.0, 0.05)
    # Grounding-score filter. Default 0.50 matches the canonical Phase-2.5 gate
    # after the 2026-05-13 honesty audit (was 0.80 under the old substring-grounding
    # regime, which inflated every score to 1.00 and made 0.80 meaningless).
    min_grounding = st.sidebar.slider("Min grounding score", 0.0, 1.0, 0.50, 0.05)

    # Search
    q = st.sidebar.text_input("Search title / description / skills", "")

    # Sort. Default is "priority (desc)" — the spec's combined demand × grounding ×
    # future_weight ranking (v8 sprint Phase 4).
    sort_by = st.sidebar.selectbox(
        "Sort by",
        options=[
            "priority (desc)",
            "future_weight (desc)",
            "grounding_score (desc)",
            "n source jobs (desc)",
            "title (a-z)",
        ],
        index=0,
    )

    # --------- Filter logic ---------
    def _match(c):
        if c.get("kkni_level") not in chosen_levels:
            return False
        if c.get("future_weight", 0.0) < min_fw:
            return False
        if c.get("grounding_score", c.get("grounding_score_preview", 0.0)) < min_grounding:
            return False
        if chosen_occupations:
            match_set = {
                m.get("occupation_en", "") for m in (c.get("occupation_matches") or [])
            }
            if not match_set.intersection(chosen_occupations):
                return False
        if q:
            blob = " ".join([
                c.get("title", ""), c.get("description", ""),
                " ".join(c.get("related_skills", [])), c.get("rationale", ""),
            ]).lower()
            if q.lower() not in blob:
                return False
        return True

    filtered = [c for c in comps if _match(c)]

    if sort_by == "priority (desc)":
        # v8 sprint Phase 4. Falls back to grounding+future when priority_score
        # is missing (legacy runs).
        def _priority_key(c):
            ps = c.get("priority_score")
            if ps is not None and ps > 0:
                return -float(ps)
            # Legacy fallback
            return -(0.30 * float(c.get("grounding_score", 0.0))
                     + 0.30 * float(c.get("future_weight", 0.0)))
        filtered.sort(key=_priority_key)
    elif sort_by == "future_weight (desc)":
        filtered.sort(key=lambda c: -float(c.get("future_weight", 0.0)))
    elif sort_by == "grounding_score (desc)":
        filtered.sort(key=lambda c: -float(c.get("grounding_score", c.get("grounding_score_preview", 0.0))))
    elif sort_by == "n source jobs (desc)":
        filtered.sort(key=lambda c: -len(c.get("source_job_ids", [])))
    else:
        filtered.sort(key=lambda c: c.get("title", "").lower())

    # --------- Header ---------
    st.markdown("# Competency Recommendations")
    st.caption(
        f"Cluster-driven curriculum competencies generated from real job-market data. "
        f"Showing **{len(filtered)} of {len(comps)}** competencies."
    )

    # Two-column layout: left = list, right = detail
    list_col, detail_col = st.columns([0.4, 0.6])

    with list_col:
        st.markdown("### Browse")
        if not filtered:
            st.info("No competencies match the current filters.")
            return
        # Use radio for selection — single-select list. Titles localised
        # via _localize so the browse panel matches the language toggle.
        labels = [
            f"{i+1}. {(_localize(c, 'title', use_indonesian) or '(untitled)')[:60]}  "
            f"(KKNI {c.get('kkni_level','?')}, grounding {c.get('grounding_score', c.get('grounding_score_preview', 0.0)):.2f})"
            for i, c in enumerate(filtered)
        ]
        selected = st.radio(
            "Pick a competency",
            options=range(len(filtered)),
            format_func=lambda i: labels[i],
            label_visibility="collapsed",
        )

    with detail_col:
        if selected is not None:
            _render_detail(
                filtered[selected], brs, clusters,
                job_titles=job_titles, target_stage=target_stage,
                use_indonesian=use_indonesian,
            )


if __name__ == "__main__":
    main()
