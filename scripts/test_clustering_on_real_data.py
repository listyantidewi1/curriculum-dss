"""
test_clustering_on_real_data.py

Phase 2.1 clustering, end-to-end on REAL Phase 1 output.

Reads two CSVs from a previous pipeline run:

    results.old2/advanced_skills.csv      (job_id, skill, type, confidence_score, source, ...)
    results.old2/advanced_knowledge.csv   (job_id, knowledge, confidence_score, source)

Converts each row into a SkillItem / KnowledgeItem (using the canonical
pipeline.py dataclasses), runs `clustering.cluster_skills(items)`, and dumps:

    results/clustering_smoke/clusters.csv         — per-cluster summary
    results/clustering_smoke/cluster_members.csv  — items per cluster
    results/clustering_smoke/report.json          — ClusteringReport
    results/clustering_smoke/cohesion_hist.png    — cohesion distribution
    results/clustering_smoke/summary.txt          — human-readable highlights

Use this to eyeball whether the clusters make sense before Phase 2.2 land.

Usage:
    python scripts/test_clustering_on_real_data.py
    python scripts/test_clustering_on_real_data.py --input-dir results.old2
    python scripts/test_clustering_on_real_data.py --cohesion-threshold 0.45 --min-cluster-size 4

Optional flags:
    --input-dir       directory containing advanced_skills.csv + advanced_knowledge.csv
    --output-dir      where to write outputs (default: results/clustering_smoke)
    --cohesion-threshold  override (default: 0.50)
    --min-cluster-size    override (default: 3)
    --min-global-freq     override (default: 2)
    --max-items           cap items for a quick sanity run (default: no cap)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Real-data loading
# --------------------------------------------------------------------------- #


def _build_synthetic_sentence_id(job_id: str, idx: int) -> str:
    """Phase 1 CSV doesn't carry sentence_id; synthesize one stable per (job, idx).

    Format mirrors the v2 sentence_id convention used elsewhere in the pipeline.
    """
    return f"{job_id}_{idx:04d}"


def _confidence_tier_from_score(score: float):
    from pipeline import ConfidenceTier

    if score >= 0.9:
        return ConfidenceTier.VERY_HIGH
    if score >= 0.8:
        return ConfidenceTier.HIGH
    if score >= 0.7:
        return ConfidenceTier.MEDIUM_HIGH
    if score >= 0.6:
        return ConfidenceTier.MEDIUM
    if score >= 0.5:
        return ConfidenceTier.MEDIUM_LOW
    if score >= 0.4:
        return ConfidenceTier.LOW
    return ConfidenceTier.VERY_LOW


_JOBS_RAW_CACHE: dict = {}
_JOBS_META_CACHE: dict = {}


def _load_jobs_metadata(jobs_metadata_csv: Path) -> dict:
    """Build {job_id -> {'title': ..., 'company': ..., ...}} lookup from
    jobs_metadata.csv. Used for Phase 2.1 Tier 2 role-context embedding.
    Cached per-path for the process lifetime.
    """
    key = str(jobs_metadata_csv) if jobs_metadata_csv else ""
    if key in _JOBS_META_CACHE:
        return _JOBS_META_CACHE[key]
    if not jobs_metadata_csv or not jobs_metadata_csv.exists():
        _JOBS_META_CACHE[key] = {}
        return {}
    import csv as _csv
    out: dict = {}
    with open(jobs_metadata_csv, encoding="utf-8-sig") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            jid = (row.get("job_id") or "").strip()
            if not jid:
                continue
            out[jid] = {
                "title": (row.get("title") or "").strip(),
                "company": (row.get("company") or "").strip(),
                "location": (row.get("location") or "").strip(),
                "min_education_kkni": (row.get("min_education_kkni") or "").strip(),
            }
    _JOBS_META_CACHE[key] = out
    return out


def _load_job_descriptions(raw_jobs_csv: Path) -> dict:
    """Build {job_id -> description} lookup once per process."""
    if str(raw_jobs_csv) in _JOBS_RAW_CACHE:
        return _JOBS_RAW_CACHE[str(raw_jobs_csv)]
    if not raw_jobs_csv.exists():
        _JOBS_RAW_CACHE[str(raw_jobs_csv)] = {}
        return {}
    import csv as _csv
    out: dict = {}
    with open(raw_jobs_csv, encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            jid = (row.get("id") or "").strip()
            desc = row.get("description") or ""
            if jid and desc:
                out[jid] = desc
    _JOBS_RAW_CACHE[str(raw_jobs_csv)] = out
    return out


def _split_into_sentences(text: str) -> list:
    """Bullet-aware sentence splitter.

    Improves over the base preprocess splitter by ALSO breaking on mid-paragraph
    bullets (`* item`, `- item`, `• item`) and on `Field:` markers, which job
    descriptions use heavily for "Key Responsibilities" / "Required Skills" lists.
    This prevents the 30-line-blob problem where everything after "Key
    Responsibilities" becomes one giant "sentence".
    """
    import re as _re
    if not isinstance(text, str):
        return []
    # 1. Break on bullets (anywhere, not just after newline) + paragraph breaks
    parts = _re.split(
        r"\n\s*[\-\*•·]\s+|\s+[\*•·]\s+|\n\n+",
        text,
    )
    out = []
    for part in parts:
        part = part.replace("\r", " ").replace("\n", " ")
        # 2. Also break on section markers like "Key Responsibilities * ..." or "Required Skills:"
        sub_parts = _re.split(
            r"\s+(?=(?:Key Responsibilities|Required Skills|Nice to Have|Responsibilities|Qualifications|Preferred Skills|About You|About Us|The Role|The Team|Your Role|We offer)\b\W*)",
            part,
        )
        for sp in sub_parts:
            sp = sp.strip()
            if not sp:
                continue
            # 3. Normal sentence-end split
            sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", sp) if s.strip()]
            out.extend(sents)
    # 4. Final length filter: drop very short fragments (< 20 chars) — they're usually splitter artifacts
    out = [s for s in out if len(s) >= 20]
    return out


def _best_matching_sentence(skill_text: str, sentences: list) -> str:
    """Pick the first sentence that contains the skill text (case-insensitive).

    Falls back to the longest sentence if no literal match (rare for our
    extracted skills since the LLM mostly preserves verbatim phrases).
    """
    if not sentences or not skill_text:
        return ""
    needle = skill_text.lower().strip()
    for s in sentences:
        if needle in s.lower():
            return s
    return ""


def load_real_items(
    input_dir: Path,
    max_items: int = 0,
    raw_jobs_csv: Path = None,
    jobs_metadata_csv: Path = None,
):
    """Load advanced_skills.csv + advanced_knowledge.csv into SkillItem / KnowledgeItem.

    If `raw_jobs_csv` (or the default `job_scraping/output/english_jobs.csv`) is
    available, best-effort looks up the actual source-sentence text for each
    skill/knowledge item by searching the job's full description. Falls back to
    using the skill text itself as `sentence_text` when no match is found
    (which preserves the legacy behavior).

    If `jobs_metadata_csv` is provided (Phase 2.1 Tier 2 role-context),
    populates each item's `job_title` from the job_id → title lookup.
    """
    from pipeline import KnowledgeItem, SkillItem, SkillType

    skills_path = input_dir / "advanced_skills.csv"
    knowledge_path = input_dir / "advanced_knowledge.csv"
    if not skills_path.exists() or not knowledge_path.exists():
        raise FileNotFoundError(
            f"need both {skills_path} and {knowledge_path}; "
            f"point --input-dir at a directory that contains both"
        )

    # Load raw-job descriptions for retroactive sentence-text recovery.
    if raw_jobs_csv is None:
        raw_jobs_csv = PROJECT_ROOT / "job_scraping" / "output" / "english_jobs.csv"
    job_descriptions = _load_job_descriptions(raw_jobs_csv)

    # Load jobs_metadata for job_title lookup (role-context). Optional — when
    # absent, items get job_title="" and the clusterer's role-context mode
    # silently falls back to the legacy uniform prefix.
    jobs_meta = _load_jobs_metadata(jobs_metadata_csv) if jobs_metadata_csv else {}

    def _job_title_for(jid: str) -> str:
        rec = jobs_meta.get(jid) if jobs_meta else None
        return (rec or {}).get("title", "") if isinstance(rec, dict) else ""

    # Pre-split descriptions per job (cached so it's only paid once per process)
    job_sentences_cache: dict = {}

    def _job_sentences(jid: str) -> list:
        if jid in job_sentences_cache:
            return job_sentences_cache[jid]
        desc = job_descriptions.get(jid, "")
        sents = _split_into_sentences(desc) if desc else []
        job_sentences_cache[jid] = sents
        return sents

    items = []

    # Per-job counter to fabricate stable sentence_ids
    job_counters = {}

    def next_sid(jid: str) -> str:
        n = job_counters.get(jid, 0)
        job_counters[jid] = n + 1
        return _build_synthetic_sentence_id(jid, n)

    # Skills
    with open(skills_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("skill") or "").strip()
            if not text:
                continue
            jid = (row.get("job_id") or "").strip()
            try:
                conf = float(row.get("confidence_score") or 0.6)
            except (TypeError, ValueError):
                conf = 0.6
            stype_raw = (row.get("type") or "Hard").strip().lower()
            stype = SkillType.SOFT if stype_raw == "soft" else SkillType.HARD
            src = (row.get("source") or "LLM").strip()
            # If the CSV carries sentence_id / sentence_text directly (modern
            # Phase-1 output, e.g. from the offline Skill-LLM batch), prefer
            # those. Otherwise fall back to the english_jobs.csv lookup.
            csv_sid = (row.get("sentence_id") or "").strip()
            csv_stext = (row.get("sentence_text") or "").strip()
            if csv_sid:
                sid = csv_sid
            else:
                sid = next_sid(jid) if jid else ""
            if csv_stext:
                sentence_text = csv_stext
            else:
                real_sentence = _best_matching_sentence(text, _job_sentences(jid))
                sentence_text = real_sentence if real_sentence else text
            items.append(
                SkillItem(
                    text=text,
                    type=stype,
                    confidence_score=conf,
                    confidence_tier=_confidence_tier_from_score(conf),
                    source=src,
                    sentence_id=sid,
                    sentence_text=sentence_text,
                    extractor_source=src,
                    job_title=_job_title_for(jid),
                )
            )

    # Knowledge
    with open(knowledge_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("knowledge") or "").strip()
            if not text:
                continue
            jid = (row.get("job_id") or "").strip()
            try:
                conf = float(row.get("confidence_score") or 0.6)
            except (TypeError, ValueError):
                conf = 0.6
            src = (row.get("source") or "LLM").strip()
            csv_sid = (row.get("sentence_id") or "").strip()
            csv_stext = (row.get("sentence_text") or "").strip()
            if csv_sid:
                sid = csv_sid
            else:
                sid = next_sid(jid) if jid else ""
            if csv_stext:
                sentence_text = csv_stext
            else:
                real_sentence = _best_matching_sentence(text, _job_sentences(jid))
                sentence_text = real_sentence if real_sentence else text
            items.append(
                KnowledgeItem(
                    text=text,
                    confidence_score=conf,
                    confidence_tier=_confidence_tier_from_score(conf),
                    source=src,
                    sentence_id=sid,
                    sentence_text=sentence_text,
                    extractor_source=src,
                    job_title=_job_title_for(jid),
                )
            )

    if max_items and len(items) > max_items:
        items = items[:max_items]

    return items


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #


def _write_cluster_summary_csv(clusters, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cluster_id",
                "stream",
                "method",
                "n_items",
                "n_unique_jobs",
                "n_skill_items",
                "n_knowledge_items",
                "cohesion_score",
                "cohesion_std",
                "summary_label",
                "top_terms",
            ]
        )
        for c in clusters:
            w.writerow(
                [
                    c.id,
                    c.stream,
                    c.method,
                    c.n_items,
                    c.n_unique_jobs,
                    c.n_skill_items,
                    c.n_knowledge_items,
                    f"{c.cohesion_score:.4f}",
                    f"{c.cohesion_std:.4f}",
                    c.summary_label,
                    " | ".join(c.top_terms),
                ]
            )


def _write_cluster_members_csv(clusters, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["cluster_id", "stream", "item_type", "item_text", "sentence_id", "extractor_source", "confidence"]
        )
        for c in clusters:
            for it in c.items:
                is_skill = hasattr(it, "type")
                item_type = (
                    f"skill_{getattr(it.type, 'value', it.type)!r}".strip("'") if is_skill else "knowledge"
                )
                w.writerow(
                    [
                        c.id,
                        c.stream,
                        item_type,
                        getattr(it, "text", ""),
                        getattr(it, "sentence_id", "") or "",
                        getattr(it, "extractor_source", "") or getattr(it, "source", ""),
                        f"{float(getattr(it, 'confidence_score', 0.0)):.3f}",
                    ]
                )


def _maybe_write_cohesion_histogram(clusters, out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cohesions = [c.cohesion_score for c in clusters]
        if not cohesions:
            return False
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(cohesions, bins=20, color="#4477aa", edgecolor="black")
        ax.axvline(0.50, linestyle="--", color="red", label="cohesion_threshold=0.50")
        ax.set_xlabel("intra-cluster cohesion (mean pairwise SBERT cosine)")
        ax.set_ylabel("number of clusters")
        ax.set_title(f"Cluster cohesion distribution (n_clusters={len(clusters)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] failed to write cohesion histogram: {e}")
        return False


def _write_summary_txt(clusters, report, out_path: Path) -> None:
    hpk = [c for c in clusters if c.stream == "hard_plus_knowledge"]
    soft = [c for c in clusters if c.stream == "soft_skill"]
    hpk_sorted = sorted(hpk, key=lambda c: c.n_items, reverse=True)
    soft_sorted = sorted(soft, key=lambda c: c.n_items, reverse=True)

    def _fmt(c) -> str:
        return (
            f"  {c.id:<18s} n={c.n_items:>3d}  jobs={c.n_unique_jobs:>3d}  "
            f"cohesion={c.cohesion_score:.3f}  ({c.method})  → {c.summary_label}"
        )

    lines = []
    lines.append("=" * 78)
    lines.append("Phase 2.1 clustering — real-data smoke test")
    lines.append("=" * 78)
    lines.append("")
    lines.append("[INPUT]")
    lines.append(f"  items input:                        {report.n_items_input}")
    lines.append(f"  items after canonicalization:       {report.n_items_canonical}")
    lines.append(f"  items after frequency filter:       {report.n_items_after_frequency_filter}")
    lines.append(f"    -> hard+knowledge stream:         {report.n_items_hard_plus_knowledge}")
    lines.append(f"    -> soft-skill stream:             {report.n_items_soft}")
    lines.append("")
    lines.append("[CLUSTERING]")
    lines.append(f"  HDBSCAN clusters:                   {report.n_clusters_hdbscan}")
    lines.append(f"  agglomerative-recovery clusters:    {report.n_clusters_recovered}")
    lines.append(f"  oversize-split children:            {report.n_clusters_split}")
    lines.append(f"  dropped (cohesion < threshold):     {report.n_clusters_dropped_low_cohesion}")
    lines.append(f"  items ungrouped (final noise):      {report.n_items_ungrouped}")
    lines.append("")
    lines.append("[COHESION]")
    lines.append(f"  mean:    {report.cohesion_mean:.4f}")
    lines.append(f"  median:  {report.cohesion_median:.4f}")
    lines.append(f"  min:     {report.cohesion_min:.4f}  (threshold = {report.config.cohesion_threshold})")
    lines.append("")
    lines.append(f"[RUNTIME] {report.runtime_seconds:.2f}s")
    lines.append("")
    lines.append("[HARD + KNOWLEDGE — top 25 by size]")
    for c in hpk_sorted[:25]:
        lines.append(_fmt(c))
    if len(hpk_sorted) > 25:
        lines.append(f"  ... and {len(hpk_sorted) - 25} more hard+knowledge clusters")
    lines.append("")
    lines.append("[SOFT SKILLS]")
    for c in soft_sorted:
        lines.append(_fmt(c))
    lines.append("")
    lines.append("[TOP TERMS — first 5 hard+knowledge clusters]")
    for c in hpk_sorted[:5]:
        lines.append(f"  {c.id} ({c.summary_label})")
        for t in c.top_terms[:8]:
            lines.append(f"    - {t}")
    lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2.1 clustering on real Phase 1 output.")
    parser.add_argument(
        "--input-dir",
        default="results.old2",
        help="directory containing advanced_skills.csv + advanced_knowledge.csv",
    )
    parser.add_argument("--output-dir", default="results/clustering_smoke")
    parser.add_argument("--cohesion-threshold", type=float, default=0.50)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-global-freq", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0, help="0 = no cap")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading items from {in_dir}")
    items = load_real_items(in_dir, max_items=args.max_items)
    print(f"[INFO] loaded {len(items)} items "
          f"(SkillItem={sum(1 for i in items if hasattr(i, 'type'))}, "
          f"KnowledgeItem={sum(1 for i in items if not hasattr(i, 'type'))})")

    from clustering import ClusteringConfig, cluster_skills

    config = ClusteringConfig(
        cohesion_threshold=args.cohesion_threshold,
        min_cluster_size=args.min_cluster_size,
        min_global_frequency=args.min_global_freq,
    )

    print(f"[INFO] config: cohesion>={config.cohesion_threshold}, "
          f"min_cluster_size={config.min_cluster_size}, "
          f"min_global_freq={config.min_global_frequency}")
    print("[INFO] clustering (cold cache may take a few minutes the first time)...")

    clusters, report = cluster_skills(items, config=config)
    print(f"[INFO] produced {len(clusters)} clusters in {report.runtime_seconds:.2f}s")

    # Write outputs
    _write_cluster_summary_csv(clusters, out_dir / "clusters.csv")
    _write_cluster_members_csv(clusters, out_dir / "cluster_members.csv")
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    if _maybe_write_cohesion_histogram(clusters, out_dir / "cohesion_hist.png"):
        print(f"[INFO] wrote cohesion_hist.png")
    _write_summary_txt(clusters, report, out_dir / "summary.txt")

    print(f"\n[INFO] outputs written to {out_dir}/")
    print("       clusters.csv          per-cluster summary")
    print("       cluster_members.csv   items per cluster")
    print("       report.json           full ClusteringReport")
    print("       cohesion_hist.png     cohesion distribution")
    print("       summary.txt           human-readable highlights (same text printed above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
