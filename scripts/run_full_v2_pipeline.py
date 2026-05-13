"""
run_full_v2_pipeline.py

End-to-end driver for the v2 pipeline. Reads Phase 1 output, runs every
v2 stage in sequence, and emits a dashboard-ready directory:

    Phase 1 (advanced_skills.csv + advanced_knowledge.csv)
        ↓
    Phase 2.1: skill_clusterer.cluster_skills        (cohesion >= 0.55)
        ↓
    Phase 2.2: competency_generator_v2               (gpt-5.4-mini default)
        ↓
    Phase 2.5: grounding gate (>= 0.80)              (auto in generate_competencies_v2)
        ↓
    Phase 2.5b: text-quality post-check              (catches "religious" bug class)
        ↓
    Phase 2.3: KKNI labeller (SBERT)                 (overrides LLM kkni_level)
        ↓
    Phase 2.4: education aggregator                  (joins jobs_metadata.csv)
        ↓
    Output: results/competency_v2_pipeline_<tag>/
        competencies.json           — final list (passing all gates)
        competencies_failed.json    — dropped competencies + reasons
        batch_reasonings.json
        clusters.json
        pipeline_report.json        — aggregate metrics from every stage

Usage:
    python scripts/run_full_v2_pipeline.py
        [--phase1-input results.old2]
        [--jobs-metadata DATA/preprocessing/data_prepared_n1k/jobs_metadata.csv]
        [--model gpt-5.4-mini]
        [--tag latest]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_clustering_on_real_data import load_real_items

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("v2_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-input", default="results.old2",
                        help="dir containing advanced_skills.csv + advanced_knowledge.csv")
    parser.add_argument("--jobs-metadata",
                        default="DATA/preprocessing/data_prepared_n1k/jobs_metadata.csv")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--cohesion-threshold", type=float, default=0.55)
    parser.add_argument("--grounding-threshold", type=float, default=0.80)
    parser.add_argument("--skip-text-quality", action="store_true")
    parser.add_argument("--skip-kkni-labeller", action="store_true")
    parser.add_argument("--skip-education", action="store_true")
    parser.add_argument(
        "--enable-role-context", action="store_true",
        help="Phase 2.1 Tier 2: per-item role-context embedding prefix "
             "(skill in role '{job_title}': ...). Requires jobs_metadata.",
    )
    parser.add_argument(
        "--enable-cluster-refinement", action="store_true",
        help="Phase 2.1 v8 sprint: LLM-as-arbiter cluster refinement pass "
             "after agglomerative recovery. Re-arranges cluster members and "
             "noise items based on LLM verdicts; commits moves only when "
             "cluster cohesion improves monotonically.",
    )
    parser.add_argument(
        "--refinement-model", default="gpt-5.4-mini",
        help="Model for the cluster-refinement arbiter (routed via llm_client_router).",
    )
    parser.add_argument(
        "--enable-it-relevance-filter", action="store_true",
        help="v9 sprint: classify each source job + each source sentence as "
             "IT/software-engineering content (yes/no via LLM). Drops items "
             "whose job or sentence is non-IT BEFORE clustering. Targets "
             "the v8.1 climbing/pizza-assembly contamination failure mode.",
    )
    parser.add_argument(
        "--raw-jobs-csv", default="job_scraping/output/english_jobs.csv",
        help="Path to the raw scraped job-postings CSV (with `id` and "
             "`description` columns). Used by the IT-relevance filter for "
             "job-level classification.",
    )
    parser.add_argument(
        "--translate-output-to-id", action="store_true",
        help="v9 sprint: after all phases complete, translate each "
             "competency's user-facing fields (title, description, "
             "rationale, soft_skills_description, related_skills, "
             "soft_skills_required) into Bahasa Indonesia and persist "
             "as {field}_id keys. Powers the dashboard language toggle.",
    )
    parser.add_argument(
        "--translation-model", default="gpt-5.4-mini",
        help="Model for EN→ID competency translation. Default is the "
             "same gpt-5.4-mini already used for generation.",
    )
    parser.add_argument(
        "--translate-input-sentences", action="store_true",
        help="v9 sprint: detect language of each loaded item's "
             "sentence_text and translate Indonesian ones to English "
             "BEFORE downstream filters and clustering. Keeps the "
             "original Indonesian text in `sentence_text_original` for "
             "provenance. No-op when items are already English.",
    )
    parser.add_argument(
        "--enable-occupation-mapping", action="store_true",
        help="v9 sprint: after Phase 2.4, map each competency to the top-K "
             "Indonesian SKKNI occupations (skema okupasi) via multilingual "
             "SBERT cosine. Populates the `occupation_matches` field used "
             "by the dashboard's occupation filter + per-competency chips.",
    )
    parser.add_argument(
        "--skkni-csv", default="DATA/skema_okupasi_indonesia_with_desc.csv",
        help="Path to the enriched SKKNI CSV (with English Description "
             "column). Defaults to DATA/skema_okupasi_indonesia_with_desc.csv.",
    )
    parser.add_argument(
        "--occupation-top-k", type=int, default=3,
        help="Number of top-K occupation matches to keep per competency.",
    )
    parser.add_argument(
        "--occupation-min-cosine", type=float, default=0.45,
        help="Minimum cosine for a SKKNI occupation match to be retained. "
             "Below this, the match is discarded (a competency with no "
             "matches above threshold gets an empty occupation_matches list, "
             "which flags it as not anchored in any Indonesian occupational "
             "standard).",
    )
    parser.add_argument(
        "--occupation-sectors", default="",
        help="Comma-separated list of SKKNI sectors to consider for "
             "occupation mapping. Default is the ICT sector only "
             "(Teknologi Informasi dan Komunikasi). Pass an empty string "
             "(default) to use the built-in IT-only whitelist; pass 'all' "
             "to disable filtering; or pass a specific list like "
             "'Teknologi Informasi dan Komunikasi,Industri Kreatif dan Komunikasi' "
             "for IT + creative industries.",
    )
    args = parser.parse_args()

    in_dir = (PROJECT_ROOT / args.phase1_input) if not Path(args.phase1_input).is_absolute() else Path(args.phase1_input)
    jobs_meta = (PROJECT_ROOT / args.jobs_metadata) if not Path(args.jobs_metadata).is_absolute() else Path(args.jobs_metadata)
    out_dir = PROJECT_ROOT / "results" / f"competency_v2_pipeline_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_report = {
        "tag": args.tag,
        "model": args.model,
        "phase1_input": str(in_dir),
        "jobs_metadata": str(jobs_meta),
        "stages": {},
    }
    t_total = time.time()

    # -------------------- Phase 1 load --------------------
    logger.info("PHASE 1: loading SkillItem + KnowledgeItem from %s", in_dir)
    items = load_real_items(
        in_dir,
        jobs_metadata_csv=jobs_meta if jobs_meta.exists() else None,
    )
    pipeline_report["stages"]["phase1_load"] = {
        "n_items": len(items),
        "n_skills": sum(1 for it in items if hasattr(it, "type")),
        "n_knowledge": sum(1 for it in items if not hasattr(it, "type")),
    }

    # -------------------- Phase 1.4: ID→EN translation (v9 sprint) --------------------
    # When the loaded items have Indonesian sentence_text, translate each
    # unique sentence to English so downstream filters / Skill-LLM (if
    # invoked) / SBERT operate on English content. Original Indonesian is
    # preserved as `sentence_text_original` for provenance display.
    if args.translate_input_sentences and items:
        from translator import langdetect_lang, translate_to_english
        unique_texts: dict = {}
        for it in items:
            stxt = (getattr(it, "sentence_text", "") or "").strip()
            if stxt:
                unique_texts.setdefault(stxt, []).append(it)
        translated_map: dict = {}
        n_id, n_translated = 0, 0
        for stxt in unique_texts:
            lang = langdetect_lang(stxt)
            if lang == "id":
                n_id += 1
                en = translate_to_english(stxt)
                if en and en != stxt:
                    translated_map[stxt] = en
                    n_translated += 1
        if translated_map:
            for stxt, en in translated_map.items():
                for it in unique_texts[stxt]:
                    setattr(it, "sentence_text_original", stxt)
                    it.sentence_text = en
        logger.info(
            "Phase 1.4: ID→EN translation — %d unique sentences scanned, "
            "%d detected ID, %d translated",
            len(unique_texts), n_id, n_translated,
        )
        pipeline_report["stages"]["phase1_4_translation"] = {
            "n_unique_sentences": len(unique_texts),
            "n_detected_indonesian": n_id,
            "n_translated": n_translated,
        }

    # -------------------- Phase 1.5: IT-relevance gate (v9 sprint) --------------------
    # Drops items whose source JOB is not IT/software-engineering, then drops
    # items whose source SENTENCE is not IT content. Targets the v8.1
    # climbing/pizza-assembly contamination failure where boilerplate
    # physical-requirements text reached the clusterer.
    if args.enable_it_relevance_filter:
        from pipeline_it_filter import apply_it_relevance_filter
        raw_jobs_path = (PROJECT_ROOT / args.raw_jobs_csv) if not Path(args.raw_jobs_csv).is_absolute() else Path(args.raw_jobs_csv)
        items, it_audit = apply_it_relevance_filter(
            items,
            raw_jobs_csv=raw_jobs_path if raw_jobs_path.exists() else None,
            cache_dir=PROJECT_ROOT / "cache" / "it_relevance_filter",
        )
        pipeline_report["stages"]["phase1_5_it_relevance"] = it_audit["summary"]
        (out_dir / "it_relevance_audit.json").write_text(
            json.dumps(it_audit, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        logger.info(
            "Phase 1.5: IT-relevance gate kept %d / %d items (jobs %d/%d, sentences %d/%d)",
            it_audit["summary"]["n_items_kept"], it_audit["summary"]["n_items_in"],
            it_audit["summary"]["n_jobs_it"], it_audit["summary"]["n_jobs_total"],
            it_audit["summary"]["n_sentences_it"], it_audit["summary"]["n_sentences_total"],
        )

    # -------------------- Phase 2.1: clustering --------------------
    logger.info(
        "PHASE 2.1: clustering (cohesion >= %.2f, role_context=%s, refinement=%s)",
        args.cohesion_threshold, args.enable_role_context, args.enable_cluster_refinement,
    )
    from clustering import ClusteringConfig, cluster_skills
    cl_cfg = ClusteringConfig(
        cohesion_threshold=args.cohesion_threshold,
        enable_role_context=args.enable_role_context,
        enable_cluster_refinement=args.enable_cluster_refinement,
        refinement_model=args.refinement_model,
    )
    clusters, cl_report = cluster_skills(items, config=cl_cfg)
    n_hpk = sum(1 for c in clusters if c.stream == "hard_plus_knowledge")
    logger.info("Phase 2.1: %d clusters total (%d hard+knowledge for Phase 2.2)", len(clusters), n_hpk)
    pipeline_report["stages"]["phase2_1_clustering"] = cl_report.to_dict()

    # Persist clusters
    (out_dir / "clusters.json").write_text(
        json.dumps([c.to_dict() for c in clusters], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # v8.1: persist the full noise audit alongside clusters.json so the
    # dashboard's Pipeline-audit page can load it as a separate file.
    if cl_report.noise_audit:
        (out_dir / "noise_audit.json").write_text(
            json.dumps(cl_report.noise_audit, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Phase 2.1: noise_audit.json written (%d items dropped, by stage: %s)",
            cl_report.noise_audit.get("n_items_dropped_total", 0),
            cl_report.noise_audit.get("by_stage", {}),
        )

    # -------------------- Phase 2.2: competency generation + 2.5 grounding gate --------------------
    logger.info("PHASE 2.2: generating competencies via %s (Phase 2.5 grounding gate auto-applied)", args.model)
    from competency_v2_schema import GeneratorConfig
    from competency_generator_v2 import generate_competencies_v2
    g_cfg = GeneratorConfig(model=args.model)

    # v8 sprint Phase 2.A: load jobs_metadata for role distribution in the
    # generator prompt. Education-stage demand stays in Phase 2.4 only —
    # NOT passed to the generator (KKNI placement must be cognitive-complexity
    # only).
    job_titles_for_generator: dict = {}
    if jobs_meta.exists():
        import csv as _csv
        try:
            with open(jobs_meta, encoding="utf-8-sig") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    jid = (row.get("job_id") or "").strip()
                    title = (row.get("title") or "").strip()
                    if jid and title:
                        job_titles_for_generator[jid] = title
            logger.info(
                "Phase 2.2: loaded %d job_id->title mappings for role distribution",
                len(job_titles_for_generator),
            )
        except Exception as e:
            logger.warning("Phase 2.2: jobs_metadata load failed (%s); proceeding without role distribution", e)

    competencies, batch_reasonings = generate_competencies_v2(
        clusters=clusters,
        config=g_cfg,
        apply_grounding_gate=True,
        job_titles=job_titles_for_generator or None,
    )
    logger.info("Phase 2.2 + 2.5: %d competencies survived grounding gate", len(competencies))
    pipeline_report["stages"]["phase2_2_generation"] = {
        "n_competencies": len(competencies),
        "n_batch_reasonings": len(batch_reasonings),
        "model": args.model,
        "grounding_threshold": args.grounding_threshold,
    }

    # -------------------- Phase 2.5b: text-quality post-check --------------------
    failed_text_quality = []
    if not args.skip_text_quality and competencies:
        logger.info("PHASE 2.5b: text-quality post-check")
        from competency_text_quality import TextQualityConfig, check_text_quality
        tq_passing, tq_failing, tq_report = check_text_quality(competencies, TextQualityConfig())
        if tq_failing:
            logger.warning(
                "Phase 2.5b: dropping %d competencies for text-quality violations",
                len(tq_failing),
            )
            for c in tq_failing:
                logger.warning("  - %s: %s", c.id, c.title)
        competencies = tq_passing
        failed_text_quality = tq_failing
        pipeline_report["stages"]["phase2_5b_text_quality"] = {
            "n_passed": tq_report["n_passed"],
            "n_failed": tq_report["n_failed"],
        }

    # -------------------- Phase 2.3: KKNI labeller --------------------
    if not args.skip_kkni_labeller and competencies:
        logger.info("PHASE 2.3: KKNI labeller (multilingual SBERT)")
        from competency_kkni_labeller import KkniLabellerConfig, label_competencies_kkni
        kkni_report = label_competencies_kkni(competencies, KkniLabellerConfig())
        pipeline_report["stages"]["phase2_3_kkni"] = kkni_report
        logger.info("Phase 2.3: %d/%d relabelled", kkni_report["n_relabeled"], kkni_report["n_evaluated"])

    # -------------------- Phase 2.4: education aggregator --------------------
    if not args.skip_education and competencies:
        if jobs_meta.exists():
            logger.info("PHASE 2.4: education-level demand aggregation")
            from competency_education_aggregator import aggregate_education_demand
            edu_report = aggregate_education_demand(competencies, jobs_meta)
            pipeline_report["stages"]["phase2_4_education"] = edu_report
            logger.info("Phase 2.4: %d/%d competencies have education data",
                        edu_report["n_with_education_data"], edu_report["n_evaluated"])
        else:
            logger.warning("PHASE 2.4 skipped — jobs_metadata not found at %s", jobs_meta)
            pipeline_report["stages"]["phase2_4_education"] = {"skipped": True, "reason": "no jobs_metadata.csv"}

    # -------------------- Phase 2.4b — SKKNI occupation mapping (v9 sprint) --------------------
    # Map each surviving competency to the top-K Indonesian occupations via
    # multilingual SBERT cosine. Populates the `occupation_matches` field on
    # each CompetencyV2; an empty list means the competency does not anchor
    # to any Indonesian occupational standard.
    if args.enable_occupation_mapping and competencies:
        skkni_path = (PROJECT_ROOT / args.skkni_csv) if not Path(args.skkni_csv).is_absolute() else Path(args.skkni_csv)
        if not skkni_path.exists():
            logger.warning(
                "PHASE 2.4b skipped — SKKNI CSV not found at %s", skkni_path,
            )
            pipeline_report["stages"]["phase2_4b_occupations"] = {
                "skipped": True, "reason": "no SKKNI CSV",
            }
        else:
            logger.info("PHASE 2.4b: mapping competencies to SKKNI occupations")
            from occupation_mapper import load_occupations, map_competencies
            occupations = load_occupations(skkni_path)
            # CompetencyV2 instances expose `occupation_matches` as an attribute,
            # but `map_competencies` operates on dicts. Round-trip via the dict
            # representation and mirror the result back onto the instances.
            comp_dicts_for_mapping = [c.to_dict() for c in competencies]
            # Parse sector override
            from occupation_mapper import NO_SECTOR_FILTER
            sectors_arg = (args.occupation_sectors or "").strip()
            if sectors_arg.lower() == "all":
                sector_kw = NO_SECTOR_FILTER
            elif sectors_arg:
                sector_kw = [s.strip() for s in sectors_arg.split(",") if s.strip()]
            else:
                sector_kw = None  # use module default whitelist
            map_competencies(
                comp_dicts_for_mapping,
                occupations=occupations,
                top_k=args.occupation_top_k,
                min_cosine=args.occupation_min_cosine,
                relevant_sectors=sector_kw,
            )
            for c, d in zip(competencies, comp_dicts_for_mapping):
                c.occupation_matches = d.get("occupation_matches", []) or []
            n_mapped = sum(1 for c in competencies if c.occupation_matches)
            pipeline_report["stages"]["phase2_4b_occupations"] = {
                "n_competencies": len(competencies),
                "n_mapped": n_mapped,
                "n_unmapped": len(competencies) - n_mapped,
                "top_k": args.occupation_top_k,
                "min_cosine": args.occupation_min_cosine,
                "skkni_n_occupations": len(occupations),
            }
            logger.info(
                "Phase 2.4b: %d / %d competencies mapped to >=1 occupation",
                n_mapped, len(competencies),
            )

    # -------------------- Phase 4 — priority / demand scoring --------------------
    # Computed AFTER Phase 2.4 so every competency has its final grounding +
    # demand + future_weight values. The priority_score formula tracks Req 9.6.
    if competencies:
        max_n_jobs = max((len(c.source_job_ids) for c in competencies), default=1) or 1
        for c in competencies:
            n_jobs = len(c.source_job_ids)
            c.demand_score = n_jobs / max_n_jobs if max_n_jobs > 0 else 0.0
            grounding = float(c.grounding_score or 0.0)
            fw = float(c.future_weight or 0.0)
            c.priority_score = round(
                0.40 * c.demand_score + 0.30 * grounding + 0.30 * fw, 4
            )
        pipeline_report["stages"]["phase4_priority"] = {
            "max_n_unique_jobs": int(max_n_jobs),
            "formula": "0.40 * demand + 0.30 * grounding + 0.30 * future_weight",
        }
        logger.info(
            "Phase 4: priority/demand scored across %d competencies (max_n_jobs=%d)",
            len(competencies), max_n_jobs,
        )

    # -------------------- v9: EN→ID translation (optional) --------------------
    competency_dicts = [c.to_dict() for c in competencies]
    if args.translate_output_to_id and competency_dicts:
        logger.info(
            "Phase v9: translating %d competencies EN→ID via %s",
            len(competency_dicts), args.translation_model,
        )
        from translator import translate_competency_to_indonesian
        competency_dicts = [
            translate_competency_to_indonesian(c, model=args.translation_model)
            for c in competency_dicts
        ]
        # Translate the BatchReasoning text too so users browsing in ID see
        # the reasoning in their language.
        from translator import translate_to_indonesian as _tr_id
        for br in batch_reasonings:
            br_text = getattr(br, "batch_reasoning", "") or ""
            if br_text:
                # Mutate the BatchReasoning so its to_dict() picks it up.
                # Add the *_id field via a per-instance attribute.
                br.batch_reasoning_id = _tr_id(br_text, model=args.translation_model)
        pipeline_report["stages"]["phase_v9_translation"] = {
            "model": args.translation_model,
            "n_competencies_translated": len(competency_dicts),
            "n_batch_reasonings_translated": sum(
                1 for br in batch_reasonings if getattr(br, "batch_reasoning_id", None)
            ),
        }

    # -------------------- Persist --------------------
    (out_dir / "competencies.json").write_text(
        json.dumps(competency_dicts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "competencies_failed.json").write_text(
        json.dumps([c.to_dict() for c in failed_text_quality], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "batch_reasonings.json").write_text(
        json.dumps([br.to_dict() for br in batch_reasonings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pipeline_report["total_runtime_seconds"] = round(time.time() - t_total, 2)
    pipeline_report["final_n_competencies"] = len(competencies)

    (out_dir / "pipeline_report.json").write_text(
        json.dumps(pipeline_report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(f"v2 pipeline complete — {len(competencies)} final competencies in {pipeline_report['total_runtime_seconds']:.1f}s")
    print(f"  Output: {out_dir}")
    print(f"  Launch dashboard:")
    print(f"    streamlit run dashboard_v2/app.py -- --run-dir {out_dir.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
