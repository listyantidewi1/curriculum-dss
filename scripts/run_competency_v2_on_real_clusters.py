"""
run_competency_v2_on_real_clusters.py

End-to-end live run of Phase 2.2 on real Phase 1 output.

Flow:
    1. Load SkillItem + KnowledgeItem from results.old2/ (advanced_skills.csv +
       advanced_knowledge.csv) — same loader used by test_clustering_on_real_data.py.
    2. Run Phase 2.1 clustering with the locked default (cohesion >= 0.55).
    3. Call generate_competencies_v2() on the resulting clusters.
       - Hits OpenRouter (DeepSeek-V3.2 by default) — costs ~$0.002 for the
         13-cluster real dataset.
    4. Dump everything to results/competency_v2_live/ for inspection:
         competencies.json    — list of CompetencyV2 dicts
         batch_reasonings.json — list of BatchReasoning dicts
         summary.txt          — printed report

Usage:
    python scripts/run_competency_v2_on_real_clusters.py
    python scripts/run_competency_v2_on_real_clusters.py --model openai/gpt-5
    python scripts/run_competency_v2_on_real_clusters.py --cohesion-threshold 0.60

Optional flags:
    --input-dir          where to load Phase 1 output (default: results.old2)
    --output-dir         where to write outputs (default: results/competency_v2_live)
    --model              LLM model id (default: deepseek/deepseek-v3.2)
    --cohesion-threshold override clustering cohesion gate (default: 0.55)
    --no-llm             dry-run: skip the LLM call and just print clusters
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_clustering_on_real_data import load_real_items


def _write_summary(competencies, batch_reasonings, out_path: Path) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append("Phase 2.2 live run — competencies from real Phase 1 output")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"  competencies produced: {len(competencies)}")
    lines.append(f"  LLM calls (BatchReasoning records): {len(batch_reasonings)}")
    total_prompt = sum(br.prompt_tokens for br in batch_reasonings)
    total_completion = sum(br.completion_tokens for br in batch_reasonings)
    total_latency = sum(br.latency_seconds for br in batch_reasonings)
    lines.append(f"  total prompt tokens:     {total_prompt:,}")
    lines.append(f"  total completion tokens: {total_completion:,}")
    lines.append(f"  total LLM wall-clock:    {total_latency:.1f}s")
    lines.append("")

    for i, c in enumerate(competencies, 1):
        lines.append("-" * 88)
        lines.append(f"[{i}] {c.title}")
        lines.append(f"    id={c.id}  cluster={c.cluster_id}  provider={c.provider}  model={c.model}")
        lines.append(f"    KKNI level: {c.kkni_level}   future_weight: {c.future_weight:.2f}   "
                     f"trend: {c.empirical_trend}   grounding_preview: {c.grounding_score_preview:.2f}")
        lines.append(f"    description: {c.description}")
        lines.append(f"    related_skills ({len(c.related_skills)}):")
        for s in c.related_skills:
            lines.append(f"       - {s}")
        if c.soft_skills_required:
            lines.append(f"    soft_skills_required: {', '.join(c.soft_skills_required)}")
        lines.append(f"    rationale ({len(c.rationale)} chars):")
        # Wrap rationale at ~80 chars for readability
        for chunk in _wrap(c.rationale, 80):
            lines.append(f"       {chunk}")
        if c.merged_from:
            lines.append(f"    merged_from: {c.merged_from}")
        lines.append(f"    provenance: {len(c.contributing_item_ids)} items, "
                     f"{len(c.source_job_ids)} unique jobs, "
                     f"{len(c.source_sentences)} source sentences")

    lines.append("")
    lines.append("=" * 88)
    lines.append("[BATCH REASONINGS]")
    lines.append("=" * 88)
    for i, br in enumerate(batch_reasonings, 1):
        lines.append("-" * 88)
        lines.append(f"[{i}] cluster={br.cluster_id}  comps_out={br.n_competencies_out}  "
                     f"latency={br.latency_seconds:.1f}s")
        lines.append(f"    batch_reasoning ({len(br.batch_reasoning)} chars):")
        for chunk in _wrap(br.batch_reasoning, 80):
            lines.append(f"       {chunk}")

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return text


def _wrap(s: str, width: int) -> list:
    import textwrap
    if not s:
        return [""]
    return textwrap.wrap(s, width=width) or [""]


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Phase 2.2 run on real clusters.")
    parser.add_argument("--input-dir", default="results.old2")
    parser.add_argument("--output-dir", default="results/competency_v2_live")
    parser.add_argument("--model", default="deepseek/deepseek-v3.2")
    parser.add_argument("--cohesion-threshold", type=float, default=0.55)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-global-freq", type=int, default=2)
    parser.add_argument("--no-llm", action="store_true",
                        help="Dry run: skip the LLM call.")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load items
    print(f"[INFO] loading items from {in_dir}")
    items = load_real_items(in_dir)
    print(f"[INFO] loaded {len(items)} items")

    # 2) Cluster
    from clustering import ClusteringConfig, cluster_skills

    cfg_c = ClusteringConfig(
        cohesion_threshold=args.cohesion_threshold,
        min_cluster_size=args.min_cluster_size,
        min_global_frequency=args.min_global_freq,
    )
    print(f"[INFO] clustering with cohesion>={cfg_c.cohesion_threshold}")
    clusters, c_report = cluster_skills(items, config=cfg_c)
    n_hpk = sum(1 for c in clusters if c.stream == "hard_plus_knowledge")
    print(f"[INFO] produced {len(clusters)} clusters total "
          f"({n_hpk} hard+knowledge will be sent to LLM)")

    # Save cluster summary so we can correlate competencies → source cluster later
    with open(out_dir / "clusters.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in clusters], f, indent=2)

    if args.no_llm:
        print("[INFO] --no-llm set; stopping after clustering. Cluster summary written.")
        return 0

    # 3) Generate competencies via LLM
    from competency_v2_schema import GeneratorConfig
    from competency_generator_v2 import generate_competencies_v2

    cfg_g = GeneratorConfig(model=args.model)
    print(f"[INFO] generating competencies via {cfg_g.model}")
    print("[INFO] this calls OpenRouter (DeepSeek) or Jatevo (GPT) — expect 30-90s wall-clock")

    competencies, batch_reasonings = generate_competencies_v2(
        clusters=clusters,
        config=cfg_g,
    )
    print(f"[INFO] generation complete: {len(competencies)} competencies, "
          f"{len(batch_reasonings)} LLM calls")

    # 4) Persist + print summary
    with open(out_dir / "competencies.json", "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in competencies], f, indent=2, ensure_ascii=False)
    with open(out_dir / "batch_reasonings.json", "w", encoding="utf-8") as f:
        json.dump([br.to_dict() for br in batch_reasonings], f, indent=2, ensure_ascii=False)

    _write_summary(competencies, batch_reasonings, out_dir / "summary.txt")

    print()
    print(f"[INFO] outputs written to {out_dir}/")
    print("       competencies.json     — full competency dicts (with provenance)")
    print("       batch_reasonings.json — LLM CoT records")
    print("       clusters.json         — Phase 2.1 cluster summary")
    print("       summary.txt           — human-readable report (same text printed above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
