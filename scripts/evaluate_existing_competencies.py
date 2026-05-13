"""
evaluate_existing_competencies.py

Apply the Phase 2.5 grounding-gate evaluator to a previously-saved Phase 2.2
output directory (one of `results/competency_v2_live_*`).

Reads competencies.json, rehydrates as CompetencyV2 dataclasses, runs the
evaluator, and writes:
    <output_dir>/competencies_evaluated.json   — all competencies with canonical grounding fields
    <output_dir>/passing.json                  — only those that meet the gate
    <output_dir>/failing.json                  — only those below threshold
    <output_dir>/evaluator_report.json         — aggregate metrics

Usage:
    python scripts/evaluate_existing_competencies.py
        --input-dir results/competency_v2_live_gpt54mini
        [--threshold 0.80]
        [--no-sbert]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _rehydrate(dicts):
    from competency_v2_schema import CompetencyV2

    out = []
    field_names = {f for f in CompetencyV2.__dataclass_fields__.keys()}
    for d in dicts:
        kwargs = {k: v for k, v in d.items() if k in field_names}
        out.append(CompetencyV2(**kwargs))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--no-sbert", action="store_true")
    parser.add_argument(
        "--sbert-threshold",
        type=float,
        default=0.45,
        help="per-pair cosine threshold (calibrated 2026-05-13 on real data)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = PROJECT_ROOT / in_dir
    if not in_dir.exists():
        print(f"[ERROR] not a directory: {in_dir}")
        return 1

    comp_path = in_dir / "competencies.json"
    if not comp_path.exists():
        print(f"[ERROR] {comp_path} not found")
        return 1

    print(f"[INFO] loading from {comp_path}")
    raw = json.loads(comp_path.read_text(encoding="utf-8"))
    comps = _rehydrate(raw)
    print(f"[INFO] {len(comps)} competencies to evaluate")

    from competency_evaluator import EvaluatorConfig, evaluate_competencies

    config = EvaluatorConfig(
        grounding_threshold=args.threshold,
        sbert_threshold=args.sbert_threshold,
        disable_sbert=args.no_sbert,
    )
    print(
        f"[INFO] threshold={config.grounding_threshold}, "
        f"sbert={'OFF' if config.disable_sbert else 'ON'} (cosine>={config.sbert_threshold})"
    )

    passing, failing, report = evaluate_competencies(comps, config=config)

    print()
    print(f"  pass:  {report.n_passed}/{report.n_evaluated}  ({report.n_passed/max(1,report.n_evaluated)*100:.0f}%)")
    print(f"  fail:  {report.n_failed}/{report.n_evaluated}")
    print(f"  mean grounding:   {report.grounding_mean:.3f}")
    print(f"  median grounding: {report.grounding_median:.3f}")
    print(f"  min grounding:    {report.grounding_min:.3f}")
    print(f"  runtime:          {report.runtime_seconds:.2f}s")
    print()

    # Per-competency status
    print("Per-competency:")
    for c in comps:
        status = "✓" if c.grounding_passed else "✗"
        print(f"  {status}  {c.id}  score={c.grounding_score:.3f}  method={c.grounding_method:<24s}  {c.title[:80]}")

    # Save outputs
    (in_dir / "competencies_evaluated.json").write_text(
        json.dumps([c.to_dict() for c in comps], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (in_dir / "passing.json").write_text(
        json.dumps([c.to_dict() for c in passing], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (in_dir / "failing.json").write_text(
        json.dumps([c.to_dict() for c in failing], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (in_dir / "evaluator_report.json").write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
    print()
    print(f"[INFO] wrote competencies_evaluated.json / passing.json / failing.json / evaluator_report.json")
    print(f"       to {in_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
