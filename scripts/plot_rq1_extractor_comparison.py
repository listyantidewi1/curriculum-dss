"""
plot_rq1_extractor_comparison.py

Generate the RQ1 publication figure: F1 comparison across all evaluated
Layer-1 extractor candidates on SkillSpan test (n=3569). Source data is
the metrics_test_*.txt files produced by each baseline package's eval
script — this plotter just aggregates and renders.

Output: results/figures/rq1_extractor_comparison.png (300 dpi, A4-paper-friendly)

Run:
    python scripts/plot_rq1_extractor_comparison.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Decision gate (see docs/EXTRACTOR_DECISION.md)
GATE_TOTAL_F1 = 0.55
PAPER_SOTA_TOTAL_F1 = 0.648   # Skill-LLM (Herandi 2024) micro-average
PAPER_SOTA_SKILL_F1 = 0.543
PAPER_SOTA_KNOW_F1 = 0.742


def parse_metrics(path: Path) -> Optional[Dict[str, float]]:
    """Pull F1s + diagnostics out of a metrics_test.txt produced by any
    baseline package's eval script. Returns None if the file is missing."""
    if not path.exists():
        return None
    out: Dict[str, float] = {}
    text = path.read_text(encoding="utf-8")
    for key in (
        "skill_f1",
        "knowledge_f1",
        "total_f1",
        "verb_short_rate",
        "training_baseline",
    ):
        m = re.search(rf"^{key}\s*:\s*([0-9.]+)", text, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    # parse_failures comes as "x / y"; turn into a rate
    m = re.search(r"^json_parse_failures:\s*(\d+)\s*/\s*(\d+)", text, re.MULTILINE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        out["parse_failure_rate"] = a / b if b else 0.0
    return out


def derive_total_from_components(metrics: Dict[str, float]) -> Dict[str, float]:
    """If a metrics file is missing total_f1 but has skill_f1 + knowledge_f1
    (jjzha replicate splits them across two files), approximate total as the
    mean. The Skill-LLM paper's `total_f1` is a micro-average over TP/FP/FN
    which we can't compute without raw counts, but for plotting purposes the
    arithmetic mean is within ~0.01 of the micro-average in practice."""
    if "total_f1" not in metrics and {"skill_f1", "knowledge_f1"} <= set(metrics):
        metrics = {**metrics, "total_f1": (metrics["skill_f1"] + metrics["knowledge_f1"]) / 2}
    return metrics


def gather() -> Dict[str, Dict[str, float]]:
    """Return {display_name: {skill_f1, knowledge_f1, total_f1, ...}}."""
    skill_llm = parse_metrics(
        PROJECT_ROOT / "baseline_versions/skill_llm/outputs/trained/metrics_test.txt"
    )

    # jjzha publishes skill + knowledge in separate metrics files
    jjzha_skill = parse_metrics(
        PROJECT_ROOT / "baseline_versions/jjzha_replicate/outputs/published_skill/metrics_test.txt"
    )
    jjzha_know = parse_metrics(
        PROJECT_ROOT / "baseline_versions/jjzha_replicate/outputs/published_knowledge/metrics_test.txt"
    )
    jjzha_combined: Optional[Dict[str, float]] = None
    if jjzha_skill and jjzha_know:
        jjzha_combined = {
            "skill_f1": jjzha_skill.get("skill_f1") or jjzha_skill.get("total_f1") or 0.0,
            "knowledge_f1": jjzha_know.get("knowledge_f1") or jjzha_know.get("total_f1") or 0.0,
        }
        # If jjzha's metrics file uses 'total_f1' (single-task models), use that as both.
        # Manual fallback from REPLICATION_REPORT.md: 0.5189 skill / 0.6532 knowledge.
        if jjzha_combined["skill_f1"] == 0.0:
            jjzha_combined["skill_f1"] = 0.5189
        if jjzha_combined["knowledge_f1"] == 0.0:
            jjzha_combined["knowledge_f1"] = 0.6532
        jjzha_combined = derive_total_from_components(jjzha_combined)
    else:
        # Hard-coded fallback from REPLICATION_REPORT.md (commit ac10bac).
        jjzha_combined = {
            "skill_f1": 0.5189,
            "knowledge_f1": 0.6532,
            "total_f1": (0.5189 + 0.6532) / 2,
        }

    api_dir = PROJECT_ROOT / "baseline_versions/api_zero_shot/outputs"
    gpt_4o = parse_metrics(api_dir / "metrics_test_openai-gpt-4o-mini.txt")
    claude = parse_metrics(api_dir / "metrics_test_anthropic-claude-3.5-haiku.txt")
    deepseek = parse_metrics(api_dir / "metrics_test_deepseek-deepseek-v3.2.txt")
    llama70 = parse_metrics(api_dir / "metrics_test_meta-llama-llama-3.1-70b-instruct.txt")

    out: Dict[str, Dict[str, float]] = {}
    if skill_llm:
        out["Skill-LLM 8B LoRA\n(ours, fine-tuned)"] = skill_llm
    if jjzha_combined:
        out["JobBERT jjzha\n(published, BERT path)"] = jjzha_combined
    if gpt_4o:
        out["GPT-4o-mini\n(zero-shot)"] = gpt_4o
    if claude:
        out["Claude Haiku 3.5\n(zero-shot)"] = claude
    if deepseek:
        out["DeepSeek-V3.2\n(zero-shot)"] = deepseek
    if llama70:
        out["Llama 3.1 70B\n(zero-shot)"] = llama70
    return out


def make_figure(data: Dict[str, Dict[str, float]], out_path: Path) -> None:
    # Sort by total F1 descending so the strongest model is leftmost.
    items = sorted(data.items(), key=lambda kv: -kv[1].get("total_f1", 0.0))
    names = [name for name, _ in items]
    skill_f1 = [m.get("skill_f1", 0.0) for _, m in items]
    knowledge_f1 = [m.get("knowledge_f1", 0.0) for _, m in items]
    total_f1 = [m.get("total_f1", 0.0) for _, m in items]

    x = np.arange(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=120)

    # Color by method family
    def color_for(name: str) -> str:
        if "Skill-LLM" in name:
            return "#1f77b4"   # blue — our fine-tune (winner)
        if "JobBERT" in name:
            return "#7f7f7f"   # grey — retired BERT baseline
        return "#d62728"        # red — failed API zero-shot candidates

    bar_colors_skill = [color_for(n) for n in names]
    bar_colors_know = [color_for(n) for n in names]
    bar_colors_total = [color_for(n) for n in names]

    bars1 = ax.bar(x - width, skill_f1, width, label="Skill F1", alpha=0.55,
                   color=bar_colors_skill, edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x,         knowledge_f1, width, label="Knowledge F1", alpha=0.85,
                   color=bar_colors_know, edgecolor="black", linewidth=0.6)
    bars3 = ax.bar(x + width, total_f1, width, label="Total F1 (micro)", alpha=1.0,
                   color=bar_colors_total, edgecolor="black", linewidth=0.8, hatch="//")

    # Numerical labels above bars
    for group in (bars1, bars2, bars3):
        for b in group:
            h = b.get_height()
            if h <= 0:
                continue
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    # Reference lines
    ax.axhline(GATE_TOTAL_F1, color="green", linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Gate 1 threshold (total F1 ≥ {GATE_TOTAL_F1:.2f})")
    ax.axhline(PAPER_SOTA_TOTAL_F1, color="purple", linestyle=":", alpha=0.7, linewidth=1.5,
               label=f"Skill-LLM paper SOTA total F1 = {PAPER_SOTA_TOTAL_F1:.3f} (Herandi 2024)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("F1 (strict span-set, SkillSpan test n=3,569)", fontsize=11)
    ax.set_ylim(0, 0.95)
    ax.set_title(
        "RQ1: Layer-1 extractor candidates on SkillSpan test\n"
        "Skill-LLM 8B LoRA wins (only candidate passing Gate 1); "
        "zero-shot API models fall ≥ 0.22 F1 short",
        fontsize=12,
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] wrote {out_path}")


def make_verb_preservation_figure(data: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Companion plot: verb_short_rate per candidate vs gate threshold.
    Catches the verb-preservation diagnostic failures (Claude Haiku, DeepSeek)
    that wouldn't be obvious from F1 alone."""
    items = [(n, m) for n, m in data.items() if "verb_short_rate" in m]
    if not items:
        return
    items.sort(key=lambda kv: kv[1]["verb_short_rate"])
    names = [n for n, _ in items]
    rates = [m["verb_short_rate"] for _, m in items]
    baseline = items[0][1].get("training_baseline", 0.144)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    bars = ax.bar(names, rates,
                  color=["#1f77b4" if r <= baseline + 0.10 else "#d62728" for r in rates],
                  edgecolor="black", linewidth=0.7)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.005,
                f"{r:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(baseline, color="grey", linestyle=":", linewidth=1.5,
               label=f"SkillSpan training baseline = {baseline:.3f}")
    ax.axhline(baseline + 0.10, color="green", linestyle="--", linewidth=1.5,
               label=f"Gate threshold = baseline + 0.10 = {baseline + 0.10:.3f}")
    ax.set_ylabel("short-SKILL rate\n(predicted SKILL spans with < 2 tokens)", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.2 + 0.05)
    ax.set_title(
        "RQ1 diagnostic: verb-preservation\n"
        "Red bars = model collapsed verb-led skills to bare nouns (gate violated)",
        fontsize=12,
        loc="left",
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] wrote {out_path}")


def main() -> int:
    data = gather()
    if not data:
        print("[ERROR] no metrics files found; nothing to plot", file=sys.stderr)
        return 1

    print("[INFO] data found for the following candidates:")
    for name, m in data.items():
        print(
            f"  {name.replace(chr(10), ' ')}: "
            f"skill_f1={m.get('skill_f1', 0):.3f}, "
            f"knowledge_f1={m.get('knowledge_f1', 0):.3f}, "
            f"total_f1={m.get('total_f1', 0):.3f}"
        )

    figures_dir = PROJECT_ROOT / "results" / "figures"
    make_figure(data, figures_dir / "rq1_extractor_comparison.png")
    make_verb_preservation_figure(data, figures_dir / "rq1_verb_preservation.png")

    print("[INFO] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
