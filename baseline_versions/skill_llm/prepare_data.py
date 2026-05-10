"""
prepare_data.py — skill_llm

Convert SkillSpan train / dev / test JSON to the Skill-LLM chat-format JSONL
expected by `train.py`. Each example becomes a chat with three messages
(system / user / assistant); the assistant message is the gold JSON output.

The output schema is verbatim from Skill-LLM Figure 1:

    {"SKILL":     [{"skill_span": <verb-led action phrase>,
                    "context":    <skill_span + 1 token on each side>}, ...],
     "KNOWLEDGE": [{"skill_span": <noun phrase>,
                    "context":    <skill_span + 1 token on each side>}, ...]}

Critical invariant — VERB PRESERVATION (see AUDIT.md):
    SkillSpan annotates skills as verb-led action phrases and knowledge as
    nominal phrases. We DO NOT modify these annotations. The training data
    inherits the gold distinction:
        SKILL.skill_span       = verb-led ("designing UI/UX", "implementing X")
        KNOWLEDGE.skill_span   = noun     ("UI/UX",          "X")

This script reports a sanity check at the end: the fraction of training-set
SKILL spans with < 2 tokens. If that's > 5%, the SkillSpan annotation is
unusual (or this script has a bug); investigate before training.

Usage:
    python prepare_data.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    DATA_DIR,
    DATASETS_DIR,
    SENTENCE_BOUNDARY_TOKEN,
    SYSTEM_PROMPT,
    VERB_PRESERVATION_MIN_TOKENS,
)


def load_skillspan(data_dir: Path) -> Dict[str, List[dict]]:
    """Load SkillSpan train / dev / test JSON. Mirrors the loader in
    `baseline_versions/jjzha_replicate/data_utils.py`."""
    out: Dict[str, List[dict]] = {}
    for split, fname in (("train", "train.json"), ("dev", "dev.json"), ("test", "test.json")):
        path = data_dir / fname
        if not path.exists():
            print(f"[WARN] {fname} not found in {data_dir}; skipping {split}")
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                out[split] = content
            elif isinstance(content, dict):
                for v in content.values():
                    if isinstance(v, list):
                        out[split] = v
                        break
        except json.JSONDecodeError:
            data: List[dict] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data.append(json.loads(line))
            out[split] = data
        if split in out:
            print(f"[INFO] {split}: {len(out[split])} examples")
    return out


def bio_to_spans(tags: List[str]) -> List[Tuple[int, int]]:
    """Reduce a token-aligned BIO sequence to (start_token_idx, end_token_idx)
    inclusive ranges, one per span."""
    spans: List[Tuple[int, int]] = []
    start = None
    for i, tag in enumerate(tags):
        if tag == "B":
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == "I" and start is not None:
            continue
        else:  # 'O' or stray 'I'
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(tags) - 1))
    return spans


def span_with_context(tokens: List[str], start: int, end: int) -> Tuple[str, str]:
    """Return (skill_span, context) for the inclusive token range [start, end].

    `context` extends one token to either side, matching Skill-LLM Fig 1. If the
    span sits at the sentence boundary, the corresponding side is omitted (the
    SENTENCE_BOUNDARY_TOKEN we add at the user-message level still gives the
    model a stable anchor)."""
    skill_span = " ".join(tokens[start : end + 1])
    ctx_start = max(0, start - 1)
    ctx_end = min(len(tokens) - 1, end + 1)
    context = " ".join(tokens[ctx_start : ctx_end + 1])
    return skill_span, context


def build_target(tokens: List[str], tags_skill: List[str], tags_knowledge: List[str]) -> dict:
    """Construct the gold assistant-message JSON object for one example."""
    skills = []
    for s, e in bio_to_spans(tags_skill):
        sp, ctx = span_with_context(tokens, s, e)
        skills.append({"skill_span": sp, "context": ctx})

    knowledges = []
    for s, e in bio_to_spans(tags_knowledge):
        sp, ctx = span_with_context(tokens, s, e)
        knowledges.append({"skill_span": sp, "context": ctx})

    return {"SKILL": skills, "KNOWLEDGE": knowledges}


def build_user_message(tokens: List[str]) -> str:
    """Wrap the source sentence with boundary markers (Skill-LLM Fig 1)."""
    sentence = " ".join(tokens)
    return f"{SENTENCE_BOUNDARY_TOKEN} {sentence} {SENTENCE_BOUNDARY_TOKEN}"


def to_chat(example: dict) -> dict:
    """Convert one SkillSpan example to a three-message chat record."""
    tokens = example["tokens"]
    tags_skill = example.get("tags_skill", [])
    tags_knowledge = example.get("tags_knowledge", [])

    target = build_target(tokens, tags_skill, tags_knowledge)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(tokens)},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "_meta": {
            "tokens": tokens,
            "n_skill_spans": len(target["SKILL"]),
            "n_knowledge_spans": len(target["KNOWLEDGE"]),
        },
    }


def write_training_stats(records: List[dict]) -> None:
    """Record training-set statistics that eval.py uses to calibrate its
    diagnostic. The key number is the fraction of SKILL spans that are
    < VERB_PRESERVATION_MIN_TOKENS tokens long — i.e., the natural rate of
    single-token skills (mostly soft skills like 'passion', 'empathetic')
    in the training distribution. Eval flags the model only if its rate
    drifts above this baseline by more than the tolerance.
    """
    total = 0
    short = 0
    short_examples: List[str] = []
    knowl_total = 0
    for rec in records:
        target = json.loads(rec["messages"][2]["content"])
        for s in target.get("SKILL", []):
            total += 1
            n_tokens = len(s["skill_span"].split())
            if n_tokens < VERB_PRESERVATION_MIN_TOKENS:
                short += 1
                if len(short_examples) < 20:
                    short_examples.append(s["skill_span"])
        knowl_total += len(target.get("KNOWLEDGE", []) or [])

    rate = (short / total) if total else 0.0
    stats = {
        "n_examples": len(records),
        "n_skill_spans": total,
        "n_skill_short_spans": short,
        "skill_short_rate": rate,
        "n_knowledge_spans": knowl_total,
        "verb_preservation_min_tokens": VERB_PRESERVATION_MIN_TOKENS,
        "sample_short_skill_spans": short_examples,
        "_note": (
            "skill_short_rate is the fraction of SKILL spans shorter than "
            "verb_preservation_min_tokens. SkillSpan natively has ~14% of "
            "SKILL spans as single-token soft skills (passion, empathetic, "
            "self-starter, ...). eval.py uses this rate as the baseline "
            "for its verb-preservation diagnostic."
        ),
    }

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS_DIR / "training_stats.json"
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[DIAGNOSTIC] training-set SKILL distribution: {short}/{total} spans "
        f"have < {VERB_PRESERVATION_MIN_TOKENS} tokens ({rate:.1%})."
    )
    if short_examples:
        print(f"[DIAGNOSTIC] sample short SKILL spans: {short_examples[:10]}")
    print(
        "[DIAGNOSTIC] These are SkillSpan's annotated soft-skill spans "
        "(personality / behavioral traits). NOT a bug. eval.py will compare "
        "the model's short-SKILL rate against this baseline."
    )
    print(f"[INFO] wrote training stats to {out_path}")


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_skillspan(DATA_DIR)

    for split in ("train", "dev", "test"):
        if split not in raw:
            continue
        records = [to_chat(ex) for ex in raw[split]]
        out_path = DATASETS_DIR / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                # Drop the _meta sidecar before serialising so HF datasets
                # only sees the canonical "messages" key.
                f.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\n")
        print(f"[INFO] wrote {len(records)} chat records to {out_path}")

        if split == "train":
            write_training_stats(records)


if __name__ == "__main__":
    main()
