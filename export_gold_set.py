"""
export_gold_set.py

Exports stratified samples for gold-set labeling (ground truth).

Produces:
    DATA/labels/gold_skills.csv      — skills to label (150 default)
    DATA/labels/gold_knowledge.csv   — knowledge to label (100 default)
    DATA/labels/gold_future_domain.csv — items for domain-mapping validation (100 default)

Stratification ensures representation across confidence tiers, extraction
sources, and skill types so evaluation results generalise across the
pipeline's output distribution.

Overlap items for IRR are selected via stratified random sampling (not
first-N) to avoid order bias.  The overlap set is tagged in a separate
column so the labeling UI can highlight them.
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

import config

LABELS_DIR = Path(config.PROJECT_ROOT) / "DATA" / "labels"


def _stratified_sample(df: pd.DataFrame, n: int, strata_cols: list,
                       seed: int = 42) -> pd.DataFrame:
    usable = [c for c in strata_cols if c in df.columns]
    if not usable or len(df) <= n:
        return df.sample(n=min(n, len(df)), random_state=seed)
    total = len(df)
    parts = []
    for _, grp in df.groupby(usable, dropna=False):
        alloc = max(1, round(n * len(grp) / total))
        parts.append(grp.sample(n=min(alloc, len(grp)), random_state=seed))
    result = pd.concat(parts).reset_index(drop=True)
    if len(result) > n:
        result = result.sample(n=n, random_state=seed)
    return result


def _make_id(prefix: str, *parts) -> str:
    raw = "|".join(str(p) for p in parts)
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _tag_overlap(df: pd.DataFrame, overlap_n: int, seed: int) -> pd.DataFrame:
    """Tag overlap items for IRR via stratified random sampling (not first-N)."""
    if len(df) <= overlap_n:
        df["is_overlap"] = True
        return df
    overlap_idx = df.sample(n=overlap_n, random_state=seed).index
    df["is_overlap"] = df.index.isin(overlap_idx)
    return df


def export_gold_skills(output_dir: Path, n: int, seed: int,
                       overlap_n: int = 30) -> pd.DataFrame:
    for name in ["verified_skills.csv", "advanced_skills.csv"]:
        path = output_dir / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"[INFO] Loaded {len(df)} skills from {name}")
            break
    else:
        raise FileNotFoundError("No skills CSV found in output dir")

    strata = ["confidence_tier", "source", "type"]
    sampled = _stratified_sample(df, n, strata, seed)

    keep = [c for c in [
        "job_id", "skill", "type", "confidence_score",
        "confidence_tier", "source", "semantic_density", "context_agreement",
    ] if c in sampled.columns]
    out = sampled[keep].copy()

    out.insert(0, "gold_id", out.apply(
        lambda r: _make_id("gs", r.get("job_id", ""), r.get("skill", "")),
        axis=1,
    ))

    out = _tag_overlap(out, overlap_n, seed)

    out["is_correct"] = ""
    out["type_label"] = ""
    out["labeler_id"] = ""
    out["notes"] = ""

    return out


def export_gold_knowledge(output_dir: Path, n: int, seed: int,
                          overlap_n: int = 30) -> pd.DataFrame:
    for name in ["advanced_knowledge.csv"]:
        path = output_dir / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"[INFO] Loaded {len(df)} knowledge from {name}")
            break
    else:
        raise FileNotFoundError("No knowledge CSV found in output dir")

    strata = ["confidence_tier", "source"]
    sampled = _stratified_sample(df, n, strata, seed)

    keep = [c for c in [
        "job_id", "knowledge", "confidence_score", "confidence_tier", "source",
    ] if c in sampled.columns]
    out = sampled[keep].copy()

    out.insert(0, "gold_id", out.apply(
        lambda r: _make_id("gk", r.get("job_id", ""), r.get("knowledge", "")),
        axis=1,
    ))

    out = _tag_overlap(out, overlap_n, seed)

    out["is_correct"] = ""
    out["labeler_id"] = ""
    out["notes"] = ""

    return out


def export_gold_future_domain(output_dir: Path, n: int, seed: int) -> pd.DataFrame:
    rows = []
    for name, col, prefix in [
        ("future_skill_weights.csv", "skill", "gfs"),
        ("future_skill_weights_dummy.csv", "knowledge", "gfk"),
    ]:
        path = output_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if col not in df.columns:
            continue
        df = df.drop_duplicates(subset=[col]).copy()

        # Add margin_bin for stratification (same approach as skills/knowledge)
        margin = pd.to_numeric(df.get("mapping_margin", 0), errors="coerce").fillna(0)
        df = df.copy()
        df["margin_bin"] = margin.apply(
            lambda m: "High" if m >= 0.10
            else "Medium" if m >= 0.05
            else "Low"
        )

        half = max(10, n // 2)
        strata = ["margin_bin", "best_future_domain"]
        usable = [c for c in strata if c in df.columns]
        if usable:
            sampled = _stratified_sample(df, half, usable, seed)
        else:
            sampled = df.sample(n=min(half, len(df)), random_state=seed)

        for _, r in sampled.iterrows():
            rows.append({
                "gold_id": _make_id(prefix, r[col]),
                "item_text": r[col],
                "item_type": "skill" if col == "skill" else "knowledge",
                "pipeline_domain": r.get("best_future_domain", ""),
                "pipeline_similarity": round(float(r.get("similarity", 0)), 3),
                "true_domain_id": "",
                "labeler_id": "",
                "notes": "",
            })

    if not rows:
        raise FileNotFoundError("No future_skill_weights CSVs found")

    out = pd.DataFrame(rows)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Export stratified samples for gold-set labeling."
    )
    parser.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    parser.add_argument("--labels_dir", type=str, default=str(LABELS_DIR))
    parser.add_argument("--n_skills", type=int, default=150)
    parser.add_argument("--n_knowledge", type=int, default=100)
    parser.add_argument("--n_domain", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap_n", type=int, default=30,
                        help="Number of randomly-sampled overlap items for IRR (default: 30)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    labels = Path(args.labels_dir)
    labels.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Labels dir: {labels}")

    skills = export_gold_skills(out_dir, args.n_skills, args.seed,
                                overlap_n=args.overlap_n)
    p = labels / "gold_skills.csv"
    skills.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[INFO] Exported {len(skills)} skills to {p}")

    knowledge = export_gold_knowledge(out_dir, args.n_knowledge, args.seed,
                                      overlap_n=args.overlap_n)
    p = labels / "gold_knowledge.csv"
    knowledge.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[INFO] Exported {len(knowledge)} knowledge to {p}")

    try:
        domain = export_gold_future_domain(out_dir, args.n_domain, args.seed)
        p = labels / "gold_future_domain.csv"
        domain.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[INFO] Exported {len(domain)} domain items to {p}")
    except FileNotFoundError as e:
        print(f"[WARN] Skipping domain export: {e}")

    print("[INFO] Gold set export complete. Fill in label columns and save.")


if __name__ == "__main__":
    main()
