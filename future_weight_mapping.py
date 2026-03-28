"""
future_weight_mapping.py

Map extracted knowledge items OR skills to future job domains (e.g., WEF/McKinsey-style)
and compute a "future_weight" score for each item.

Scientific methods (see SCIENTIFIC_METHODOLOGY.md §6–7):
    - future_weight = similarity(item, best_domain) × trend_score
    - mapping_margin = top1_sim - top2_sim (uncertainty)
    - Normalization for grouping: lowercase, collapse punctuation

Similarity and mapping_margin are used by generate_competencies.py for domain-based
batching: high confidence (similarity >= 0.45, margin >= 0.05) keeps best_future_domain;
low confidence goes to "Uncertain" batch.

Modes:
  - knowledge (default): advanced_knowledge.csv -> future_skill_weights_dummy.csv
  - skills: advanced_skills.csv -> future_skill_weights.csv (for competency generation)

Inputs:
    - config.OUTPUT_DIR / advanced_knowledge.csv or advanced_skills.csv
    - future_domains.csv or future_domains_dummy.csv

Outputs:
    - future_skill_weights_dummy.csv (knowledge mode)
    - future_skill_weights.csv (skills mode; includes similarity, mapping_margin)
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

import config  # uses config.OUTPUT_DIR
from pipeline import AdvancedPipelineConfig  # reuse embedding model name


# ------------------------ helpers ------------------------ #

def _normalize_for_grouping(text: str) -> str:
    """
    Deterministic normalization for grouping equivalent skills/knowledge.
    Used to merge case variants (e.g. 'AI' vs 'ai') and punctuation variants
    (e.g. 'AI/ML' vs 'AI ML'). Scientifically rigorous: reproducible, documented.
    """
    if not text or not isinstance(text, str):
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"[\s_/|,;:.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_knowledge_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file not found: {path}")

    df = pd.read_csv(path)
    if "knowledge" not in df.columns:
        raise ValueError(f"{path} must contain a 'knowledge' column.")

    # Basic cleaning
    df["knowledge"] = df["knowledge"].astype(str).str.strip()
    df = df[df["knowledge"] != ""].copy()
    return df


def load_skills_df(path: Path) -> pd.DataFrame:
    """Load skills for future weight mapping. Expects 'skill' column."""
    if not path.exists():
        raise FileNotFoundError(f"Skills file not found: {path}")

    df = pd.read_csv(path)
    if "skill" not in df.columns:
        raise ValueError(f"{path} must contain a 'skill' column.")

    df["skill"] = df["skill"].astype(str).str.strip()
    df = df[df["skill"] != ""].copy()
    return df


def load_future_domains(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Future domains file not found: {path}. "
            f"Expected dummy file like future_domains_dummy.csv"
        )

    df = pd.read_csv(path)

    required = {"domain_id", "future_domain", "example_terms",
                "trend_label", "trend_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Future domains file must contain columns: {missing}"
        )

    # Ensure numeric
    df["trend_score"] = pd.to_numeric(df["trend_score"], errors="coerce").fillna(0.0)

    # Add ordinal trend_tier if not already present (derived from trend_score).
    _TIER_WEIGHT_MAP = {
        "Strong_Growth": 1.0,
        "Moderate_Growth": 0.7,
        "Stable": 0.4,
        "Declining": 0.0,
    }
    if "trend_tier" not in df.columns:
        def _score_to_tier(s: float) -> str:
            if s >= 0.8: return "Strong_Growth"
            elif s >= 0.5: return "Moderate_Growth"
            elif s >= 0.3: return "Stable"
            else: return "Declining"
        df["trend_tier"] = df["trend_score"].apply(_score_to_tier)
    df["tier_weight"] = df["trend_tier"].map(_TIER_WEIGHT_MAP).fillna(0.0)

    # Build combined text for embedding (domain name + examples)
    df["domain_text"] = (
        df["future_domain"].astype(str).str.strip()
        + ". "
        + df["example_terms"].astype(str).str.strip()
    )

    return df


def compute_embeddings(texts, model_name: str) -> np.ndarray:
    """Encode a list/Series of texts into embeddings using SentenceTransformer."""
    print(f"[INFO] Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print("[INFO] Computing embeddings...")
    emb = embedder.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between
    a: (n_a, d) and b: (n_b, d)
    assuming both are L2-normalized.
    """
    return np.matmul(a, b.T)


# ------------------------ main logic ------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Map extracted knowledge items or skills to future job domains "
                    "and compute future_weight scores."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Output directory (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["knowledge", "skills"],
        default="knowledge",
        help="Input type: knowledge (default) or skills",
    )
    parser.add_argument(
        "--knowledge_file",
        type=str,
        default="advanced_knowledge.csv",
        help="Knowledge CSV file name when input_type=knowledge",
    )
    parser.add_argument(
        "--skills_file",
        type=str,
        default="advanced_skills.csv",
        help="Skills CSV file name when input_type=skills",
    )
    parser.add_argument(
        "--future_domains_file",
        type=str,
        default="future_domains.csv",
        help="Future domains CSV file (default: future_domains.csv)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output CSV file (default: future_skill_weights_dummy.csv for knowledge, "
             "future_skill_weights.csv for skills)",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="Minimum frequency to include (default: 1)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Max number of items to embed (for quick testing). Default: None (use all).",
    )
    parser.add_argument(
        "--similarity_floor",
        type=float,
        default=0.3,
        help="Minimum similarity for valid domain assignment. Below this, "
             "future_weight is set to 0 (default: 0.3).",
    )
    parser.add_argument(
        "--spektrum-code",
        type=str,
        default=None,
        help="Spektrum Keahlian code (e.g. 4.1.1) to filter future_domains via spektrum_mapping.csv. "
             "When set, only primary_domain_ids for this code are used.",
    )
    parser.add_argument(
        "--spektrum-mapping-file",
        type=str,
        default=None,
        help="Path to spektrum_mapping.csv (default: data/spektrum_keahlian/spektrum_mapping.csv)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    domains_path = Path(args.future_domains_file)

    # Resolve output path
    if args.output_file:
        output_path = out_dir / args.output_file
    else:
        output_path = (
            out_dir / "future_skill_weights.csv"
            if args.input_type == "skills"
            else out_dir / "future_skill_weights_dummy.csv"
        )

    item_col = "skill" if args.input_type == "skills" else "knowledge"

    def _aggregate_with_canonical(df, item_col):
        df = df.copy()
        df["_group_key"] = df[item_col].apply(_normalize_for_grouping)
        df = df[df["_group_key"] != ""]
        agg_dict = {"freq": ("_group_key", "count")}
        if "confidence_score" in df.columns:
            agg_dict["mean_confidence"] = ("confidence_score", "mean")
        else:
            agg_dict["mean_confidence"] = ("_group_key", "count")
        grp = df.groupby("_group_key").agg(**agg_dict).reset_index()
        canonical = df.groupby("_group_key")[item_col].apply(
            lambda x: x.value_counts().index[0]
        ).reset_index()
        canonical.columns = ["_group_key", item_col]
        grp = grp.merge(canonical, on="_group_key").drop(columns=["_group_key"])
        return grp

    if args.input_type == "skills":
        skills_path = out_dir / args.skills_file
        print(f"[INFO] Using OUTPUT_DIR = {out_dir}")
        print(f"[INFO] Reading skills from {skills_path}")
        skills_df = load_skills_df(skills_path)
        grp = _aggregate_with_canonical(skills_df, "skill")
    else:
        knowledge_path = out_dir / args.knowledge_file
        print(f"[INFO] Using OUTPUT_DIR = {out_dir}")
        print(f"[INFO] Reading knowledge from {knowledge_path}")
        knowledge_df = load_knowledge_df(knowledge_path)
        grp = _aggregate_with_canonical(knowledge_df, "knowledge")

    print(f"[INFO] Unique {args.input_type} items (after grouping) before freq filter: {len(grp)}")
    grp = grp[grp["freq"] >= args.min_freq].copy()
    print(f"[INFO] Unique {args.input_type} items after freq >= {args.min_freq}: {len(grp)}")

    if args.max_items is not None and len(grp) > args.max_items:
        grp = grp.nlargest(args.max_items, "freq").copy()
        print(f"[INFO] Limiting to top {args.max_items} items by freq.")

    if grp.empty:
        print(f"[WARN] No {args.input_type} items after filtering. Exiting.")
        return

    # Load future domains
    print(f"[INFO] Reading future domains from {domains_path}")
    domains_df = load_future_domains(domains_path)

    # Filter by Spektrum code when provided (department-scoped domain selection)
    if args.spektrum_code and str(args.spektrum_code).strip():
        if args.spektrum_mapping_file:
            mapping_path = Path(args.spektrum_mapping_file)
        else:
            base = Path(config.PROJECT_ROOT)
            mapping_path = base / "data" / "spektrum_keahlian" / "spektrum_mapping.csv"
            if not mapping_path.exists():
                mapping_path = base / "DATA" / "spektrum_keahlian" / "spektrum_mapping.csv"
        if mapping_path.exists():
            mapping_df = pd.read_csv(mapping_path)
            code = str(args.spektrum_code).strip()
            row = mapping_df[mapping_df["spektrum_code"].astype(str).str.strip() == code]
            if not row.empty:
                primary = str(row.iloc[0].get("primary_domain_ids", "")).strip()
                domain_ids = [d.strip() for d in primary.split(";") if d.strip()]
                if domain_ids:
                    domains_df = domains_df[domains_df["domain_id"].astype(str).str.strip().isin(domain_ids)].copy()
                    print(f"[INFO] Filtered to {len(domains_df)} domains for Spektrum {code}")
            else:
                fallback = mapping_df[mapping_df["spektrum_code"].astype(str).str.strip() == "*"]
                if not fallback.empty:
                    primary = str(fallback.iloc[0].get("primary_domain_ids", "")).strip()
                    domain_ids = [d.strip() for d in primary.split(";") if d.strip()]
                    if domain_ids:
                        domains_df = domains_df[domains_df["domain_id"].astype(str).str.strip().isin(domain_ids)].copy()
                        print(f"[INFO] Spektrum {code} not in mapping; using * fallback ({len(domains_df)} domains)")
        else:
            print(f"[WARN] Spektrum mapping not found at {mapping_path}; using all domains")
    print(f"[INFO] Number of future domains: {len(domains_df)}")

    # Embeddings
    model_name = AdvancedPipelineConfig.EMBEDDING_MODEL
    item_texts = grp[item_col].tolist()
    domain_texts = domains_df["domain_text"].tolist()

    item_emb = compute_embeddings(item_texts, model_name)
    domain_emb = compute_embeddings(domain_texts, model_name)

    sim_mat = cosine_sim_matrix(item_emb, domain_emb)
    best_domain_idx = np.argmax(sim_mat, axis=1)
    best_sim = sim_mat[np.arange(len(item_texts)), best_domain_idx]

    # Mapping uncertainty: margin between top-1 and top-2 similarity
    if sim_mat.shape[1] >= 2:
        sorted_sims = np.sort(sim_mat, axis=1)[:, ::-1]
        top1_sim = sorted_sims[:, 0]
        top2_sim = sorted_sims[:, 1]
        mapping_margin = top1_sim - top2_sim
    else:
        top1_sim = best_sim
        top2_sim = np.zeros_like(best_sim)
        mapping_margin = top1_sim

    # Top-2 and Top-3 domain IDs and similarities (argsort ascending: [:, -1]=max, [:, -2]=2nd, [:, -3]=3rd)
    sorted_idx = np.argsort(sim_mat, axis=1)
    if sim_mat.shape[1] >= 2:
        top2_domain_ids = domains_df["domain_id"].iloc[sorted_idx[:, -2]].values
    else:
        top2_domain_ids = np.array([""] * len(item_texts))
    if sim_mat.shape[1] >= 3:
        top3_domain_ids = domains_df["domain_id"].iloc[sorted_idx[:, -3]].values
        top3_sim = np.sort(sim_mat, axis=1)[:, ::-1][:, 2]
    else:
        top3_domain_ids = np.array([""] * len(item_texts))
        top3_sim = np.zeros_like(best_sim)

    matched_domains = domains_df.iloc[best_domain_idx].reset_index(drop=True)

    result_df = pd.DataFrame(
        {
            item_col: grp[item_col].values,
            "freq": grp["freq"].values,
            "mean_confidence": grp["mean_confidence"].values,
            "best_domain_id": matched_domains["domain_id"].values,
            "best_future_domain": matched_domains["future_domain"].values,
            "trend_label": matched_domains["trend_label"].values,
            "trend_score": matched_domains["trend_score"].values,
            "similarity": best_sim,
            "top1_similarity": top1_sim,
            "top2_similarity": top2_sim,
            "top3_similarity": top3_sim,
            "mapping_margin": mapping_margin,
            "top2_domain_id": top2_domain_ids,
            "top3_domain_id": top3_domain_ids,
        }
    )

    if "source" in matched_domains.columns:
        result_df["domain_source"] = matched_domains["source"].values

    # Zero out future_weight for items below similarity floor
    raw_fw = result_df["similarity"] * result_df["trend_score"]
    below_floor = result_df["similarity"] < args.similarity_floor
    raw_fw[below_floor] = 0.0
    result_df["future_weight"] = raw_fw
    if below_floor.any():
        print(f"[INFO] {below_floor.sum()}/{len(result_df)} items below "
              f"similarity floor ({args.similarity_floor}); future_weight set to 0")

    # Add trend_tier and tier-based future_weight (ordinal alternative to continuous trend_score).
    # future_weight_tier uses tier_weight (1.0/0.7/0.4/0.0) instead of raw trend_score.
    # This addresses the ordinal-scale concern (trend_scores are expert judgements, not
    # interval measurements). See SCIENTIFIC_METHODOLOGY.md §7.
    result_df["trend_tier"] = matched_domains["trend_tier"].values
    result_df["tier_weight"] = matched_domains["tier_weight"].values
    raw_tier_fw = result_df["similarity"] * result_df["tier_weight"]
    raw_tier_fw = raw_tier_fw.copy()
    raw_tier_fw[below_floor] = 0.0
    result_df["future_weight_tier"] = raw_tier_fw

    result_df = result_df.sort_values("future_weight", ascending=False)

    print(f"[INFO] Mapping margin stats: "
          f"mean={mapping_margin.mean():.3f}, "
          f"median={np.median(mapping_margin):.3f}, "
          f"min={mapping_margin.min():.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved future weights to {output_path}")

    # Sensitivity: compare top-20 by continuous future_weight vs tier-based future_weight_tier.
    # A high Jaccard indicates the ranking is robust to the ordinal-vs-interval distinction.
    try:
        import json as _json
        top20_continuous = set(
            result_df.nlargest(20, "future_weight")[item_col].tolist()
        )
        top20_tier = set(
            result_df.nlargest(20, "future_weight_tier")[item_col].tolist()
        )
        union_t = top20_continuous | top20_tier
        inter_t = top20_continuous & top20_tier
        jaccard_tier = round(len(inter_t) / len(union_t), 4) if union_t else 1.0
        print(f"[INFO] Top-20 Jaccard (continuous vs tier-based future_weight): {jaccard_tier:.4f}")
        sensitivity_tier = {
            "top20_continuous": sorted(top20_continuous),
            "top20_tier": sorted(top20_tier),
            "jaccard_top20_continuous_vs_tier": jaccard_tier,
            "interpretation": (
                "Jaccard >= 0.80 indicates ranking is robust to ordinal-vs-interval distinction."
                if jaccard_tier >= 0.80 else
                "Jaccard < 0.80: tier-based ranking differs from continuous; review top items."
            ),
        }
        sens_path = out_dir / "future_weight_tier_sensitivity.json"
        sens_path.write_text(_json.dumps(sensitivity_tier, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Saved tier sensitivity to {sens_path}")
    except Exception as e:
        print(f"[WARN] Could not compute tier sensitivity: {e}")

    # Plots only for knowledge mode (skills output is consumed by generate_competencies)
    if args.input_type == "knowledge":
        try:
            plt.figure(figsize=(8, 5))
            plt.hist(result_df["future_weight"].dropna(), bins=30)
            plt.xlabel("future_weight")
            plt.ylabel("Count")
            plt.title("Distribution of future_weight for knowledge items")
            plt.tight_layout()
            plt.savefig(out_dir / "future_weight_histogram.png", dpi=300)
            plt.close()
            print(f"[INFO] Saved future_weight_histogram.png")

            # Dedupe by normalized key (merge AI/ML, AI / ML:, etc.)
            r = result_df.copy()
            r["_key"] = r["knowledge"].apply(_normalize_for_grouping)
            r = r[r["_key"] != ""]
            r = r.loc[r.groupby("_key")["future_weight"].idxmax()].drop(columns=["_key"])
            r = r.sort_values("future_weight", ascending=False)
            # Top 20: highest future_weight (exclude zeros so we show meaningful items)
            r_positive = r[r["future_weight"] > 0]
            top_n = r_positive.head(20) if len(r_positive) >= 20 else r_positive
            if top_n.empty:
                top_n = r.head(20)
            plt.figure(figsize=(10, 6))
            plt.barh(top_n["knowledge"], top_n["future_weight"])
            plt.xlabel("future_weight")
            plt.ylabel("Knowledge")
            plt.title("Top 20 Knowledge Items by future_weight")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(out_dir / "top_future_weight_knowledge.png", dpi=300)
            plt.close()
            print(f"[INFO] Saved top_future_weight_knowledge.png")

            bottom_n = r.tail(20).sort_values("future_weight")
            plt.figure(figsize=(10, 6))
            plt.barh(bottom_n["knowledge"], bottom_n["future_weight"])
            plt.xlabel("future_weight")
            plt.ylabel("Knowledge")
            plt.title("Bottom 20 Knowledge Items by future_weight")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(out_dir / "bottom_future_weight_knowledge.png", dpi=300)
            plt.close()
            print(f"[INFO] Saved bottom_future_weight_knowledge.png")
        except Exception as e:
            print(f"[WARN] Could not create plots: {e}")


if __name__ == "__main__":
    main()
