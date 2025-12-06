"""
future_weight_mapping.py

Map extracted knowledge items to future job domains (e.g., WEF/McKinsey-style)
and compute a "future_weight" score for each knowledge item.

future_weight = cosine_similarity(knowledge, domain_example_terms) * trend_score

Inputs:
    - config.OUTPUT_DIR / advanced_knowledge.csv
      (or any compatible CSV with a 'knowledge' column)
    - future_domains_dummy.csv
      (dummy future job domains with trend_score)

Outputs:
    - future_skill_weights_dummy.csv (in OUTPUT_DIR)
    - optional plots:
        * future_weight_histogram.png
        * top_future_weight_knowledge.png
        * bottom_future_weight_knowledge.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

import config  # uses config.OUTPUT_DIR
from pipeline import AdvancedPipelineConfig  # reuse embedding model name


# ------------------------ helpers ------------------------ #

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
        description="Map extracted knowledge items to future job domains "
                    "and compute future_weight scores."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing advanced_knowledge.csv (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--knowledge_file",
        type=str,
        default="advanced_knowledge.csv",
        help="Knowledge CSV file name (default: advanced_knowledge.csv)",
    )
    parser.add_argument(
        "--future_domains_file",
        type=str,
        default="future_domains_dummy.csv",
        help="Future domains CSV file (default: future_domains_dummy.csv)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="future_skill_weights_dummy.csv",
        help="Output CSV file (default: future_skill_weights_dummy.csv)",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=1,
        help="Minimum frequency of a knowledge item to include (default: 1)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Max number of knowledge items to embed (for quick testing). "
             "Default: None (use all).",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    knowledge_path = out_dir / args.knowledge_file
    domains_path = Path(args.future_domains_file)
    output_path = out_dir / args.output_file

    print(f"[INFO] Using OUTPUT_DIR = {out_dir}")
    print(f"[INFO] Reading knowledge from {knowledge_path}")
    knowledge_df = load_knowledge_df(knowledge_path)

    # Aggregate knowledge by phrase
    grp = knowledge_df.groupby("knowledge").agg(
        freq=("knowledge", "count"),
        mean_confidence=("confidence_score", "mean")
        if "confidence_score" in knowledge_df.columns
        else ("knowledge", "count"),  # fallback dummy
    ).reset_index()

    print(f"[INFO] Unique knowledge items before freq filter: {len(grp)}")
    grp = grp[grp["freq"] >= args.min_freq].copy()
    print(f"[INFO] Unique knowledge items after freq >= {args.min_freq}: {len(grp)}")

    # Optional limit for quick testing
    if args.max_items is not None and len(grp) > args.max_items:
        grp = grp.nlargest(args.max_items, "freq").copy()
        print(f"[INFO] Limiting to top {args.max_items} knowledge items by freq.")

    if grp.empty:
        print("[WARN] No knowledge items after filtering. Exiting.")
        return

    # Load future domains
    print(f"[INFO] Reading future domains from {domains_path}")
    domains_df = load_future_domains(domains_path)
    print(f"[INFO] Number of future domains: {len(domains_df)}")

    # Embeddings
    model_name = AdvancedPipelineConfig.EMBEDDING_MODEL
    knowledge_texts = grp["knowledge"].tolist()
    domain_texts = domains_df["domain_text"].tolist()

    knowledge_emb = compute_embeddings(knowledge_texts, model_name)
    domain_emb = compute_embeddings(domain_texts, model_name)

    # Cosine similarity matrix: (n_knowledge, n_domains)
    sim_mat = cosine_sim_matrix(knowledge_emb, domain_emb)

    # For each knowledge, find best matching domain
    best_domain_idx = np.argmax(sim_mat, axis=1)
    best_sim = sim_mat[np.arange(len(knowledge_texts)), best_domain_idx]

    # Attach domain info
    matched_domains = domains_df.iloc[best_domain_idx].reset_index(drop=True)

    result_df = pd.DataFrame(
        {
            "knowledge": grp["knowledge"].values,
            "freq": grp["freq"].values,
            "mean_confidence": grp["mean_confidence"].values,
            "best_domain_id": matched_domains["domain_id"].values,
            "best_future_domain": matched_domains["future_domain"].values,
            "trend_label": matched_domains["trend_label"].values,
            "trend_score": matched_domains["trend_score"].values,
            "similarity": best_sim,
        }
    )

    # Compute future_weight
    result_df["future_weight"] = result_df["similarity"] * result_df["trend_score"]

    # Sort for inspection
    result_df = result_df.sort_values("future_weight", ascending=False)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved future weights to {output_path}")

    # Simple diagnostic plots (optional)
    try:
        # 1) Histogram of future_weight
        plt.figure(figsize=(8, 5))
        plt.hist(result_df["future_weight"].dropna(), bins=30)
        plt.xlabel("future_weight")
        plt.ylabel("Count")
        plt.title("Distribution of future_weight for knowledge items")
        plt.tight_layout()
        plt.savefig(out_dir / "future_weight_histogram.png", dpi=300)
        plt.close()
        print(f"[INFO] Saved future_weight_histogram.png")

        # 2) Top 20 by future_weight
        top_n = result_df.head(20)
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

        # 3) Bottom 20 (most negative)
        bottom_n = result_df.tail(20).sort_values("future_weight")
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
