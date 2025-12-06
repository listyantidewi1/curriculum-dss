"""
advanced_visualizations.py

Visual analytics for the Advanced Skill Extraction Pipeline.

Usage:
    python advanced_visualizations.py --output_dir output

Assumes the following files exist in output_dir:
    - advanced_skills.csv
    - advanced_knowledge.csv
    - coverage_report.csv
    - comprehensive_analysis.csv
    - model_comparison.csv

Outputs:
    - Multiple PNG plots into <output_dir>/figures/
    - Text report: <output_dir>/advanced_metrics_report.txt
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn is optional but recommended for nicer plots
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:  # fallback to plain matplotlib
    sns = None

# Embeddings & clustering
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Your project config (for curriculum, etc.)
try:
    from config import CURRICULUM_COMPONENTS, OUTPUT_DIR as CONFIG_OUTPUT_DIR
except ImportError:
    CURRICULUM_COMPONENTS = {}
    CONFIG_OUTPUT_DIR = "results"


# -------------------------------------------------------------------
# Global constants
# -------------------------------------------------------------------

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "N/A"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # keep in sync with pipeline if you change it

FIG_DIR_NAME = "figures"


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_model_name(name: str) -> str:
    """
    Normalize model labels to something consistent for plotting if needed.
    """
    if not isinstance(name, str):
        return str(name)
    name_lower = name.lower()
    if "bert" in name_lower and "gpt" not in name_lower and "hybrid" not in name_lower:
        return "JobBERT"
    if "gpt" in name_lower or "deepseek" in name_lower:
        return "GPT"
    if "hybrid" in name_lower or "fusion" in name_lower:
        return "Hybrid"
    return name


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_all_data(output_dir: Path):
    paths = {
        "advanced_skills": output_dir / "advanced_skills.csv",
        "advanced_knowledge": output_dir / "advanced_knowledge.csv",
        "coverage_report": output_dir / "coverage_report.csv",
        "comprehensive_analysis": output_dir / "comprehensive_analysis.csv",
        "model_comparison": output_dir / "model_comparison.csv",
    }

    dfs = {name: safe_read_csv(path) for name, path in paths.items()}
    return dfs


# -------------------------------------------------------------------
# 1. JobBERT vs GPT vs Hybrid – skill & knowledge extraction
# -------------------------------------------------------------------

def plot_skill_knowledge_counts(model_comp: pd.DataFrame, fig_dir: Path):
    if model_comp.empty:
        print("[WARN] model_comparison.csv is empty; skipping skill/knowledge count plots.")
        return

    df = model_comp.copy()
    df["model_norm"] = df["model"].apply(normalize_model_name)

    agg = df.groupby("model_norm").agg(
        total_skills=("skill_count", "sum"),
        total_knowledge=("knowledge_count", "sum"),
        mean_skills=("skill_count", "mean"),
        mean_knowledge=("knowledge_count", "mean"),
        jobs=("job_id", "nunique"),
    ).reset_index()

    # Barplot: total skills & knowledge per model
    melted = agg.melt(
        id_vars=["model_norm"],
        value_vars=["total_skills", "total_knowledge"],
        var_name="metric",
        value_name="count",
    )

    plt.figure(figsize=(8, 5))
    if sns:
        ax = sns.barplot(data=melted, x="model_norm", y="count", hue="metric")
    else:
        for metric in ["total_skills", "total_knowledge"]:
            subset = melted[melted["metric"] == metric]
            plt.bar(subset["model_norm"], subset["count"], label=metric, alpha=0.7)
        ax = plt.gca()
    ax.set_xlabel("Model")
    ax.set_ylabel("Total Count (across all jobs)")
    ax.set_title("Total Skills & Knowledge Extracted per Model")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(fig_dir / "skills_knowledge_total_per_model.png", dpi=300)
    plt.close()

    # Barplot: mean per job
    melted_mean = agg.melt(
        id_vars=["model_norm"],
        value_vars=["mean_skills", "mean_knowledge"],
        var_name="metric",
        value_name="mean_count",
    )

    plt.figure(figsize=(8, 5))
    if sns:
        ax = sns.barplot(data=melted_mean, x="model_norm", y="mean_count", hue="metric")
    else:
        for metric in ["mean_skills", "mean_knowledge"]:
            subset = melted_mean[melted_mean["metric"] == metric]
            plt.bar(subset["model_norm"], subset["mean_count"], label=metric, alpha=0.7)
        ax = plt.gca()
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Count per Job")
    ax.set_title("Average Skills & Knowledge per Job")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(fig_dir / "skills_knowledge_mean_per_job.png", dpi=300)
    plt.close()

    return agg


# -------------------------------------------------------------------
# 2. Hybrid fusion improvement plots
# -------------------------------------------------------------------

def plot_hybrid_improvement(model_comp: pd.DataFrame, fig_dir: Path):
    if model_comp.empty:
        print("[WARN] model_comparison.csv is empty; skipping hybrid improvement plots.")
        return

    df = model_comp.copy()
    df["model_norm"] = df["model"].apply(normalize_model_name)

    # Pivot to have columns JobBERT, GPT, Hybrid for selected metrics
    pivot = df.pivot_table(
        index="job_id",
        columns="model_norm",
        values=["skill_count", "knowledge_count", "coverage_pct", "avg_confidence"],
    )

    # Only continue if we have Hybrid column
    if ("skill_count", "Hybrid") not in pivot.columns:
        print("[WARN] Hybrid model not found in model_comparison; skipping improvement plots.")
        return

    # Differences: Hybrid - JobBERT, Hybrid - GPT
    diffs = {}
    for base_model in ["JobBERT", "GPT"]:
        if ("skill_count", base_model) not in pivot.columns:
            continue
        diff_df = pd.DataFrame({
            f"skill_diff_{base_model}": pivot[("skill_count", "Hybrid")] - pivot[("skill_count", base_model)],
            f"knowledge_diff_{base_model}": pivot[("knowledge_count", "Hybrid")] - pivot[("knowledge_count", base_model)],
            f"coverage_diff_{base_model}": pivot[("coverage_pct", "Hybrid")] - pivot[("coverage_pct", base_model)],
            f"avg_conf_diff_{base_model}": pivot[("avg_confidence", "Hybrid")] - pivot[("avg_confidence", base_model)],
        })
        diffs[base_model] = diff_df

        # Histogram of coverage improvement
        plt.figure(figsize=(8, 5))
        if sns:
            sns.histplot(diff_df[f"coverage_diff_{base_model}"].dropna(), kde=True)
        else:
            plt.hist(diff_df[f"coverage_diff_{base_model}"].dropna(), bins=20, alpha=0.8)
        plt.axvline(0, color="red", linestyle="--", linewidth=1)
        plt.xlabel("Coverage Improvement (Hybrid - {})".format(base_model))
        plt.ylabel("Number of Jobs")
        plt.title(f"Coverage Improvement: Hybrid vs {base_model}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"coverage_improvement_hybrid_vs_{base_model.lower()}.png", dpi=300)
        plt.close()

    return diffs


# -------------------------------------------------------------------
# 3. Bloom taxonomy distribution (hard skills)
# -------------------------------------------------------------------

def plot_bloom_distribution(model_comp: pd.DataFrame, fig_dir: Path):
    if model_comp.empty:
        print("[WARN] model_comparison.csv is empty; skipping Bloom distribution plots.")
        return

    df = model_comp.copy()
    df["model_norm"] = df["model"].apply(normalize_model_name)

    bloom_cols = {
        "bloom_remember": "Remember",
        "bloom_understand": "Understand",
        "bloom_apply": "Apply",
        "bloom_analyze": "Analyze",
        "bloom_evaluate": "Evaluate",
        "bloom_create": "Create",
        "bloom_na": "N/A",
    }

    agg = df.groupby("model_norm")[list(bloom_cols.keys())].sum().reset_index()

    melted = agg.melt(
        id_vars=["model_norm"],
        value_vars=list(bloom_cols.keys()),
        var_name="bloom_raw",
        value_name="count",
    )
    melted["Bloom Level"] = melted["bloom_raw"].map(bloom_cols)
    melted["Bloom Level"] = pd.Categorical(melted["Bloom Level"], categories=BLOOM_ORDER, ordered=True)
    melted = melted.sort_values(["model_norm", "Bloom Level"])

    plt.figure(figsize=(10, 6))
    if sns:
        ax = sns.barplot(
            data=melted[melted["Bloom Level"] != "N/A"],
            x="Bloom Level",
            y="count",
            hue="model_norm",
        )
    else:
        for model_name, subset in melted[melted["Bloom Level"] != "N/A"].groupby("model_norm"):
            plt.bar(subset["Bloom Level"], subset["count"], alpha=0.7, label=model_name)
        ax = plt.gca()
    ax.set_xlabel("Bloom Level (Hard Skills)")
    ax.set_ylabel("Total Count (across all jobs)")
    ax.set_title("Bloom Taxonomy Distribution for Hard Skills (JobBERT vs GPT vs Hybrid)")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(fig_dir / "bloom_distribution_hard_skills.png", dpi=300)
    plt.close()

    return agg


# -------------------------------------------------------------------
# 4. Curriculum × Bloom heatmap (skills)
# -------------------------------------------------------------------

def build_curriculum_phrase_map():
    """
    Build phrase list per curriculum component from CURRICULUM_COMPONENTS.

    Expected shape (flexible):
        CURRICULUM_COMPONENTS = {
            "component_id": {
                "name": "...",
                "description": "...",
                "remember": [...],
                "understand": [...],
                ...
            },
            ...
        }
    """
    comp_phrases = {}
    for comp_id, data in CURRICULUM_COMPONENTS.items():
        phrases = []

        # Optional name/description
        if isinstance(data, dict):
            if "name" in data:
                phrases.append(str(data["name"]))
            if "description" in data:
                phrases.append(str(data["description"]))

            # Collect list-valued fields (Bloom phrases)
            for key, value in data.items():
                if isinstance(value, list):
                    phrases.extend([str(v) for v in value])
        else:
            phrases.append(str(data))

        phrases = [p for p in phrases if isinstance(p, str) and p.strip()]
        comp_phrases[comp_id] = list(sorted(set(phrases)))

    return comp_phrases


def map_items_to_components(skills_df: pd.DataFrame,
                            knowledge_df: pd.DataFrame,
                            embedder: SentenceTransformer,
                            similarity_threshold: float = 0.45):
    """
    Map skills & knowledge to curriculum components using embeddings + cosine similarity.

    Returns:
        mapping_df: DataFrame with columns:
            ['item_type', 'text', 'bloom', 'confidence_score',
             'component_id', 'similarity']
    """
    comp_phrases = build_curriculum_phrase_map()

    if not comp_phrases:
        print("[WARN] CURRICULUM_COMPONENTS is empty; skipping curriculum-based mappings.")
        return pd.DataFrame()

    # Precompute component embeddings
    comp_embeddings = {}
    for comp_id, phrases in comp_phrases.items():
        if not phrases:
            continue
        comp_embeddings[comp_id] = embedder.encode(phrases, convert_to_tensor=True)

    rows = []

    # Prepare skill items
    for _, row in skills_df.iterrows():
        text = str(row.get("skill", "")).strip()
        if not text:
            continue
        bloom = row.get("bloom", "N/A")
        conf = float(row.get("confidence_score", 0.0))
        rows.append(("skill", text, bloom, conf))

    # Prepare knowledge items (Bloom = 'N/A')
    for _, row in knowledge_df.iterrows():
        text = str(row.get("knowledge", "")).strip()
        if not text:
            continue
        bloom = "N/A"
        conf = float(row.get("confidence_score", 0.0))
        rows.append(("knowledge", text, bloom, conf))

    if not rows:
        print("[WARN] No skills/knowledge rows available for curriculum mapping.")
        return pd.DataFrame()

    texts = [r[1] for r in rows]
    embeddings = embedder.encode(texts, convert_to_tensor=True)

    mapping_records = []
    for idx, (item_type, text, bloom, conf) in enumerate(rows):
        emb = embeddings[idx]
        best_comp = None
        best_sim = 0.0

        for comp_id, comp_emb in comp_embeddings.items():
            sims = util.cos_sim(emb, comp_emb)  # shape [1, n_phrases]
            max_sim = float(sims.max())
            if max_sim > best_sim:
                best_sim = max_sim
                best_comp = comp_id

        if best_comp is not None and best_sim >= similarity_threshold:
            mapping_records.append({
                "item_type": item_type,
                "text": text,
                "bloom": bloom,
                "confidence_score": conf,
                "component_id": best_comp,
                "similarity": best_sim,
            })
        else:
            # not mapped (could be used later to detect demanded-but-not-covered)
            mapping_records.append({
                "item_type": item_type,
                "text": text,
                "bloom": bloom,
                "confidence_score": conf,
                "component_id": None,
                "similarity": best_sim,
            })

    mapping_df = pd.DataFrame(mapping_records)
    return mapping_df


def plot_curriculum_bloom_heatmap(mapping_df: pd.DataFrame, fig_dir: Path):
    if mapping_df.empty:
        print("[WARN] Mapping DF empty; skipping curriculum × Bloom heatmap.")
        return

    # Hard skills only, valid component, valid Bloom
    mask = (
        (mapping_df["item_type"] == "skill") &
        mapping_df["component_id"].notna() &
        (~mapping_df["bloom"].isna())
    )
    sub = mapping_df[mask].copy()
    if sub.empty:
        print("[WARN] No mapped hard skills for heatmap.")
        return

    sub["bloom"] = sub["bloom"].fillna("N/A")

    # Aggregate: weighted by confidence_score
    pivot = sub.pivot_table(
        index="component_id",
        columns="bloom",
        values="confidence_score",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Reindex columns to Bloom order (drop NA if it doesn't exist)
    cols = [b for b in BLOOM_ORDER if b in pivot.columns]
    pivot = pivot.reindex(columns=cols)

    plt.figure(figsize=(12, max(6, len(pivot) * 0.4)))
    if sns:
        ax = sns.heatmap(
            pivot,
            annot=False,
            cmap="viridis",
            linewidths=0.5,
        )
    else:
        ax = plt.gca()
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im)

    ax.set_xlabel("Bloom Level (Hard Skills)")
    ax.set_ylabel("Curriculum Component ID")
    ax.set_title("Curriculum × Bloom Heatmap (Weighted by Confidence)")
    plt.tight_layout()
    plt.savefig(fig_dir / "heatmap_curriculum_bloom_hard_skills.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------
# 5 & 6. Clustering plots with SentenceTransformers
# -------------------------------------------------------------------

def valid_knowledge_for_plot(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t or is_pure_punct(t):
        return False
    if is_education_or_language(t):
        return False
    return True


def cluster_and_plot_texts(texts, weights, fig_path: Path, title: str, n_clusters: int = 8):
    """
    Cluster texts using embeddings + KMeans, then project to 2D via PCA.
    """
    if not texts:
        print(f"[WARN] No texts for clustering: {title}")
        return

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(texts, convert_to_tensor=False)

    # PCA to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    # Choose number of clusters not exceeding number of samples
    k = min(n_clusters, len(texts))
    if k <= 1:
        labels = np.zeros(len(texts), dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

    plt.figure(figsize=(10, 7))
    if sns:
        scatter = sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels,
            size=weights,
            sizes=(40, 200),
            alpha=0.8,
            legend="brief",
        )
    else:
        scatter = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            s=np.array(weights) * 50,
            alpha=0.8,
        )

    for i, txt in enumerate(texts):
        plt.text(coords[i, 0], coords[i, 1], txt, fontsize=7, alpha=0.7)

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()


import re

EDUCATION_KEYWORDS = {
    "bachelor", "bachelor's", "masters", "master", "master's",
    "degree", "bsc", "msc", "phd"
}

LANGUAGE_KEYWORDS = {
    "english", "french", "german", "dutch", "spanish", "italian",
    "portuguese", "japanese", "korean", "chinese", "mandarin"
}

GENERIC_MODIFIERS = {
    "based", "scalable", "observability", "oriented", "hands",
    "detail", "problem", "issues", "issue"
}

SOFT_PATTERNS_IN_HARD = [
    "analytical skill", "analytical skills", "analytical",
    "problem solving", "problem-solving", "solving skills",
    "attention to detail", "detail oriented", "detail-oriented",
    "critical thinking", "team player",
    "detail oriented", "detail-oriented","discipline employees",
    "leadership skill", "leadership skills", "collaborate closely",
    "critical thinking", "team player", "continuous improvement","discipline"
]

def is_softish(text: str) -> bool:
    t = text.lower()
    return any(pat in t for pat in SOFT_PATTERNS_IN_HARD)


PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")

def is_pure_punct(text: str) -> bool:
    return bool(PUNCT_ONLY_RE.match(text.strip()))

def is_education_or_language(text: str) -> bool:
    t = text.lower()
    if any(k in t for k in EDUCATION_KEYWORDS):
        return True
    if t in LANGUAGE_KEYWORDS:
        return True
    return False

def valid_hard_skill_for_plot(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    if is_pure_punct(t):          # kills "/" etc.
        return False
    if is_education_or_language(t):
        return False

    words = t.split()
    # For *skills* we require at least verb + something, i.e. >= 2 tokens
    if len(words) < 2:
        return False

    # Drop items that are only generic modifiers ("problem", "based", etc.)
    if all(w.lower() in GENERIC_MODIFIERS for w in words):
        return False

    return True


def plot_top_skill_knowledge_clusters(adv_skills: pd.DataFrame,
                                      adv_knowledge: pd.DataFrame,
                                      mapping_df: pd.DataFrame,
                                      fig_dir: Path,
                                      top_n: int = 40):
    # Top HARD skills (by frequency, weighted by confidence)
# Top HARD skills (by frequency, weighted by confidence)
    if not adv_skills.empty:
        hard = adv_skills.loc[adv_skills["type"] == "Hard"].copy()
        # filter out '/', degrees, languages, 1-word verbs, generic nouns
        hard = hard.loc[hard["skill"].apply(valid_hard_skill_for_plot)]
        # also remove soft-ish phrases that leaked into hard skills
        hard = hard.loc[~hard["skill"].apply(is_softish)]

        if not hard.empty:
            grp = hard.groupby("skill").agg(
                freq=("skill", "count"),
                mean_conf=("confidence_score", "mean"),
            ).reset_index()
            grp["score"] = grp["freq"] * grp["mean_conf"]
            top_hard = grp.sort_values("score", ascending=False).head(top_n)
            cluster_and_plot_texts(
                top_hard["skill"].tolist(),
                top_hard["score"].tolist(),
                fig_dir / "clusters_top_hard_skills.png",
                "Clusters of Top Hard Skills (Hybrid)",
            )


        # Top SOFT skills
        soft = adv_skills[adv_skills["type"] == "Soft"].copy()
        if not soft.empty:
            grp = soft.groupby("skill").agg(
                freq=("skill", "count"),
                mean_conf=("confidence_score", "mean"),
            ).reset_index()
            grp["score"] = grp["freq"] * grp["mean_conf"]
            top_soft = grp.sort_values("score", ascending=False).head(top_n)
            cluster_and_plot_texts(
                top_soft["skill"].tolist(),
                top_soft["score"].tolist(),
                fig_dir / "clusters_top_soft_skills.png",
                "Clusters of Top Soft Skills (Hybrid)",
            )

    # Top knowledge items
    # Top knowledge items
    if not adv_knowledge.empty:
        # remove degrees, languages, pure punctuation, empty strings
        kn = adv_knowledge.loc[adv_knowledge["knowledge"].apply(valid_knowledge_for_plot)].copy()
        if not kn.empty:
            grp_k = kn.groupby("knowledge").agg(
                freq=("knowledge", "count"),
                mean_conf=("confidence_score", "mean"),
            ).reset_index()
            grp_k["score"] = grp_k["freq"] * grp_k["mean_conf"]
            top_k = grp_k.sort_values("score", ascending=False).head(top_n)
            cluster_and_plot_texts(
                top_k["knowledge"].tolist(),
                top_k["score"].tolist(),
                fig_dir / "clusters_top_knowledge.png",
                "Clusters of Top Knowledge Items (Hybrid)",
            )


    # Demanded but NOT covered by curriculum (component_id is None)
# Demanded but NOT covered by curriculum (component_id is None)
    if mapping_df is not None and not mapping_df.empty:
        # only HARD skills here – curriculum is ~99% hard skills
        not_mapped = mapping_df[
            (mapping_df["component_id"].isna()) &
            (mapping_df["item_type"] == "skill")
        ].copy()

        if not not_mapped.empty:
            # clean up the texts (no '/', no degrees/languages, no 1-word verbs)
            not_mapped = not_mapped.loc[not_mapped["text"].apply(valid_hard_skill_for_plot)]


            # extra filter: remove soft-ish phrases that slipped into hard skills
            not_mapped = not_mapped.loc[
                ~not_mapped["text"].apply(is_softish)
            ]
            
            if not not_mapped.empty:
                grp_nm = not_mapped.groupby("text").agg(
                    freq=("text", "count"),
                    mean_conf=("confidence_score", "mean"),
                ).reset_index()
                grp_nm["score"] = grp_nm["freq"] * grp_nm["mean_conf"]
                top_nm = grp_nm.sort_values("score", ascending=False).head(top_n)
                cluster_and_plot_texts(
                    top_nm["text"].tolist(),
                    top_nm["score"].tolist(),
                    fig_dir / "clusters_demanded_not_covered.png",
                    "Demanded Hard Skills NOT Mapped to Curriculum",
                )


# -------------------------------------------------------------------
# 7. Curriculum components with low demand
# -------------------------------------------------------------------

def plot_low_demand_components(mapping_df: pd.DataFrame, fig_dir: Path, top_k: int = 10):
    if mapping_df.empty:
        print("[WARN] Mapping DF empty; skipping low-demand curriculum plots.")
        return

    # Consider both skills & knowledge that *are mapped*
    mapped = mapping_df[mapping_df["component_id"].notna()].copy()
    if mapped.empty:
        print("[WARN] No mapped items; skipping low-demand curriculum plots.")
        return

    # Demand score per component = sum(confidence) * count
    comp_agg = mapped.groupby("component_id").agg(
        count=("text", "count"),
        total_conf=("confidence_score", "sum"),
    ).reset_index()
    comp_agg["demand_score"] = comp_agg["count"] * comp_agg["total_conf"]

    # Components sorted by demand (ascending)
    comp_agg_sorted = comp_agg.sort_values("demand_score", ascending=True)

    # Choose some lowest-demand components
    bottom = comp_agg_sorted.head(top_k).copy()

    # Map component_id → readable name if available
    def comp_label(cid):
        data = CURRICULUM_COMPONENTS.get(cid, {})
        if isinstance(data, dict) and "name" in data:
            return f"{cid}: {data['name']}"
        return cid

    bottom["label"] = bottom["component_id"].apply(comp_label)

    plt.figure(figsize=(10, max(5, len(bottom) * 0.4)))
    if sns:
        ax = sns.barplot(
            data=bottom,
            x="demand_score",
            y="label",
            orient="h",
        )
    else:
        ax = plt.gca()
        ax.barh(bottom["label"], bottom["demand_score"])
    ax.set_xlabel("Demand Score (count × sum(confidence))")
    ax.set_ylabel("Curriculum Component")
    ax.set_title("Curriculum Components with Lowest Demand in Job Market")
    plt.tight_layout()
    plt.savefig(fig_dir / "low_demand_curriculum_components.png", dpi=300)
    plt.close()


# -------------------------------------------------------------------
# Confidence distributions, coverage distributions, etc.
# -------------------------------------------------------------------

def plot_confidence_distributions(adv_skills: pd.DataFrame,
                                  adv_knowledge: pd.DataFrame,
                                  model_comp: pd.DataFrame,
                                  fig_dir: Path):
    # Hybrid skill/knowledge confidence distribution
    if not adv_skills.empty:
        plt.figure(figsize=(8, 5))
        if sns:
            sns.histplot(adv_skills["confidence_score"].dropna(), kde=True)
        else:
            plt.hist(adv_skills["confidence_score"].dropna(), bins=20, alpha=0.8)
        plt.xlabel("Confidence Score (Hybrid Skills)")
        plt.ylabel("Count")
        plt.title("Distribution of Hybrid Skill Confidence Scores")
        plt.tight_layout()
        plt.savefig(fig_dir / "conf_distribution_hybrid_skills.png", dpi=300)
        plt.close()

    if not adv_knowledge.empty:
        plt.figure(figsize=(8, 5))
        if sns:
            sns.histplot(adv_knowledge["confidence_score"].dropna(), kde=True)
        else:
            plt.hist(adv_knowledge["confidence_score"].dropna(), bins=20, alpha=0.8)
        plt.xlabel("Confidence Score (Hybrid Knowledge)")
        plt.ylabel("Count")
        plt.title("Distribution of Hybrid Knowledge Confidence Scores")
        plt.tight_layout()
        plt.savefig(fig_dir / "conf_distribution_hybrid_knowledge.png", dpi=300)
        plt.close()

    # avg_confidence per model from model_comparison
    if not model_comp.empty:
        df = model_comp.copy()
        df["model_norm"] = df["model"].apply(normalize_model_name)
        plt.figure(figsize=(8, 5))
        if sns:
            sns.boxplot(data=df, x="model_norm", y="avg_confidence")
        else:
            # simple boxplot for avg_confidence grouped by model
            groups = [g["avg_confidence"].dropna().values for _, g in df.groupby("model_norm")]
            labels = list(df.groupby("model_norm").groups.keys())
            plt.boxplot(groups, labels=labels)
        plt.xlabel("Model")
        plt.ylabel("Avg Confidence per Job")
        plt.title("Average Confidence per Job by Model")
        plt.tight_layout()
        plt.savefig(fig_dir / "avg_conf_per_job_by_model.png", dpi=300)
        plt.close()


def plot_coverage_distributions(coverage_report: pd.DataFrame,
                                model_comp: pd.DataFrame,
                                fig_dir: Path):
    # From coverage_report.csv (Hybrid coverage)
    if not coverage_report.empty:
        plt.figure(figsize=(8, 5))
        if sns:
            sns.histplot(coverage_report["coverage_percentage"].dropna(), kde=True)
        else:
            plt.hist(coverage_report["coverage_percentage"].dropna(), bins=20, alpha=0.8)
        plt.xlabel("Coverage Percentage (Hybrid)")
        plt.ylabel("Number of Jobs")
        plt.title("Distribution of Hybrid Curriculum Coverage")
        plt.tight_layout()
        plt.savefig(fig_dir / "coverage_distribution_hybrid.png", dpi=300)
        plt.close()

    # From model_comparison.csv (coverage per model)
    # NOTE: coverage is only defined for the Hybrid model in the current pipeline.
    if not model_comp.empty:
        df = model_comp.copy()
        df["model_norm"] = df["model"].apply(normalize_model_name)

        # keep ONLY Hybrid rows, since JobBERT/GPT don't have meaningful coverage
        df = df[df["model_norm"] == "Hybrid"].copy()

        # if everything is NaN, skip the plot
        if df["coverage_pct"].notna().any():
            plt.figure(figsize=(8, 5))
            if sns:
                sns.boxplot(data=df, x="model_norm", y="coverage_pct")
            else:
                groups = [df["coverage_pct"].dropna().values]
                labels = ["Hybrid"]
                plt.boxplot(groups, labels=labels)

            plt.xlabel("Model")
            plt.ylabel("Coverage Percentage")
            plt.title("Coverage Percentage per Job (Hybrid)")
            plt.tight_layout()
            plt.savefig(fig_dir / "coverage_distribution_by_model.png", dpi=300)
            plt.close()



# -------------------------------------------------------------------
# Text report
# -------------------------------------------------------------------

def write_text_report(agg_model: pd.DataFrame,
                      hybrid_diffs: dict,
                      bloom_agg: pd.DataFrame,
                      coverage_report: pd.DataFrame,
                      mapping_df: pd.DataFrame,
                      output_dir: Path):
    report_path = output_dir / "advanced_metrics_report.txt"
    lines = []

    lines.append("=" * 80)
    lines.append("ADVANCED METRICS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Model-level aggregation
    if agg_model is not None and not agg_model.empty:
        lines.append("1. Model-Level Extraction Summary")
        lines.append("--------------------------------")
        for _, row in agg_model.iterrows():
            lines.append(f"- Model: {row['model_norm']}")
            lines.append(f"    Jobs                     : {int(row['jobs'])}")
            lines.append(f"    Total skills extracted   : {int(row['total_skills'])}")
            lines.append(f"    Total knowledge extracted: {int(row['total_knowledge'])}")
            lines.append(f"    Mean skills per job      : {row['mean_skills']:.2f}")
            lines.append(f"    Mean knowledge per job   : {row['mean_knowledge']:.2f}")
            lines.append("")
        lines.append("")

    # Hybrid improvement
    if hybrid_diffs:
        lines.append("2. Hybrid Improvement (Hybrid - Base Model)")
        lines.append("-------------------------------------------")
        for base_model, diff_df in hybrid_diffs.items():
            lines.append(f"- Compared to {base_model}:")
            for metric in ["skill_diff", "knowledge_diff", "coverage_diff", "avg_conf_diff"]:
                col = f"{metric}_{base_model}"
                if col not in diff_df.columns:
                    continue
                mean_val = diff_df[col].mean()
                median_val = diff_df[col].median()
                lines.append(f"    {col}: mean={mean_val:.4f}, median={median_val:.4f}")
            lines.append("")
        lines.append("")

    # Bloom distribution
    if bloom_agg is not None and not bloom_agg.empty:
        lines.append("3. Bloom Taxonomy Distribution (Hard Skills)")
        lines.append("-------------------------------------------")
        for _, row in bloom_agg.iterrows():
            lines.append(f"- Model: {row['model_norm']}")
            for col in bloom_agg.columns:
                if col == "model_norm":
                    continue
                lines.append(f"    {col}: {int(row[col])}")
            lines.append("")
        lines.append("")

    # Coverage report
    if coverage_report is not None and not coverage_report.empty:
        lines.append("4. Coverage Metrics (Hybrid)")
        lines.append("----------------------------")
        cov_mean = coverage_report["coverage_percentage"].mean()
        cov_med = coverage_report["coverage_percentage"].median()
        skill_cov_mean = coverage_report["skill_coverage_pct"].mean()
        kn_cov_mean = coverage_report["knowledge_coverage_pct"].mean()
        avg_skill_conf = coverage_report["avg_skill_confidence"].mean()
        avg_kn_conf = coverage_report["avg_knowledge_confidence"].mean()

        lines.append(f"- Mean overall coverage percentage  : {cov_mean:.4f}")
        lines.append(f"- Median overall coverage percentage: {cov_med:.4f}")
        lines.append(f"- Mean skill coverage percentage    : {skill_cov_mean:.44f}")
        lines.append(f"- Mean knowledge coverage percentage: {kn_cov_mean:.4f}")
        lines.append(f"- Mean skill confidence (hybrid)    : {avg_skill_conf:.4f}")
        lines.append(f"- Mean knowledge confidence (hybrid): {avg_kn_conf:.4f}")
        lines.append("")

        if "missing_components_count" in coverage_report.columns:
            miss_mean = coverage_report["missing_components_count"].mean()
            miss_med = coverage_report["missing_components_count"].median()
            lines.append(f"- Mean missing components per job   : {miss_mean:.2f}")
            lines.append(f"- Median missing components per job : {miss_med:.2f}")
            lines.append("")
        lines.append("")

    # Curriculum demand summary
    if mapping_df is not None and not mapping_df.empty:
        lines.append("5. Curriculum Demand Summary (from Mapping)")
        lines.append("------------------------------------------")
        mapped = mapping_df[mapping_df["component_id"].notna()].copy()
        comp_counts = mapped["component_id"].value_counts()
        lines.append(f"- Total mapped items: {len(mapped)}")
        lines.append(f"- Unique curriculum components with demand: {len(comp_counts)}")
        lines.append("")
        lines.append("  Top 10 most demanded components (by count):")
        for cid, cnt in comp_counts.head(10).items():
            lines.append(f"    {cid}: {cnt}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Text report written to {report_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG_OUTPUT_DIR,
        help="Directory that contains the pipeline CSV outputs.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = ensure_dir(output_dir / FIG_DIR_NAME)

    print(f"[INFO] Using output_dir = {output_dir}")
    print(f"[INFO] Figures will be saved to {fig_dir}")

    dfs = load_all_data(output_dir)
    adv_skills = dfs["advanced_skills"]
    adv_knowledge = dfs["advanced_knowledge"]
    coverage_report = dfs["coverage_report"]
    comprehensive = dfs["comprehensive_analysis"]
    model_comp = dfs["model_comparison"]

    # 1. JobBERT vs GPT vs Hybrid extraction performance
    agg_model = plot_skill_knowledge_counts(model_comp, fig_dir)

    # 2. Hybrid improvements
    hybrid_diffs = plot_hybrid_improvement(model_comp, fig_dir)

    # 3. Bloom taxonomy distribution
    bloom_agg = plot_bloom_distribution(model_comp, fig_dir)

    # 4. Curriculum × Bloom heatmap + mapping
    if not adv_skills.empty or not adv_knowledge.empty:
        print("[INFO] Building curriculum mappings (this may take some time)...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)

        # 🔧 Only map HARD skills (curriculum is ~99% hard skills)
        hard_skills = adv_skills[adv_skills["type"] == "Hard"].copy()

        mapping_df = map_items_to_components(
            hard_skills,
            adv_knowledge,
            embedder,
            similarity_threshold=0.45,
        )
        if mapping_df is None:
            mapping_df = pd.DataFrame()
        plot_curriculum_bloom_heatmap(mapping_df, fig_dir)
    else:
        mapping_df = pd.DataFrame()

    # 5 & 6. Clustering of top skills/knowledge and demanded-but-not-covered items
    plot_top_skill_knowledge_clusters(adv_skills, adv_knowledge, mapping_df, fig_dir, top_n=40)

    # 7. Curriculum components with low demand
    if mapping_df is not None and not mapping_df.empty:
        plot_low_demand_components(mapping_df, fig_dir, top_k=10)

    # Confidence & coverage distributions
    plot_confidence_distributions(adv_skills, adv_knowledge, model_comp, fig_dir)
    plot_coverage_distributions(coverage_report, model_comp, fig_dir)

    # Text report
    write_text_report(
        agg_model=agg_model if agg_model is not None else pd.DataFrame(),
        hybrid_diffs=hybrid_diffs if hybrid_diffs is not None else {},
        bloom_agg=bloom_agg if bloom_agg is not None else pd.DataFrame(),
        coverage_report=coverage_report if coverage_report is not None else pd.DataFrame(),
        mapping_df=mapping_df if mapping_df is not None else pd.DataFrame(),
        output_dir=output_dir,
    )

    print("[INFO] Visualization pipeline completed.")


if __name__ == "__main__":
    main()
