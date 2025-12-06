"""
generate_competencies_llm.py

Takes high/medium-verified skills from verified_skills.csv and asks an LLM
(through OpenRouter) to propose competency statements / curriculum components.

Output:
    competency_proposals.json  (list of JSON objects with competencies)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from openai import OpenAI

import config  # uses config.OUTPUT_DIR


def build_future_context(output_dir: Path,
                         future_file: str = "future_skill_weights_dummy.csv",
                         top_k_domains: int = 5,
                         top_k_knowledge: int = 15) -> str:
    """
    Build a short text summary of future-critical domains and knowledge items
    from future_skill_weights_dummy.csv to guide competency generation.
    """
    fw_path = output_dir / future_file
    if not fw_path.exists():
        print(f"[WARN] Future weights file not found at {fw_path}. "
              f"Continuing without future context.")
        return ""

    df = pd.read_csv(fw_path)

    required_cols = {"knowledge", "best_future_domain",
                     "trend_label", "future_weight"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Future weights file missing columns {missing}. "
              f"Continuing without future context.")
        return ""

    # Aggregate by domain
    domain_stats = (
        df.groupby("best_future_domain")["future_weight"]
        .mean()
        .reset_index()
        .sort_values("future_weight", ascending=False)
        .head(top_k_domains)
    )

    # Top knowledge items overall
    top_kw = (
        df.sort_values("future_weight", ascending=False)
        .head(top_k_knowledge)[["knowledge", "best_future_domain", "future_weight"]]
    )

    # Build human-readable context text
    lines = []
    lines.append("Future-critical domains (average future_weight):")
    for _, row in domain_stats.iterrows():
        lines.append(
            f"- {row['best_future_domain']} "
            f"(avg future_weight={row['future_weight']:.2f})"
        )

    lines.append("")
    lines.append("Example high future-weight knowledge items:")
    for _, row in top_kw.iterrows():
        lines.append(
            f"- {row['knowledge']} "
            f"(domain={row['best_future_domain']}, "
            f"future_weight={row['future_weight']:.2f})"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Helper: load OpenRouter / OpenAI client the same way as in pipeline
# ----------------------------------------------------------------------

def load_openrouter_client() -> OpenAI:
    base_url = "https://openrouter.ai/api/v1"

    # 1) env var, if available
    api_key = os.getenv("OPENROUTER_API_KEY")

    # 2) fallback: api_keys/OpenRouter.txt
    if not api_key:
        key_path = Path("api_keys") / "OpenRouter.txt"
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise RuntimeError(
                f"OpenRouter API key not found. Set OPENROUTER_API_KEY "
                f"or create {key_path}"
            )

    if not api_key:
        raise RuntimeError("OpenRouter API key is empty.")

    return OpenAI(api_key=api_key, base_url=base_url)


# ----------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------

def build_prompt(skills: List[str], future_context: str = "") -> str:
    bullet_list = "\n".join(f"- {s}" for s in skills)

    future_block = ""
    if future_context.strip():
        future_block = f"""
Additional context about future-critical domains and technologies:

{future_context}
"""

    return f"""
You are an expert in competency-based education and vocational curriculum design.

You are given a list of VERIFIED job skills (mostly hard skills) extracted from
real job postings in the Software & Game Development domain.

Your task is to group related skills and generate COMPETENCY STATEMENTS that are
suitable for use in an upper-secondary / vocational curriculum (e.g., Indonesian SMK).

Please follow these rules:

1. Output MUST be valid JSON only (no explanation text).
2. The root should be a JSON object with key "competencies" whose value is a list.
3. Each competency is an object with:
   - "id": a short identifier (e.g., "C1", "C2", ...).
   - "title": a concise competency title.
   - "description": 2–3 sentences competency statement that can be used in a curriculum.
   - "related_skills": list of skill phrases from the input that this competency covers.
   - "future_relevance": a short note (1–2 sentences) on why this competency
     matters for the future of work (based on the context if available).
4. Try to produce between 10 and 25 competencies for this batch.
5. Prefer higher-level, integrative competencies, not trivial one-skill items.
6. Give slightly higher priority and more detail to skills and themes that appear
   in future-critical domains (AI, data, cloud, security, human–AI collaboration),
   if such context is provided.

{future_block}

Here are the verified skills:

{bullet_list}
"""


# ----------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------

import re
import json
from typing import List, Dict
from openai import OpenAI

def call_llm_for_competencies(client: OpenAI,
                              skills: List[str],
                              model_name: str,
                              future_context: str = "") -> Dict:
    prompt = build_prompt(skills, future_context=future_context)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise curriculum designer. "
                    "Always respond with VALID JSON only. "
                    "Do NOT include any markdown fences or commentary."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1500,
    )


    content = resp.choices[0].message.content or ""
    # strip possible ```json ... ``` fences
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n```$", "", content)
    content = content.strip()

    # 1st attempt: direct parse
    # 1st attempt: direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 2nd attempt: truncate at last closing brace/bracket
        last_brace = max(content.rfind("}"), content.rfind("]"))
        if last_brace != -1:
            truncated = content[: last_brace + 1]
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

        # 3rd: save raw to file for debugging and return empty result instead of crashing
        with open("last_competency_raw_response.txt", "w", encoding="utf-8") as f:
            f.write(content)

        print(
            "[WARN] Failed to parse LLM JSON response for this batch. "
            f"Error: {e}. Raw content saved to last_competency_raw_response.txt. "
            "Continuing with an empty 'competencies' list for this batch."
        )
        return {"competencies": []}



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate competency proposals from verified skills using an LLM."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.OUTPUT_DIR),
        help="Directory containing verified_skills.csv (default: config.OUTPUT_DIR)",
    )
    parser.add_argument(
        "--verified_file",
        type=str,
        default="verified_skills.csv",
        help="Input verified skills file (default: verified_skills.csv)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="competency_proposals.json",
        help="Output JSON file (default: competency_proposals.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-chat",
        help="Model name on OpenRouter (default: deepseek/deepseek-chat)",
    )
    parser.add_argument(
        "--max_skills_per_call",
        type=int,
        default=30,
        help="Number of skills per LLM call (default: 30)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    verified_path = out_dir / args.verified_file
    output_path = out_dir / args.output_json

    # Build future-of-work context from future_skill_weights_dummy.csv (if available)
    future_context = build_future_context(out_dir)
    if future_context:
        print("[INFO] Future context loaded and will be injected into LLM prompt.")
    else:
        print("[INFO] No future context available; generating competencies from skills only.")


    if not verified_path.exists():
        raise FileNotFoundError(f"verified_skills.csv not found: {verified_path}")

    print(f"[INFO] Reading verified skills from {verified_path}")
    df = pd.read_csv(verified_path)

    if "is_verified" not in df.columns:
        raise ValueError("verified_skills.csv must contain 'is_verified' column")

    # Use only high + medium verified skills
    df_v = df[df["is_verified"] == True].copy()

    if df_v.empty:
        raise RuntimeError("No verified skills found (is_verified == True).")

    # Aggregate unique skill phrases with frequency (for info only)
    grp = df_v.groupby("skill").size().reset_index(name="freq")
    skills_sorted = grp.sort_values("freq", ascending=False)["skill"].tolist()

    print(f"[INFO] Total unique verified skills: {len(skills_sorted)}")

    client = load_openrouter_client()

    all_competencies = []
    batch_id = 1

    # Chunk skills for multiple LLM calls
    for i in range(0, len(skills_sorted), args.max_skills_per_call):
        chunk = skills_sorted[i : i + args.max_skills_per_call]
        print(f"[INFO] Calling LLM for batch {batch_id} "
              f"({len(chunk)} skills)...")

        data = call_llm_for_competencies(
            client,
            chunk,
            args.model,
            future_context=future_context,
        )

        # Expecting {"competencies": [ ... ]}
        comps = data.get("competencies", [])
        # Annotate with batch id
        for c in comps:
            c.setdefault("batch_id", batch_id)
        all_competencies.extend(comps)

        batch_id += 1

    print(f"[INFO] Total competencies generated: {len(all_competencies)}")

    output_path.write_text(
        json.dumps({"competencies": all_competencies}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Saved competency proposals to {output_path}")


if __name__ == "__main__":
    main()
