"""Render a v8 build_cluster_prompt for one real cluster from v7's output so
we can eyeball the new sentence-included prompt + role distribution. Does not
call the LLM.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from competency_v2_schema import GeneratorConfig
from competency_generator_v2 import build_cluster_prompt


class _StubItem:
    def __init__(self, text, sentence_id, sentence_text, confidence_score=0.9):
        self.text = text
        self.sentence_id = sentence_id
        self.sentence_text = sentence_text
        self.confidence_score = confidence_score
        # The generator's _build_skill_lines uses hasattr(it, 'type') to mark
        # skill vs knowledge — we mark it as a skill.
        self.type = "Hard"


class _StubCluster:
    def __init__(self, id, summary_label, cohesion_score, items):
        self.id = id
        self.stream = "hard_plus_knowledge"
        self.cohesion_score = cohesion_score
        self.summary_label = summary_label
        self.n_items = len(items)
        self.n_unique_jobs = len({i.sentence_id.rpartition("_")[0] for i in items})
        self.items = items


def main() -> int:
    # Build a stub cluster modelled on v7's DevOps cluster (which had a strong,
    # role-coherent shape).
    items = [
        _StubItem("DevOps", "in00485717888f4779_0001", "You bring 5+ years of experience in DevOps and CI/CD pipelines in a production environment."),
        _StubItem("DevOps", "in0cac4c44efc9dd8a_0003", "It is a hands-on Senior DevOps Engineer role with strong automation and DevOps practices."),
        _StubItem("CI/CD", "in1d10b0b5d6dc92b3_0002", "Bachelor's degree in Computer Science required. Build CI/CD pipelines and automate deployment."),
        _StubItem("software development lifecycle (SDLC)", "in00485717888f4779_0005", "Lead end-to-end solution development across the software development lifecycle (SDLC)."),
        _StubItem("SDLC", "in2a94a33a01c30b61_0007", "Apply container orchestration patterns using Kubernetes and Helm for production services."),
    ]
    cluster = _StubCluster(
        id="cluster_h_smoke",
        summary_label="devops ci/cd sdlc: lifecycle delivery",
        cohesion_score=0.846,
        items=items,
    )

    # Load real jobs_metadata so we get a meaningful role distribution
    jm_path = PROJECT_ROOT / "DATA/preprocessing/data_prepared_n1k/jobs_metadata.csv"
    job_titles = {}
    if jm_path.exists():
        with open(jm_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                jid = (row.get("job_id") or "").strip()
                t = (row.get("title") or "").strip()
                if jid and t:
                    job_titles[jid] = t

    cfg = GeneratorConfig(model="gpt-5.4-mini")
    prompt = build_cluster_prompt(cluster, cfg, job_titles=job_titles)
    print("--- FULL PROMPT (v8) ---")
    print(prompt)
    print("--- END ---")
    print(f"\nPrompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
