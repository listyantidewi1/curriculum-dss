"""smoke_role_normalizer.py — eyeball role-bucket normalisation on a real run."""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mirror the dashboard's normalizer here (kept in sync via the smoke check).
_ROLE_BUCKETS = [
    ("DevOps / SRE",          ("devops", "site reliability", "platform engineer", "infrastructure engineer", "sre ")),
    ("Data Engineer",         ("data engineer", "etl ", "etl/", "data infrastructure")),
    ("Data Analyst",          ("data analyst", "business analyst", "data scientist", "analytics")),
    ("AI / ML Engineer",      ("machine learning", "ai engineer", "ml engineer", "ai/ml", "ai &", "ai,", "ai/", " ai ", "applied ai", "ai product")),
    ("Cloud / Solutions Arch",("cloud architect", "solutions architect", "solution architect", "cloud engineer", "aws ", " aws", "azure ", " azure", "gcp ")),
    ("Security / IAM",        ("security", "iam ", "identity ", "compliance", "penetration", "soc analyst")),
    ("Mobile Engineer",       ("mobile", "ios ", "android", " ios", " android")),
    ("Frontend / Web",        ("frontend", "front-end", "front end", "ui ", "react", "angular", "vue")),
    ("Backend Engineer",      ("backend", "back-end", "back end", "api engineer", "server engineer")),
    ("Full-Stack Engineer",   ("full stack", "full-stack", "fullstack")),
    ("Software Engineer",     ("software engineer", "software developer", "software dev", "swe ", "software design", "applications programmer", "applications engineer")),
    ("QA / Test Engineer",    ("qa ", "test engineer", "quality engineer", "sdet")),
    ("Product / Project Mgr", ("product manager", "project manager", "program manager", "scrum master", "delivery manager")),
    ("Sales / Customer",      ("customer", "sales", "account", "consultant")),
    ("Other Engineer",        ("engineer",)),
    ("Other Developer",       ("developer", "programmer")),
]


def _normalize_role(title: str) -> str:
    if not title:
        return "Other"
    low = title.lower()
    for bucket, needles in _ROLE_BUCKETS:
        for n in needles:
            if n in low:
                return bucket
    return "Other"


def main() -> int:
    run = "results/competency_v2_pipeline_n1k_v6_role_context"
    titles_path = PROJECT_ROOT / "DATA/preprocessing/data_prepared_n1k/jobs_metadata.csv"
    titles = {}
    with open(titles_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            jid = (row.get("job_id") or "").strip()
            t = (row.get("title") or "").strip()
            if jid and t:
                titles[jid] = t
    data = json.loads((PROJECT_ROOT / run / "competencies.json").read_text(encoding="utf-8"))
    for c in data:
        sj = c.get("source_job_ids") or []
        rc: collections.Counter = collections.Counter()
        for j in sj:
            t = titles.get(j, "")
            if t:
                rc[_normalize_role(t)] += 1
        top = rc.most_common(4)
        top_str = "  ".join(f"{r}:{n}" for r, n in top)
        print(f"{c.get('title','?')[:55]:55s}  n={len(sj):>2d}  {top_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
