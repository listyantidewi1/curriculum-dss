"""
role_normalization.py — bucket raw job-posting titles into broad role families.

Job-posting titles are uniquely worded ("Senior DevOps Engineer III", "Lead
Reliability Engineer II") so a histogram of raw titles is always flat. This
module maps a raw title to one of ~15 role families via substring match —
first hit wins, more specific patterns come first.

Used by:
- `dashboard_v2/app.py` — role-shape panel per competency
- `competency_generator_v2.py` — role distribution shown to the LLM in the
  Phase 2.2 prompt (Phase 2.A of the v8 sprint)
- `scripts/smoke_role_normalizer.py` — regression sanity check

Single source of truth — keep `_ROLE_BUCKETS` here. Callers must NOT
duplicate the table.
"""
from __future__ import annotations

from typing import Iterable, Tuple


# Role-normalisation table. Order matters: more specific patterns come first.
ROLE_BUCKETS: list = [
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
    ("Other Engineer",        ("engineer",)),  # last-resort catch
    ("Other Developer",       ("developer", "programmer")),
]


def normalize_role(title: str) -> str:
    """Map a raw job-posting title to a broad role bucket. Falls back to
    'Other' when no pattern matches."""
    if not title:
        return "Other"
    low = title.lower()
    for bucket, needles in ROLE_BUCKETS:
        for n in needles:
            if n in low:
                return bucket
    return "Other"


def role_distribution(titles: Iterable[str]) -> list:
    """Return a sorted-by-count list of (bucket, count, fraction) tuples for
    the given collection of titles. Unknown titles still go to 'Other'."""
    from collections import Counter
    c: Counter = Counter()
    for t in titles:
        if t:
            c[normalize_role(t)] += 1
    total = sum(c.values()) or 1
    return [(bucket, n, n / total) for bucket, n in c.most_common()]
