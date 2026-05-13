"""Smoke-test the shared role_normalization module."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from role_normalization import normalize_role, role_distribution


def main() -> int:
    cases = [
        ("Senior DevOps Engineer III", "DevOps / SRE"),
        ("Lead Software Developer - IAM", "Security / IAM"),
        ("Customer Identity Access Management Engineer - Senior", "Security / IAM"),
        ("AI/ML Engineer", "AI / ML Engineer"),
        ("Senior Product Designer", "Other"),
        ("Mechanical Design Engineer", "Other Engineer"),
        ("", "Other"),
    ]
    for title, expected in cases:
        got = normalize_role(title)
        ok = "OK " if got == expected else "FAIL"
        print(f"{ok} {title!r:60s} -> {got!r:25s} (expected {expected!r})")
    dist = role_distribution([
        "Senior DevOps Engineer", "Senior Site Reliability Engineer",
        "Software Developer", "AI/ML Engineer", "Mechanical Design Engineer",
    ])
    print(f"\nrole_distribution sample: {dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
