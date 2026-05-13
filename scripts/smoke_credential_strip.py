"""Smoke-test the credential-stripping helper used by the Phase 2 generator
prompt (v8 sprint Phase 2.A). Goal: confirm seniority cues, years-of-experience,
and degree phrases are removed; the rest of the sentence stays intact."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from competency_generator_v2 import _strip_credential_cues


def main() -> int:
    cases = [
        # Years of experience (single + ranged + with experience suffix)
        "You bring 5+ years of experience in DevOps roles.",
        "You bring: 3-4+ years of experience in Site Reliability, Platform Engineering, DevOps or similar roles.",
        "Minimum 3 years of relevant experience required.",
        # Seniority prefixes
        "Senior DevOps Engineer with strong automation focus.",
        "Lead Software Developer needed for IAM platform.",
        "We are hiring a Principal Cloud Architect.",
        # Degree phrases
        "Bachelor's degree in Computer Science or related field required.",
        "Master's degree in Engineering, Mathematics, or Statistics preferred.",
        "PhD in Computer Science is a plus.",
        # Combined (the v6 review case)
        "You have 5+ years of experience as a Senior DevOps Engineer with a Bachelor's degree in CS.",
        # No credentialing — should pass through largely unchanged
        "You will build CI/CD pipelines and automate deployment for production services.",
        "Apply container orchestration patterns using Kubernetes and Helm.",
    ]
    for s in cases:
        stripped = _strip_credential_cues(s)
        print(f"IN : {s}")
        print(f"OUT: {stripped}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
