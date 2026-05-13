"""smoke_it_relevance.py — sanity-check the v9 IT-relevance gate on the
exact failure cases from n1k_v8.1 (climbing/pizza/USPS) plus a few clear
software-engineering positives, before wiring into the pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from it_relevance_filter import classify_jobs, classify_it_sentences


JOB_CASES = [
    # ===== Should be NO (the failure modes we want to kill) =====
    ("Domino's Climbing Team", "Climbing Team members must infrequently navigate stairs or climb a ladder to change prices on signs, wash walls, perform maintenance. Pizza assembly station bending required. We offer competitive pay and benefits."),
    ("USPS delivery boilerplate", "Climbing During delivery of product, navigation of five or more flights of stairs may be required. WORK CONDITIONS Exposure To Varying and sometimes adverse weather conditions when delivering product, driving and couponing. Carrier route required."),
    ("Retail with computer mention", "Retail Associate needed at our flagship store. Greet customers, operate point-of-sale system, restock shelves, use computer for inventory entry. Must be able to lift 25 lbs."),
    ("Construction supervisor", "Construction Supervisor for residential projects. Oversee crew, read blueprints, coordinate trades, ensure safety compliance. Comfort with project-management software a plus."),
    # ===== Should be YES (clear IT positives) =====
    ("Senior DevOps Engineer", "You bring experience in Site Reliability, Platform Engineering, DevOps or similar roles, with a strong focus on production systems. Build CI/CD pipelines, manage infrastructure-as-code with Terraform and Kubernetes."),
    ("Junior Software Developer", "We are hiring a Software Developer to build and maintain web applications using React, TypeScript, and Node.js. Familiarity with REST APIs and Git required."),
    ("Data Engineer", "Design and scale data pipelines on AWS or Databricks. Apply best-practice engineering, automation, and governance to deliver reliable analytics-ready datasets."),
    ("Cyber Security Specialist", "Lead vulnerability assessments and incident response. Build SIEM detections, run penetration tests, and uphold ISO 27001 controls across cloud and on-prem environments."),
    # ===== Edge cases =====
    ("AI/ML Research Scientist", "Develop novel deep-learning architectures for computer vision and NLP. Publish at top conferences. PhD preferred."),
    ("IT Project Manager", "Coordinate cross-functional software delivery projects. Manage sprint planning, vendor relationships, and stakeholder communications."),
]


SENTENCE_CASES = [
    # ===== Should be NO (boilerplate that contaminated v8.1) =====
    "Climbing Team members must infrequently navigate stairs or climb a ladder to change prices on signs, wash walls, perform maintenance.",
    "Climbing During delivery of product, navigation of five or more flights of stairs may be required.",
    "Rarely: Climbing, Crouching, Kneeling, Pulling (5-15lbs), Pushing (5-15lbs), Lifting (5-15lbs), Stooping is required.",
    "Stooping/Bending Forward bending at the waist is necessary at the pizza assembly station.",
    "We offer competitive pay, paid time off, and a comprehensive benefits package.",
    "Equal Opportunity Employer. All qualified applicants will receive consideration for employment without regard to race, color, religion, sex, or national origin.",
    "Please submit your application via our online portal.",
    "Job Type: Full-time. Location: Remote/Hybrid.",
    # ===== Should be YES (real IT skill sentences) =====
    "You bring 5+ years of experience building CI/CD pipelines and managing infrastructure-as-code with Terraform.",
    "Strong understanding of REST APIs, microservices architecture, and event-driven systems.",
    "Hands-on experience training and evaluating machine-learning models using TensorFlow or PyTorch.",
    "Apply container orchestration patterns using Kubernetes and Helm for production services.",
    "Design and integrate cloud-native solutions across AWS and GCP environments.",
    "Build secure authentication flows using OAuth 2.0 and OpenID Connect.",
]


def main() -> int:
    cache_dir = PROJECT_ROOT / "cache" / "it_relevance_filter"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("JOB-LEVEL classifier")
    print("=" * 72)
    descs = [d for _, d in JOB_CASES]
    verdicts = classify_jobs(descs, cache_path=cache_dir / "smoke_job.json")
    for (name, _), v in zip(JOB_CASES, verdicts):
        flag = "[YES IT]" if v else "[NO ]   "
        print(f"  {flag}  {name}")
    print()
    print("=" * 72)
    print("SENTENCE-LEVEL classifier")
    print("=" * 72)
    verdicts = classify_it_sentences(SENTENCE_CASES, cache_path=cache_dir / "smoke_sent.json")
    for s, v in zip(SENTENCE_CASES, verdicts):
        flag = "[YES IT]" if v else "[NO ]   "
        print(f"  {flag}  {s[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
