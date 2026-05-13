"""
user_ratings.py

Phase 2.6 sub-deliverable: user-rating store + aggregation for the public
dashboard.

Per locked design (see docs/PHASE_2_2_DESIGN.md notes + memory):

  * Light-signup gated — anonymous browse is free; rating requires a
    minimal account (email + role chip stored hashed).
  * Star rating 1-5 + optional free-text feedback (max 2000 chars).
  * **Research signal only.** Ratings do NOT enter the `priority_score`
    formula or affect the live competency ranking. They power a post-hoc
    research analysis (paper RQ5 augmentation: correlation between user
    ratings and automated grounding scores).
  * Per-competency aggregation surfaces: mean, std, count, distribution,
    and per-role breakdown.

Storage: CSV files in `feedback_store/`. No database — keeps deployment
simple for Streamlit Community Cloud (which has no persistent DB) and
keeps the audit trail human-readable.

    feedback_store/public_users.csv
        user_id, email_hash, role, signed_up_at
    feedback_store/public_competency_ratings.csv
        rating_id, user_id, competency_id, rating, feedback_text, role, timestamp,
        pipeline_run_tag

Note: emails are stored ONLY as SHA-256 hashes so the project never holds
plaintext PII. The hash is salted with a project-local secret to prevent
rainbow-table linkage.
"""
from __future__ import annotations

import csv
import hashlib
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# Salt: read from env var, fallback to a project-local file, fallback to a
# weak default that should be overridden for any public deployment.
def _load_salt() -> str:
    env = os.environ.get("RATINGS_HASH_SALT")
    if env:
        return env
    salt_file = Path("feedback_store") / ".ratings_salt"
    if salt_file.exists():
        return salt_file.read_text(encoding="utf-8").strip()
    return "dev-only-salt-CHANGE-FOR-PRODUCTION"


VALID_ROLES = ("educator", "student", "industry", "government", "other")
USERS_PATH = Path("feedback_store") / "public_users.csv"
RATINGS_PATH = Path("feedback_store") / "public_competency_ratings.csv"

USERS_COLS = ["user_id", "email_hash", "role", "signed_up_at"]
RATINGS_COLS = [
    "rating_id", "user_id", "competency_id", "rating",
    "feedback_text", "role", "timestamp", "pipeline_run_tag",
]


def _ensure_csv(path: Path, columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(columns)


def _hash_email(email: str) -> str:
    email_norm = email.strip().lower()
    salted = (_load_salt() + "\x00" + email_norm).encode("utf-8")
    return hashlib.sha256(salted).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Sign-up / lookup
# --------------------------------------------------------------------------- #


def _read_users() -> List[Dict]:
    _ensure_csv(USERS_PATH, USERS_COLS)
    with open(USERS_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_or_create_user(email: str, role: str) -> Dict:
    """Look up a user by email; create them if not present. Returns the user record."""
    if not email or "@" not in email:
        raise ValueError("invalid email")
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role; pick one of {VALID_ROLES}")

    email_hash = _hash_email(email)
    for u in _read_users():
        if u["email_hash"] == email_hash:
            # Allow role update if the user picked a different one this session
            if u["role"] != role:
                _update_user_role(email_hash, role)
                u["role"] = role
            return u

    new_user = {
        "user_id": f"user_{uuid.uuid4().hex[:10]}",
        "email_hash": email_hash,
        "role": role,
        "signed_up_at": _now_iso(),
    }
    _ensure_csv(USERS_PATH, USERS_COLS)
    with open(USERS_PATH, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=USERS_COLS).writerow(new_user)
    return new_user


def _update_user_role(email_hash: str, role: str) -> None:
    users = _read_users()
    for u in users:
        if u["email_hash"] == email_hash:
            u["role"] = role
    with open(USERS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=USERS_COLS)
        w.writeheader()
        w.writerows(users)


# --------------------------------------------------------------------------- #
# Rating submission
# --------------------------------------------------------------------------- #


def _read_ratings() -> List[Dict]:
    _ensure_csv(RATINGS_PATH, RATINGS_COLS)
    with open(RATINGS_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def submit_rating(
    user_id: str,
    role: str,
    competency_id: str,
    rating: int,
    feedback_text: str = "",
    pipeline_run_tag: str = "",
) -> str:
    """Insert or overwrite a rating. One rating per (user_id, competency_id).

    Returns the rating_id (newly minted or pre-existing-overwritten).
    """
    if not (1 <= int(rating) <= 5):
        raise ValueError("rating must be 1-5")
    if len(feedback_text) > 2000:
        feedback_text = feedback_text[:2000]

    rows = _read_ratings()
    existing_idx = None
    for i, r in enumerate(rows):
        if r["user_id"] == user_id and r["competency_id"] == competency_id:
            existing_idx = i
            break

    if existing_idx is not None:
        rid = rows[existing_idx]["rating_id"]
        rows[existing_idx].update({
            "rating": str(int(rating)),
            "feedback_text": feedback_text,
            "role": role,
            "timestamp": _now_iso(),
            "pipeline_run_tag": pipeline_run_tag,
        })
    else:
        rid = f"rate_{uuid.uuid4().hex[:10]}"
        rows.append({
            "rating_id": rid,
            "user_id": user_id,
            "competency_id": competency_id,
            "rating": str(int(rating)),
            "feedback_text": feedback_text,
            "role": role,
            "timestamp": _now_iso(),
            "pipeline_run_tag": pipeline_run_tag,
        })

    with open(RATINGS_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RATINGS_COLS)
        w.writeheader()
        w.writerows(rows)
    return rid


# --------------------------------------------------------------------------- #
# Aggregation for the dashboard
# --------------------------------------------------------------------------- #


@dataclass
class RatingAggregate:
    competency_id: str
    rating_count: int
    rating_mean: float
    rating_std: float
    distribution: Dict[int, int]                  # 1→n, 2→n, ... 5→n
    by_role: Dict[str, Dict[str, float]]          # role → {mean, count}

    def to_dict(self) -> dict:
        return {
            "competency_id": self.competency_id,
            "rating_count": self.rating_count,
            "rating_mean": round(self.rating_mean, 3) if self.rating_count else None,
            "rating_std": round(self.rating_std, 3) if self.rating_count > 1 else None,
            "distribution": {str(k): v for k, v in self.distribution.items()},
            "by_role": self.by_role,
        }


def get_aggregate(competency_id: str) -> RatingAggregate:
    """Compute aggregate stats for one competency."""
    rows = [r for r in _read_ratings() if r.get("competency_id") == competency_id]

    if not rows:
        return RatingAggregate(
            competency_id=competency_id,
            rating_count=0, rating_mean=0.0, rating_std=0.0,
            distribution={i: 0 for i in range(1, 6)},
            by_role={},
        )

    import statistics
    values = [int(r["rating"]) for r in rows]
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    dist = Counter(values)
    distribution = {i: dist.get(i, 0) for i in range(1, 6)}

    by_role: Dict[str, Dict[str, float]] = {}
    role_buckets: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        role_buckets[r.get("role") or "other"].append(int(r["rating"]))
    for role, vs in role_buckets.items():
        by_role[role] = {
            "mean": round(statistics.mean(vs), 3),
            "count": len(vs),
        }

    return RatingAggregate(
        competency_id=competency_id,
        rating_count=len(rows),
        rating_mean=mean,
        rating_std=std,
        distribution=distribution,
        by_role=by_role,
    )


def get_user_rating(user_id: str, competency_id: str) -> Optional[Dict]:
    """Return the user's existing rating for this competency, if any."""
    for r in _read_ratings():
        if r.get("user_id") == user_id and r.get("competency_id") == competency_id:
            return r
    return None
