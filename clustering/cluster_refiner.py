"""
cluster_refiner.py — Phase 2.1 LLM-as-arbiter cluster refinement.

Runs INSIDE `_cluster_one_stream` after agglomerative recovery and before
oversize split. For each surviving cluster, asks an LLM to verdict each
cluster member (keep / move_out) plus the top-K SBERT-closest items from the
current noise pool (join / skip). Applies moves only when cluster cohesion
improves monotonically.

The LLM sees item TEXTS only — no source sentences, no job titles, no
education-demand. Sentences/demand/role-shape enter the pipeline in Phase 2.2
generation, never here.

Public entry:

    refined_surviving, refined_labels, report = refine_cluster_memberships(
        canonical_items, embeddings, surviving_clusters, raw_labels,
        config=RefinementConfig(),
    )
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config + report
# --------------------------------------------------------------------------- #


@dataclass
class RefinementConfig:
    """Knobs for the LLM-as-arbiter refinement pass."""

    enabled: bool = False                       # off by default until validated
    model: str = "gpt-5.4-mini"
    max_iterations: int = 2
    n_noise_candidates: int = 10                # top-K from noise pool to consider per cluster
    noise_candidate_min_cosine: float = 0.40    # only consider noise items with centroid cosine >= this
    require_monotonic_cohesion: bool = True     # reject moves that lower cohesion
    min_cluster_size_after_moves: int = 3
    request_timeout: float = 60.0


@dataclass
class _ClusterAudit:
    cluster_idx: int
    initial_cohesion: float = 0.0
    final_cohesion: float = 0.0
    iterations: int = 0
    n_items_moved_out: int = 0
    n_items_moved_in: int = 0
    rejected_rounds: int = 0
    move_decisions: List[dict] = field(default_factory=list)


@dataclass
class RefinementReport:
    enabled: bool
    model: str
    n_clusters_processed: int = 0
    n_clusters_unchanged: int = 0
    n_clusters_dropped_below_min_size: int = 0
    total_items_moved_out: int = 0
    total_items_moved_in: int = 0
    total_llm_calls: int = 0
    cohesion_mean_before: float = 0.0
    cohesion_mean_after: float = 0.0
    runtime_seconds: float = 0.0
    per_cluster: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "n_clusters_processed": self.n_clusters_processed,
            "n_clusters_unchanged": self.n_clusters_unchanged,
            "n_clusters_dropped_below_min_size": self.n_clusters_dropped_below_min_size,
            "total_items_moved_out": self.total_items_moved_out,
            "total_items_moved_in": self.total_items_moved_in,
            "total_llm_calls": self.total_llm_calls,
            "cohesion_mean_before": round(float(self.cohesion_mean_before), 4),
            "cohesion_mean_after": round(float(self.cohesion_mean_after), 4),
            "runtime_seconds": round(float(self.runtime_seconds), 2),
            "per_cluster": self.per_cluster,
        }


# --------------------------------------------------------------------------- #
# LLM prompt
# --------------------------------------------------------------------------- #


_SYSTEM_PROMPT = (
    "You are reviewing a semantic cluster of skill/knowledge items extracted "
    "from real job postings. The cluster is supposed to represent ONE coherent "
    "capability area (a curriculum competency theme). Your job is to spot "
    "items that don't truly belong AND items from the noise pool that should "
    "join.\n\n"
    "You must output valid JSON only. No commentary outside the JSON object."
)


def _build_prompt(
    cluster_label: str,
    cohesion: float,
    items: List[Tuple[int, str]],         # (slot_idx, text)
    candidates: List[Tuple[int, str]],    # (slot_idx, text)
) -> str:
    item_lines = "\n".join(f'  {idx}. "{txt}"' for idx, txt in items)
    cand_lines = (
        "\n".join(f'  {idx}. "{txt}"' for idx, txt in candidates)
        if candidates
        else "  (none)"
    )
    return f"""CLUSTER HEURISTIC LABEL: "{cluster_label}"
CLUSTER COHESION: {cohesion:.3f}

CURRENT ITEMS IN THIS CLUSTER:
{item_lines}

CANDIDATES TO ADD (from noise pool, SBERT-close to this cluster):
{cand_lines}

TASK:
For each CURRENT item, decide whether it BELONGS in this cluster.
  - "keep"     → the item is on-theme; leave it in.
  - "move_out" → the item is off-theme; remove it from this cluster.

For each CANDIDATE, decide whether it should JOIN this cluster.
  - "join"     → the candidate fits this cluster's theme; add it.
  - "skip"     → the candidate doesn't fit well enough; leave it in noise.

Be conservative. Only move an item OUT if it clearly doesn't fit (e.g., a
mechanical-engineering item in a UX cluster). Only JOIN a candidate if it
clearly belongs (e.g., "wireframing" joining a UX cluster).

For each verdict, give a ONE-LINE reason — concrete, not generic.

OUTPUT JSON exactly in this shape:
{{
  "decisions": [
    {{"kind": "item", "idx": <int>, "verdict": "keep" | "move_out", "reason": "..."}},
    {{"kind": "candidate", "idx": <int>, "verdict": "join" | "skip", "reason": "..."}}
  ]
}}"""


# --------------------------------------------------------------------------- #
# JSON parsing (tolerant)
# --------------------------------------------------------------------------- #


_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #


def _pairwise_cosine_mean(emb: np.ndarray) -> float:
    n = emb.shape[0]
    if n <= 1:
        return 1.0
    sims = emb @ emb.T
    iu = np.triu_indices(n, k=1)
    return float(sims[iu].mean())


def _centroid(emb: np.ndarray) -> np.ndarray:
    c = emb.mean(axis=0)
    norm = np.linalg.norm(c)
    if norm > 0:
        c = c / norm
    return c.astype(np.float32)


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #


def _call_llm_arbiter(prompt: str, config: RefinementConfig) -> Optional[dict]:
    """One LLM call. Returns parsed JSON or None on failure."""
    try:
        from llm_client_router import get_client_for_model, strip_provider_prefix
    except ImportError as e:
        logger.warning("cluster_refiner: llm_client_router import failed: %s", e)
        return None

    client, provider = get_client_for_model(
        config.model, request_timeout=config.request_timeout
    )
    model_id = strip_provider_prefix(config.model) if provider == "jatevo" else config.model
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("cluster_refiner: LLM call failed: %s", e)
        return None
    return _parse_llm_json(raw)


# --------------------------------------------------------------------------- #
# Top-level entry — works on raw indices into the canonical-items array
# --------------------------------------------------------------------------- #


def refine_cluster_memberships(
    canonical_items: list,                                  # list of _CanonicalItem (clusterer-local)
    embeddings: np.ndarray,                                 # (n_canonical, d), L2-normalised
    surviving_clusters: List[Tuple[np.ndarray, str]],       # [(member_idx, method)]
    raw_labels: np.ndarray,                                 # int array; -1 = noise, sentinel int = clustered
    cluster_labels: List[str],                              # heuristic label per surviving cluster (same order)
    config: Optional[RefinementConfig] = None,
) -> Tuple[List[Tuple[np.ndarray, str]], np.ndarray, RefinementReport]:
    """Refine surviving cluster memberships in place.

    Inputs reference the canonical-items array. We mutate `surviving_clusters`
    (each tuple becomes a new `(member_idx, method)` with updated member_idx)
    and `raw_labels` (items moved to noise become -1; items joined into a
    cluster get the cluster's sentinel label).

    Args:
        canonical_items: list of _CanonicalItem; only `.canonical_text` is used.
        embeddings: aligned with canonical_items.
        surviving_clusters: clusters that passed the cohesion gate so far.
        raw_labels: int label array (HDBSCAN labels with recovery rewrites).
        cluster_labels: heuristic label per surviving cluster.
        config: optional RefinementConfig (disabled by default).

    Returns:
        (refined_surviving_clusters, refined_raw_labels, report)
    """
    if config is None:
        config = RefinementConfig()
    t0 = time.time()
    report = RefinementReport(enabled=config.enabled, model=config.model)

    if not config.enabled or not surviving_clusters:
        report.runtime_seconds = time.time() - t0
        return surviving_clusters, raw_labels, report

    # Working copies
    cluster_member_sets: List[set] = [set(m.tolist()) for m, _ in surviving_clusters]
    cluster_methods: List[str] = [meth for _, meth in surviving_clusters]
    audits = [
        _ClusterAudit(
            cluster_idx=i,
            initial_cohesion=_pairwise_cosine_mean(embeddings[m]) if len(m) > 1 else 1.0,
        )
        for i, (m, _) in enumerate(surviving_clusters)
    ]

    # Noise pool: indices currently in noise (raw_labels == -1)
    noise_indices: set = set(np.where(raw_labels == -1)[0].tolist())

    report.cohesion_mean_before = float(np.mean([a.initial_cohesion for a in audits]))
    report.n_clusters_processed = len(surviving_clusters)

    for iteration in range(config.max_iterations):
        any_change = False
        # Snapshot noise so two clusters in this iteration can't claim the same item
        noise_snapshot = sorted(noise_indices)
        used_noise_this_round: set = set()

        for cl_idx, members in enumerate(cluster_member_sets):
            if len(members) < 2:
                continue
            member_idx_array = np.array(sorted(members))
            cur_emb = embeddings[member_idx_array]
            current_cohesion = _pairwise_cosine_mean(cur_emb)
            centroid = _centroid(cur_emb)

            # Pick top-K noise candidates by centroid cosine
            available_noise = [
                ni for ni in noise_snapshot
                if ni not in used_noise_this_round
            ]
            if available_noise:
                noise_emb = embeddings[np.array(available_noise)]
                cosines = noise_emb @ centroid
                eligible_mask = cosines >= config.noise_candidate_min_cosine
                if eligible_mask.any():
                    eligible_local = np.where(eligible_mask)[0]
                    top_k_local = eligible_local[
                        np.argsort(-cosines[eligible_local])
                    ][: config.n_noise_candidates]
                    candidate_global_idx = [available_noise[i] for i in top_k_local]
                else:
                    candidate_global_idx = []
            else:
                candidate_global_idx = []

            # Build prompt slot indices (compact 1..N per call)
            items_slots = [
                (slot, canonical_items[gidx].canonical_text)
                for slot, gidx in enumerate(member_idx_array.tolist())
            ]
            cand_slots = [
                (slot, canonical_items[gidx].canonical_text)
                for slot, gidx in enumerate(candidate_global_idx)
            ]
            slot_to_global_item = {s: g for s, g in enumerate(member_idx_array.tolist())}
            slot_to_global_cand = {s: g for s, g in enumerate(candidate_global_idx)}

            prompt = _build_prompt(
                cluster_label=cluster_labels[cl_idx] or "(no label)",
                cohesion=current_cohesion,
                items=items_slots,
                candidates=cand_slots,
            )
            parsed = _call_llm_arbiter(prompt, config)
            report.total_llm_calls += 1
            if not parsed or "decisions" not in parsed:
                continue

            staged_move_out_globals: List[int] = []
            staged_join_globals: List[int] = []
            staged_reasons: List[dict] = []
            for d in parsed["decisions"]:
                kind = d.get("kind")
                idx = d.get("idx")
                verdict = d.get("verdict")
                reason = (d.get("reason") or "").strip()
                if not isinstance(idx, int):
                    continue
                if kind == "item" and verdict == "move_out" and idx in slot_to_global_item:
                    g = slot_to_global_item[idx]
                    staged_move_out_globals.append(g)
                    staged_reasons.append({
                        "kind": "item", "verdict": "move_out", "reason": reason,
                        "text": canonical_items[g].canonical_text,
                    })
                elif kind == "candidate" and verdict == "join" and idx in slot_to_global_cand:
                    g = slot_to_global_cand[idx]
                    staged_join_globals.append(g)
                    staged_reasons.append({
                        "kind": "candidate", "verdict": "join", "reason": reason,
                        "text": canonical_items[g].canonical_text,
                    })

            if not staged_move_out_globals and not staged_join_globals:
                continue

            new_members = (members - set(staged_move_out_globals)) | set(staged_join_globals)
            if len(new_members) < config.min_cluster_size_after_moves:
                audits[cl_idx].rejected_rounds += 1
                continue

            new_emb = embeddings[np.array(sorted(new_members))]
            new_cohesion = _pairwise_cosine_mean(new_emb)
            if config.require_monotonic_cohesion and new_cohesion + 1e-6 < current_cohesion:
                audits[cl_idx].rejected_rounds += 1
                continue

            # Commit
            cluster_member_sets[cl_idx] = new_members
            for g in staged_join_globals:
                used_noise_this_round.add(g)
            audits[cl_idx].n_items_moved_out += len(staged_move_out_globals)
            audits[cl_idx].n_items_moved_in += len(staged_join_globals)
            audits[cl_idx].iterations = iteration + 1
            audits[cl_idx].final_cohesion = new_cohesion
            audits[cl_idx].move_decisions.extend(staged_reasons)
            # Update noise pool: claimed items leave; moved-out items enter
            for g in staged_join_globals:
                noise_indices.discard(g)
            for g in staged_move_out_globals:
                noise_indices.add(g)
            any_change = True

        if not any_change:
            break

    # Rebuild raw_labels from final cluster membership
    refined_labels = raw_labels.copy()
    # Reset to noise where we removed items, then re-stamp cluster ownership
    # We use the original sentinel labels for non-noise items so downstream
    # code can still identify which cluster they belong to.
    # First: clear non-noise entries for items that moved out
    for cl_idx, members in enumerate(cluster_member_sets):
        # Get the sentinel label the cluster originally used (any value != -1
        # from the original member_idx array)
        orig_member_idx = surviving_clusters[cl_idx][0]
        if len(orig_member_idx):
            sentinel = int(raw_labels[orig_member_idx[0]])
        else:
            sentinel = -1
        # Re-stamp: every current member gets the sentinel; nothing else is changed here.
        for g in members:
            refined_labels[g] = sentinel
    # Items now in noise_indices (which may include items moved out of clusters)
    for g in noise_indices:
        refined_labels[g] = -1

    # Build refined surviving_clusters
    refined_surviving: List[Tuple[np.ndarray, str]] = []
    for cl_idx, members in enumerate(cluster_member_sets):
        if len(members) < config.min_cluster_size_after_moves:
            report.n_clusters_dropped_below_min_size += 1
            # Drop the cluster — push its items to noise
            for g in members:
                refined_labels[g] = -1
            continue
        member_arr = np.array(sorted(members))
        refined_surviving.append((member_arr, cluster_methods[cl_idx]))

    # Tally
    cohesions_after: List[float] = []
    for member_arr, _meth in refined_surviving:
        cohesions_after.append(
            _pairwise_cosine_mean(embeddings[member_arr]) if len(member_arr) > 1 else 1.0
        )
    report.cohesion_mean_after = float(np.mean(cohesions_after)) if cohesions_after else 0.0
    report.total_items_moved_out = sum(a.n_items_moved_out for a in audits)
    report.total_items_moved_in = sum(a.n_items_moved_in for a in audits)
    report.n_clusters_unchanged = sum(
        1 for a in audits if a.n_items_moved_out == 0 and a.n_items_moved_in == 0
    )
    report.per_cluster = [
        {
            "cluster_idx": a.cluster_idx,
            "iterations": a.iterations,
            "initial_cohesion": round(a.initial_cohesion, 4),
            "final_cohesion": round(a.final_cohesion, 4),
            "n_items_moved_out": a.n_items_moved_out,
            "n_items_moved_in": a.n_items_moved_in,
            "rejected_rounds": a.rejected_rounds,
            "move_decisions": a.move_decisions,
        }
        for a in audits
    ]
    report.runtime_seconds = time.time() - t0

    return refined_surviving, refined_labels, report
