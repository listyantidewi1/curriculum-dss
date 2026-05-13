"""
competency_generator_v2.py

Phase 2.2 cluster-driven competency generator with reasoning + provenance.

Replaces the legacy domain-batched `generate_competencies.py` entry point.
See docs/PHASE_2_2_DESIGN.md for the architectural rationale.

Top-level entry point:

    from competency_generator_v2 import generate_competencies_v2, GeneratorConfig
    competencies, batch_reasonings = generate_competencies_v2(
        clusters=clusters,  # output of clustering.cluster_skills(...)
        config=GeneratorConfig(model="deepseek/deepseek-v3.2"),
        top_soft_skills=top_soft_skills_list,  # optional, for soft-skill reference
        future_weights=future_weights_dict,    # optional, for future-relevance hints
    )

Output:
    competencies: List[CompetencyV2] (post-dedup, capped at config.final_competency_cap)
    batch_reasonings: List[BatchReasoning] (one per LLM call)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from competency_v2_schema import (
    GENERATOR_VERSION,
    BatchReasoning,
    CompetencyV2,
    GeneratorConfig,
)
from llm_client_router import get_client_for_model, strip_provider_prefix

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = (
    "You are an expert in competency-based education. You are given a "
    "CLUSTER of verified hard skills and knowledge items extracted from "
    "real software-engineering job postings. Your job is to synthesize "
    "them into curriculum-ready competency statements that match the "
    "evidence in the cluster — at whatever depth and complexity the "
    "evidence supports.\n\n"
    "You MUST output valid JSON only. No markdown fences, no commentary outside "
    "the JSON object."
)


# Brief KKNI reference for the LLM (Indonesian National Qualifications Framework).
# Full reference is in kkni.py; this is the inline summary for the prompt.
# v8 sprint update (2026-05-13): extended with Bloom-style cognitive-verb
# anchors per level so the LLM can place KKNI from cognitive complexity ALONE.
# IMPORTANT — KKNI is a cognitive-complexity label, NOT a market-credentialing
# label. The LLM must NOT consider what education stage employers currently
# hire at (a D3 institution can teach S2 competencies; a KKNI-7 competency
# can be in demand at SMK, D3, or S1). Education-stage demand stays a
# separate label and never enters this decision.
KKNI_BRIEF = """KKNI levels (Indonesian framework; pick from cognitive complexity ONLY):

  Level 3 — operational, supervised work; follows procedures.
            Cognitive verbs: identify, follow, perform, record, report.

  Level 4 — minor diagnostic decisions; applies standard methods within
            defined scope; some autonomy on routine choices.
            Cognitive verbs: apply, configure, troubleshoot, monitor, adjust.

  Level 5 — analyses + selects among standard methods; supervises routine
            work; combines techniques in established patterns.
            Cognitive verbs: analyse, select, compare, supervise, plan, refine.

  Level 6 — designs novel-within-standard solutions; leads small teams; makes
            architectural / methodological choices; integrates multiple
            domains within a discipline.
            Cognitive verbs: design, lead, integrate, optimise, evaluate, derive.

  Level 7 — complex novel problems; multidisciplinary integration; leadership
            responsibility; defines strategies and frameworks.
            Cognitive verbs: synthesise, formulate, strategise, govern, innovate.

  Level 8+ — research, innovation, original contribution. (Rare in software-
            engineering operational work; usually only for research roles.)

Pick the level whose cognitive descriptor + verbs best match the evidence
in the cluster (source sentences). Do NOT consider what education stage
employers currently hire at — that is a separate label that does not enter
this decision."""


# v8 sprint — strip patterns that leak credentialing / seniority cues from
# source sentences before showing them to the LLM. Three families:
#  - Job titles ("Senior X Engineer", "Lead Y Developer", "Junior Z Analyst")
#  - Years-of-experience ("5+ years", "3-4 years' experience")
#  - Degree/credential phrases ("Bachelor's degree in CS", "Master's degree
#    in Engineering", "PhD")
# These cues bias the LLM toward placing KKNI from seniority/credentialing
# instead of from cognitive complexity.

_CREDENTIAL_STRIP_PATTERNS = [
    # Years of experience — single or ranged, with optional + and "of experience".
    # Examples: "5+ years", "3-4 years", "3-4+ years of experience", "5 years exp"
    re.compile(
        r"\b(at\s+least\s+|minimum\s+|min\.?\s+)?"
        r"\d+(\s*[-–]\s*\d+)?\s*\+?\s*years?[\'`’]?\b"
        r"(\s*\+?\s*(of|in)\s+\w+)?"
        r"(\s+(experience|exp|exp\.|relevant\s+experience))?",
        re.IGNORECASE,
    ),
    # Job-title seniority prefixes that signal level rather than cognitive demand
    re.compile(
        r"\b(senior|sr\.?|junior|jr\.?|lead|principal|staff|chief|head\s+of|director\s+of)\b\s+",
        re.IGNORECASE,
    ),
    # Degree / credential phrases. Greedy enough to grab the field name but
    # stops at sentence punctuation or common conjunctions to avoid eating
    # the rest of the sentence.
    re.compile(
        r"\b(bachelor['’]?s?|master['’]?s?|ph\.?d\.?|doctorate|associate['’]?s?|undergraduate|graduate)\b"
        r"(\s+(of|in|degree|degrees))?"
        r"(\s+[a-z][a-z\s,/&-]*?)?"
        r"(\s+(is\s+)?(preferred|required|a\s+plus|advantageous|desired|essential))?"
        r"(?=[.;,]|\s+(or|and|with|including)\b|\Z|\n)",
        re.IGNORECASE,
    ),
]


def _strip_credential_cues(text: str) -> str:
    """Remove credentialing / seniority leakage from a source sentence before
    showing it to the LLM. See `_CREDENTIAL_STRIP_PATTERNS` for the families.
    Leaves the rest of the sentence intact."""
    if not text:
        return ""
    s = text
    for pat in _CREDENTIAL_STRIP_PATTERNS:
        s = pat.sub(" ", s)
    # Collapse whitespace introduced by the substitutions
    s = re.sub(r"\s+", " ", s).strip()
    # Drop stray dangling punctuation
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    return s


def _sample_sentences_for_item(
    items: list,
    max_sentences: int = 2,
) -> list:
    """Pick up to `max_sentences` distinct, credential-stripped source sentences
    that represent this canonical item. Sentences shorter than 25 chars after
    stripping are skipped (likely fragments)."""
    seen: set = set()
    out: list = []
    for it in items:
        stxt = (getattr(it, "sentence_text", "") or "").strip()
        if not stxt:
            continue
        stripped = _strip_credential_cues(stxt)
        if len(stripped) < 25:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
        if len(out) >= max_sentences:
            break
    return out


def _summarize_skill_item_for_prompt(item, source_sentence_counter: Dict[str, int]) -> str:
    """Render a SkillItem / KnowledgeItem as a compact prompt line.

    Format: `- "<text>" (type=skill|knowledge, jobs=N, confidence=X.XX)`

    `source_sentence_counter` is precomputed: text -> count of unique source sentences.
    """
    text = getattr(item, "text", "")
    is_skill = hasattr(item, "type")
    item_type = "skill" if is_skill else "knowledge"
    sid = getattr(item, "sentence_id", "") or ""
    src_count = source_sentence_counter.get(text, 1)
    conf = float(getattr(item, "confidence_score", 0.0) or 0.0)
    return f'- "{text}" (type={item_type}, sentences={src_count}, conf={conf:.2f})'


def _build_skill_lines(
    cluster,
    max_items_in_prompt: int = 25,
    include_sentences: bool = False,
    max_sentences_per_item: int = 2,
) -> str:
    """Produce the bulleted SKILLS-IN-THIS-CLUSTER section of the prompt.

    Deduplicates by text so the LLM sees each distinct phrase once. Caps at
    `max_items_in_prompt`. When `include_sentences=True` (v8 sprint Phase 2.A),
    appends up to `max_sentences_per_item` representative source sentences
    per item, with credentialing/seniority cues stripped via
    `_strip_credential_cues`.
    """
    seen_text = {}
    for it in cluster.items:
        t = getattr(it, "text", "")
        if not t:
            continue
        seen_text.setdefault(t, []).append(it)

    # Count distinct sentence_ids per text for the "sentences=N" hint
    src_count: Dict[str, int] = {}
    for text, items in seen_text.items():
        sids = {getattr(i, "sentence_id", "") for i in items if getattr(i, "sentence_id", "")}
        src_count[text] = len(sids) if sids else len(items)

    # Sort by source-count descending so the highest-evidence items are at top
    ordered_texts = sorted(seen_text.keys(), key=lambda t: -src_count.get(t, 0))
    if len(ordered_texts) > max_items_in_prompt:
        ordered_texts = ordered_texts[:max_items_in_prompt]

    lines = []
    for text in ordered_texts:
        rep = seen_text[text][0]
        lines.append(_summarize_skill_item_for_prompt(rep, src_count))
        if include_sentences:
            samples = _sample_sentences_for_item(seen_text[text], max_sentences=max_sentences_per_item)
            for s in samples:
                lines.append(f'    • "{s}"')
    return "\n".join(lines)


def _build_role_distribution_block(
    cluster,
    job_titles: Optional[Dict[str, str]] = None,
    top_n: int = 5,
) -> str:
    """Produce a compact role-distribution block for the prompt, using the
    shared `role_normalization.role_distribution`. Returns empty string when
    no job_titles map is available or no cluster source jobs resolve.

    Job-posting titles are used here ONLY to compute role-family fractions
    (which the LLM uses to spot cross-domain clusters); the raw titles
    themselves are NOT shown, so the LLM cannot read seniority/credentialing
    cues from them.
    """
    if not job_titles:
        return ""
    # Collect unique source job_ids from this cluster's items
    job_ids: set = set()
    for it in cluster.items:
        sid = getattr(it, "sentence_id", "") or ""
        if "_" in sid:
            head, _sep, tail = sid.rpartition("_")
            jid = head if tail.isdigit() else sid
        else:
            jid = sid
        if jid:
            job_ids.add(jid)
    titles_for_cluster = [job_titles.get(j, "") for j in job_ids if j in job_titles]
    if not titles_for_cluster:
        return ""
    try:
        from role_normalization import role_distribution
    except ImportError:
        return ""
    dist = role_distribution(titles_for_cluster)
    if not dist:
        return ""
    n_resolved = sum(n for _, n, _ in dist)
    lines = [f"\nROLE DISTRIBUTION across {n_resolved} source job postings:"]
    for bucket, n, frac in dist[:top_n]:
        lines.append(f"  - {bucket}: {n} postings ({frac*100:.0f}%)")
    if len(dist) > top_n:
        extra = sum(n for _, n, _ in dist[top_n:])
        lines.append(f"  - other role families: {extra} postings")
    lines.append(
        "(Used to spot cross-domain clusters. If the distribution is dominated"
        " by one or two role families, this is a coherent capability area; if"
        " it spans many unrelated families, the cluster may need splitting.)\n"
    )
    return "\n".join(lines)


def build_cluster_prompt(
    cluster,
    config: GeneratorConfig,
    top_soft_skills: Optional[Sequence[str]] = None,
    future_context: Optional[str] = None,
    job_titles: Optional[Dict[str, str]] = None,
) -> str:
    """Build the user-prompt for one cluster.

    Args:
        cluster: a Phase 2.1 `Cluster` object.
        config: GeneratorConfig — controls max_competencies_per_cluster, rationale length.
        top_soft_skills: list of soft-skill texts globally, used as the reference
            list of soft skills the LLM may pull from (not clustered with hard skills).
        future_context: a short paragraph summarizing future-domain weights /
            empirical trends for skills in this cluster (optional).
        job_titles: optional {job_id: title} map used to compute the role
            distribution for this cluster. v8 sprint Phase 2.A.
    """
    soft_skill_list = ""
    if top_soft_skills:
        soft_skill_list = (
            "\nGLOBAL SOFT-SKILL REFERENCE LIST (you may pull 3-6 of these into "
            "soft_skills_required for each competency, choosing those that "
            "support the hard skills in this cluster):\n"
            + ", ".join(top_soft_skills[:30])
            + "\n"
        )

    future_block = ""
    if future_context:
        future_block = f"\nFUTURE-WORK CONTEXT FOR THIS CLUSTER:\n{future_context}\n"

    # v8 sprint: include example sentences per item AND a role-family
    # distribution. Sentences have credentialing/seniority cues stripped.
    skill_lines = _build_skill_lines(
        cluster, include_sentences=True, max_sentences_per_item=2
    )
    role_block = _build_role_distribution_block(cluster, job_titles=job_titles)

    rationale_target = (
        f"{config.min_rationale_chars}-{config.max_rationale_chars} chars (4-6 sentences)"
    )

    return f"""The skills below are grouped into a single semantic CLUSTER from
real job postings. Each line shows a skill/knowledge text plus a sample of
the source sentences that contributed evidence. Job titles, seniority cues,
and degree/credential phrases have been stripped from the sentences so you
judge cognitive complexity from the work itself, not from who employs for it.

CLUSTER ID: {cluster.id}
CLUSTER STREAM: {cluster.stream}
CLUSTER COHESION: {cluster.cohesion_score:.3f} (mean intra-cluster SBERT cosine)
CLUSTER HEURISTIC LABEL: "{cluster.summary_label}"
N ITEMS: {cluster.n_items}  N UNIQUE JOBS: {cluster.n_unique_jobs}

ITEMS IN THIS CLUSTER:
{skill_lines}
{role_block}{soft_skill_list}{future_block}
{KKNI_BRIEF}

YOUR TASK:

Step 1 — In a top-level field called "batch_reasoning", explain your thinking
step-by-step in 3-6 sentences:
  - What sub-themes do you see in this cluster? Use the example sentences
    to confirm what the skills actually MEAN in context.
  - Should they be ONE competency or MULTIPLE? Justify the split. If the
    role distribution spans many unrelated families, lean toward splitting.
  - Are there skills that don't fit any competency? List them and say why
    (dropping weak fits is better than force-including them).
  - Which skills did you keep / drop / merge, and why?

Step 2 — In a top-level field called "competencies", produce 1-{config.max_competencies_per_cluster}
well-anchored competency statements. For each competency, write a "rationale"
of {rationale_target} explaining specifically:
  - Why this title and description match the included skills
  - What a learner could demonstrate to prove mastery
  - Why this competency is distinct from any other competency you produced

RULES (must follow):
  - HARD CAP: at most {config.max_competencies_per_cluster} competencies for this cluster.
  - Every "related_skills" entry MUST be drawn from the ITEMS list above
    verbatim (or very close paraphrase). Do NOT invent skills.
  - Each competency must include AT LEAST {config.min_related_skills_per_competency}
    related_skills (singletons are usually too narrow for a curriculum competency).
  - "title" should be a verb-led learning outcome (e.g. "Implement secure
    authentication flows", "Build modern web application UIs").
  - "description" is 1-3 sentences, concrete and assessable. Avoid HR jargon.
  - Use the example sentences to DISAMBIGUATE skill meaning. Do NOT copy them
    verbatim into descriptions or titles. The description must ABSTRACT from
    the contexts, not quote them.
  - "kkni_level" — judge ONLY by cognitive complexity using the KKNI brief
    above (verbs, decision-autonomy, methodological depth). Do NOT consider
    what education stage employers currently hire at — that is a separate,
    independent label. A KKNI-7 competency can still be in demand at SMK or
    D3; a KKNI-4 competency can still demand S1 hiring. The two never
    influence each other.
  - If the example sentences span multiple unrelated roles (e.g. UX + mechanical
    CAD), produce SEPARATE competencies. Do not synthesise across role
    boundaries.
  - "future_relevance" is 1 sentence describing whether this competency is
    Emerging / Stable / Declining and why.
  - "rationale" length MUST be between {config.min_rationale_chars} and {config.max_rationale_chars} characters.
  - Do NOT start the rationale with "This competency".

OUTPUT — JSON only, no markdown fences, exactly this shape:
{{
  "batch_reasoning": "<your 3-6 sentence chain-of-thought for the whole cluster>",
  "competencies": [
    {{
      "title": "...",
      "description": "...",
      "related_skills": ["<must be from ITEMS>", "..."],
      "rationale": "<{rationale_target}, public-facing, explains why this competency is well-anchored>",
      "soft_skills_required": ["...", "..."],
      "soft_skills_description": "...",
      "future_relevance": "...",
      "future_weight": 0.7,
      "empirical_trend": "Emerging",
      "kkni_level": 5
    }}
  ]
}}"""


# --------------------------------------------------------------------------- #
# JSON parsing — tolerant of the usual LLM quirks
# --------------------------------------------------------------------------- #


_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _parse_llm_json(text: str) -> Optional[dict]:
    """Tolerant JSON parser for LLM output.

    1. Strip leading/trailing whitespace.
    2. Strip ```...``` fences if present.
    3. Try json.loads on the cleaned text.
    4. Fallback: find the outermost {...} block and parse that.
    """
    if not text:
        return None
    text = text.strip()
    text = _JSON_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- #
# LLM call with retry/backoff
# --------------------------------------------------------------------------- #


def _call_llm_for_cluster(
    cluster,
    prompt: str,
    config: GeneratorConfig,
) -> Tuple[Optional[dict], BatchReasoning]:
    """Call the LLM for one cluster.

    Returns (parsed_json | None, BatchReasoning) — the BatchReasoning is always
    populated (raw_response + latency) even when parsing fails, so we can audit.
    """
    client, provider = get_client_for_model(config.model, request_timeout=config.request_timeout_seconds)
    api_model_name = strip_provider_prefix(config.model)

    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    started_at = datetime.now(timezone.utc).isoformat()

    raw_text = ""
    parsed: Optional[dict] = None
    prompt_tokens = completion_tokens = 0
    last_err: Optional[Exception] = None

    t0 = time.time()
    for attempt in range(config.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=api_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            raw_text = resp.choices[0].message.content or ""
            if hasattr(resp, "usage") and resp.usage:
                prompt_tokens = int(getattr(resp.usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(resp.usage, "completion_tokens", 0) or 0)
            parsed = _parse_llm_json(raw_text)
            if parsed is not None:
                break
            logger.warning(
                "cluster %s attempt %d: LLM returned unparseable JSON (%d chars)",
                cluster.id, attempt + 1, len(raw_text),
            )
        except Exception as e:
            last_err = e
            logger.warning("cluster %s attempt %d failed: %s", cluster.id, attempt + 1, e)
        if attempt < config.max_retries:
            time.sleep(config.retry_backoff_base ** (attempt + 1))
    latency = time.time() - t0

    n_comp_out = 0
    batch_reasoning_text = ""
    if isinstance(parsed, dict):
        n_comp_out = len(parsed.get("competencies") or [])
        batch_reasoning_text = str(parsed.get("batch_reasoning") or "")

    br = BatchReasoning(
        id=f"br_{uuid.uuid4().hex[:10]}",
        cluster_id=cluster.id,
        timestamp=started_at,
        provider=provider,
        model=config.model,
        prompt_sha256=prompt_sha,
        batch_reasoning=batch_reasoning_text,
        n_skills_in=cluster.n_items,
        n_competencies_out=n_comp_out,
        raw_response=raw_text,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if parsed is None and last_err:
        logger.error("cluster %s all retries failed: %s", cluster.id, last_err)
    return parsed, br


# --------------------------------------------------------------------------- #
# Per-competency validation + provenance attachment
# --------------------------------------------------------------------------- #


def _canonical_text_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# --------------------------------------------------------------------------- #
# Source-sentence quality filter — drops things that aren't real evidence:
# - too-short fragments (the loader's fallback often emits skill texts here)
# - degree/qualification boilerplate ("Bachelor's degree in CS")
# - noun-phrase fragments with no verb
# - anything identical (or substring of) a related_skill
# Applied AFTER provenance attachment in _attach_provenance_and_validate.
# --------------------------------------------------------------------------- #


_DEGREE_BOILERPLATE_RE = re.compile(
    r"\b(bachelor[''']?s?\s+degree|master[''']?s?\s+degree|phd\b|doctorate|"
    r"\d+\+?\s*years?\s+(of\s+)?(experience|exp)|degree\s+in\s+(computer\s+science|"
    r"engineering|data|mathematics|science|economics))\b",
    re.IGNORECASE,
)

# Lightweight verb heuristic: a real sentence usually contains at least one
# common English verb stem. Empty-of-verb strings are almost always
# noun-phrase fragments ("solve technical problems", "analyze systems")
# that leaked from the skill list into source_sentences.
_COMMON_VERB_RE = re.compile(
    r"\b(is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|"
    r"can|could|should|may|might|must|build|develop|design|implement|create|"
    r"deploy|use|using|work|works|working|manage|lead|support|maintain|"
    r"analyze|operate|integrate|deliver|provide|enable|require|seek|seeking|"
    r"join|looking|drive|ensure|apply|apply|configure|automate|optimize|"
    r"collaborate|communicate|need|needs|needed|own|owns|owned|"
    r"help|helps|focus|focuses|love|loves|believe|grow|growing)\b",
    re.IGNORECASE,
)


def _is_low_quality_sentence(text: str, related_skill_keys: set) -> bool:
    """Return True if `text` should be excluded from source_sentences."""
    if not text:
        return True
    stripped = text.strip()
    if len(stripped) < 30:
        return True
    # Identical or substring of a related_skill (the loader's fallback)
    low = stripped.lower()
    if any(low == k or k in low and len(k) >= len(low) * 0.85 for k in related_skill_keys if k):
        return True
    # Degree / qualification boilerplate
    if _DEGREE_BOILERPLATE_RE.search(stripped):
        return True
    # No verb → almost certainly a noun-phrase fragment
    if not _COMMON_VERB_RE.search(stripped):
        return True
    return False


# Token-Jaccard threshold for collapsing near-duplicate boilerplate sentences.
# Empirically 0.80 catches "key player ..." / "key leader ..." and reused
# AWS / consulting templates without merging genuinely-distinct sentences.
_NEAR_DUP_JACCARD_THRESHOLD = 0.80
_TOKEN_SPLIT_RE = re.compile(r"\w+")


def _sentence_tokens(text: str) -> set:
    """Lower-cased token set used for Jaccard similarity. Drops 1- and
    2-character tokens (whitespace artefacts / function words) so the
    comparison reflects content overlap, not stopword overlap."""
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_SPLIT_RE.findall(text) if len(t) >= 3}


def _dedup_near_duplicate_sentences(sentences: list) -> list:
    """Greedy pairwise Jaccard dedup. For each candidate sentence, drop it if
    its token set has Jaccard >= _NEAR_DUP_JACCARD_THRESHOLD with any sentence
    already kept. Preserves order (first occurrence wins).
    """
    if not sentences or len(sentences) < 2:
        return list(sentences) if sentences else []
    kept: list = []
    kept_tokens: list = []
    for s in sentences:
        st = _sentence_tokens(s)
        if not st:
            kept.append(s)
            kept_tokens.append(st)
            continue
        is_near_dup = False
        for prev in kept_tokens:
            if not prev:
                continue
            inter = len(st & prev)
            union = len(st | prev)
            if union and (inter / union) >= _NEAR_DUP_JACCARD_THRESHOLD:
                is_near_dup = True
                break
        if not is_near_dup:
            kept.append(s)
            kept_tokens.append(st)
    return kept


# ---------------------------------------------------------------------------
# Tier 2D — title evidence audit (product-name literal-presence check)
# ---------------------------------------------------------------------------
# When a competency title mentions a vendor product / framework / language
# (e.g. "Azure DevOps", ".NET Core", "ASP.NET", "React"), the source
# sentences should contain that literal token at least N times. If they
# don't, the title is overreaching what the cluster's evidence supports
# — exactly the failure mode the 2026-05-13 review identified for the
# "Microsoft-based systems" integration competency.
#
# We extract candidate product terms by pattern, not by an exhaustive list:
#   - Capitalised tokens that contain at least one uppercase letter inside
#     the word (e.g. "ASP.NET", "DevOps", "iOS")
#   - Tokens starting with a dot (".NET", ".NET Core")
#   - All-caps acronyms 2-5 chars (AWS, GCP, NLP, AI/ML)
#   - Multi-word brand phrases joined by a space (Azure DevOps, .NET Core)
#
# This is intentionally over-permissive on the extraction side — false
# positives just produce an informational chip, false negatives miss real
# overreach. The min_required threshold defaults to ⌈2⌉ for short source
# pools and scales mildly with size.

# Common verbs / English words that look Title-Cased at title-position but
# aren't brand/proper-noun terms. These are skipped during candidate extraction.
_TITLE_NOISE_TOKENS = frozenset({
    # Verbs (the v2 prompt requires titles to be verb-led)
    "build", "apply", "use", "develop", "design", "train", "coordinate",
    "lead", "implement", "partner", "propose", "select", "organize",
    "manage", "create", "maintain", "support", "deploy", "integrate",
    "explain", "operate", "analyze", "evaluate", "act",
    # Connectors / determiners / generic nouns
    "across", "through", "with", "for", "and", "or", "the", "of", "in", "on",
    "into", "to", "from", "as", "by", "using", "via",
    "software", "application", "applications", "web", "data", "cloud", "major",
    "modern", "standard", "effective", "successful", "multiple", "appropriate",
    "delivery", "team", "teams", "system", "systems", "service", "services",
    "platform", "platforms", "framework", "frameworks",
    "interface", "interfaces", "process", "processes", "workflow", "workflows",
    "model", "models", "modelling", "modeling", "pipeline", "pipelines",
    "method", "methods", "practice", "practices", "pattern", "patterns",
    "tasks", "task", "project", "projects", "outcome", "outcomes",
    "expertise", "context", "best", "lifecycle",
    "small", "large", "complex", "simple",
    # Capitalised English words seen at sentence-initial positions
    "you", "your", "we", "our", "they", "their",
})

# Tokenisation: pull out anything that starts with an alpha and may contain
# common product punctuation (#, +, ., -, /).
_TITLE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9#+./\-]*")


def _normalise_for_match(s: str) -> str:
    """Lowercase + collapse whitespace + strip whitespace around in-token
    punctuation so 'C #' / '.NET Core' / 'ASP. NET' all collapse to the same
    canonical form as the title token. This lets the literal-presence check
    survive the artefacts the loader leaves in source-sentence text."""
    if not s:
        return ""
    low = s.lower()
    # Drop whitespace adjacent to in-word punctuation (PDF-source artefact)
    low = re.sub(r"\s*([#+./\-])\s*", r"\1", low)
    # Collapse runs of whitespace
    low = re.sub(r"\s+", " ", low).strip()
    return low


def _extract_product_terms(title: str) -> list:
    """Return a list of (term, normalised) tuples for the brand / proper-noun
    candidates in a competency title.

    Heuristic: take any Capitalised token whose lower-case form is not in
    `_TITLE_NOISE_TOKENS`. Adjacent capitalised non-noise tokens are chained
    into a multi-word phrase ("Azure DevOps", ".NET Core", "Microsoft-based")
    so we audit the phrase as a unit. Hyphens are kept as part of the term
    (catches "Microsoft-based"). Commas, semicolons, and the literal
    connectives "and", "or", "with", "across", "using" break the chain so we
    don't end up auditing nonsensical compound phrases like "C# ASP.NET"
    when the title actually reads "C#, ASP.NET, and .NET Core".
    """
    if not title:
        return []
    # Split the title into chunks at clause boundaries so we never chain
    # across commas or connectives. Tokens are re-extracted per chunk.
    chunks = re.split(
        r",|;|/|&|\bwith\b|\band\b|\bor\b|\bacross\b|\bin\b|\bthrough\b|\bfor\b|\busing\b|\bvia\b|\bof\b",
        title,
        flags=re.IGNORECASE,
    )
    phrases: list = []
    for chunk in chunks:
        tokens = _TITLE_TOKEN_RE.findall(chunk)
        # Mark each as capitalised + non-noise
        keepers: list = []
        for tok in tokens:
            is_cap = tok[0].isupper() if tok else False
            is_noise = tok.lower() in _TITLE_NOISE_TOKENS
            # Single-letter or 2-char tokens are rarely brand names by themselves
            # (KKNI, AWS, GCP, AI, ML are all 2-4 chars all-caps — keep those).
            is_long_enough = len(tok) >= 3 or tok.isupper()
            keepers.append((tok, is_cap and not is_noise and is_long_enough))

        # Chain consecutive keepers into phrases (within this chunk only)
        cur: list = []
        for tok, keep in keepers:
            if keep:
                cur.append(tok)
            else:
                if cur:
                    phrases.append(" ".join(cur))
                    cur = []
        if cur:
            phrases.append(" ".join(cur))

    out: list = []
    seen: set = set()
    for phrase in phrases:
        norm = _normalise_for_match(phrase)
        if not norm or norm in seen:
            continue
        # Reject single-character residues (e.g. "X" alone) but keep "AI", "ML"
        if len(norm) < 2:
            continue
        seen.add(norm)
        out.append((phrase, norm))
        # Also audit individual sub-tokens of a multi-word phrase so we catch
        # cases where (say) "Microsoft" is named broadly but "Azure DevOps" is
        # the full phrase.
        sub_tokens = phrase.split()
        if len(sub_tokens) > 1:
            for st in sub_tokens:
                if st.lower() in _TITLE_NOISE_TOKENS:
                    continue
                st_norm = _normalise_for_match(st)
                if st_norm and st_norm not in seen and len(st_norm) >= 2:
                    seen.add(st_norm)
                    out.append((st, st_norm))
    return out


def _audit_title_evidence(
    title: str,
    source_sentences: list,
    min_required: int = 2,
) -> list:
    """Return a list of title_evidence_concerns dicts. Each entry records one
    product/proper-noun term in the title, how many source sentences contain
    it literally (after whitespace/punctuation normalisation), the threshold,
    and a severity tag.
    """
    terms = _extract_product_terms(title)
    if not terms:
        return []
    sents_norm = [_normalise_for_match(s) for s in (source_sentences or [])]
    concerns: list = []
    n_sents = len(sents_norm)
    if n_sents == 0:
        return []
    # Scale min-required mildly with source pool size
    threshold = max(min_required, n_sents // 4)
    for term, key in terms:
        hits = sum(1 for s in sents_norm if key in s)
        if hits < threshold:
            severity = "warning" if hits == 0 else "info"
            concerns.append({
                "term": term,
                "literal_hits": hits,
                "n_sentences": n_sents,
                "min_required": threshold,
                "severity": severity,
            })
    return concerns


def _build_text_to_items_index(cluster) -> Dict[str, list]:
    """Map canonical-text → list of underlying SkillItem / KnowledgeItem."""
    idx: Dict[str, list] = {}
    for it in cluster.items:
        k = _canonical_text_key(getattr(it, "text", ""))
        if k:
            idx.setdefault(k, []).append(it)
    return idx


def _job_id_from_sentence_id(sid: str) -> str:
    if not sid or "_" not in sid:
        return sid or ""
    head, _, tail = sid.rpartition("_")
    return head if tail.isdigit() else sid


def _attach_provenance_and_validate(
    raw_comp: dict,
    cluster,
    config: GeneratorConfig,
    batch_reasoning_id: str,
) -> Tuple[Optional[CompetencyV2], List[str]]:
    """Convert an LLM-emitted competency dict into a fully-populated CompetencyV2.

    Returns (competency_or_None, list_of_violation_messages).
    """
    violations: List[str] = []

    # Surface required fields
    title = str(raw_comp.get("title", "")).strip()
    description = str(raw_comp.get("description", "")).strip()
    related = raw_comp.get("related_skills") or []
    if not isinstance(related, list):
        violations.append("related_skills is not a list")
        related = []
    related = [str(s).strip() for s in related if str(s).strip()]

    rationale = str(raw_comp.get("rationale", "")).strip()
    soft_required = raw_comp.get("soft_skills_required") or []
    if not isinstance(soft_required, list):
        soft_required = []
    soft_required = [str(s).strip() for s in soft_required if str(s).strip()]
    soft_desc = str(raw_comp.get("soft_skills_description", "")).strip()
    future_relevance = str(raw_comp.get("future_relevance", "")).strip()
    future_weight = raw_comp.get("future_weight", 0.0)
    try:
        future_weight = float(future_weight)
    except (TypeError, ValueError):
        future_weight = 0.0
    empirical_trend = str(raw_comp.get("empirical_trend", "Stable")).strip() or "Stable"
    kkni_level = raw_comp.get("kkni_level")
    try:
        kkni_level = int(kkni_level) if kkni_level is not None else None
    except (TypeError, ValueError):
        kkni_level = None

    # Anti-hallucination: every related_skill must map to a cluster item.
    # Also dedupe case-variant / near-duplicate surface forms (e.g. "Machine
    # Learning" + "machine learning" + "AI/ML" + "Artificial Intelligence"
    # would otherwise all show up in the list).
    text_idx = _build_text_to_items_index(cluster)
    matched_items: list = []
    matched_skill_texts: List[str] = []
    seen_canonical_keys: set = set()
    for skill_text in related:
        k = _canonical_text_key(skill_text)
        if not k or k in seen_canonical_keys:
            continue  # case-variant duplicate of an already-kept skill
        if k in text_idx:
            matched_items.extend(text_idx[k])
            matched_skill_texts.append(skill_text)
            seen_canonical_keys.add(k)
        else:
            # Fuzzy match — substring containment
            hit = None
            hit_key = None
            for cand_k, items in text_idx.items():
                if cand_k == k or k in cand_k or cand_k in k:
                    hit = items
                    hit_key = cand_k
                    break
            if hit:
                matched_items.extend(hit)
                matched_skill_texts.append(skill_text)
                seen_canonical_keys.add(k)
                if hit_key:
                    seen_canonical_keys.add(hit_key)

    grounding_preview = (
        len(matched_skill_texts) / max(1, len(related)) if related else 0.0
    )

    # Build provenance. For source_sentences, prefer sentences that actually
    # mention at least one of the kept related_skills (drops boilerplate like
    # "The Mission Kong is scaling rapidly" from job postings whose other
    # sentences happen to match).
    related_skill_keys = {
        _canonical_text_key(s) for s in matched_skill_texts if s
    }
    contributing_item_ids: List[str] = []
    source_job_ids: set = set()
    source_sentence_ids: set = set()
    source_sentences_keep: list = []           # ordered, deduped
    source_sentences_fallback: list = []       # used when no skill-mentioning sentence is available
    seen_sentences: set = set()
    for it in matched_items:
        sid = getattr(it, "sentence_id", "") or ""
        if sid:
            source_sentence_ids.add(sid)
            jid = _job_id_from_sentence_id(sid)
            if jid:
                source_job_ids.add(jid)
        stxt = getattr(it, "sentence_text", "") or ""
        if stxt and stxt not in seen_sentences:
            seen_sentences.add(stxt)
            # Quality filter (post-2026-05-12 honesty audit) — drop fragments,
            # boilerplate, and skill-text proxies before they reach provenance.
            if _is_low_quality_sentence(stxt, related_skill_keys):
                continue
            stxt_low = stxt.lower()
            # Prefer skill-mentioning sentences; the rest are fallbacks.
            if any(k and k in stxt_low for k in related_skill_keys):
                source_sentences_keep.append(stxt)
            else:
                source_sentences_fallback.append(stxt)
        # Synthesize a stable item id: hash(text + sentence_id) — items don't
        # carry an .id attribute in the existing pipeline schema.
        item_id_seed = f"{getattr(it,'text','')}\x00{sid}"
        contributing_item_ids.append(
            "item_" + hashlib.sha256(item_id_seed.encode("utf-8")).hexdigest()[:10]
        )
    # Prefer skill-mentioning sentences; fall back to all if none match
    if source_sentences_keep:
        source_sentences = source_sentences_keep
    else:
        source_sentences = source_sentences_fallback

    # Tier 2D near-duplicate dedup (2026-05-13). Job postings frequently reuse
    # boilerplate phrasing ("key player driving customer success" vs "key leader
    # driving customer success"). Exact-string seen_sentences misses these and
    # they inflate the grounding score by counting one template twice. Drop
    # sentences whose token-Jaccard against any earlier kept sentence exceeds
    # NEAR_DUP_THRESHOLD.
    source_sentences = _dedup_near_duplicate_sentences(source_sentences)
    # Dedup item_ids while preserving order
    seen_item_ids = set()
    contributing_item_ids = [
        x for x in contributing_item_ids if not (x in seen_item_ids or seen_item_ids.add(x))
    ]

    # Tier 2D — title evidence audit. Flags brand/proper-noun terms in the
    # title that don't have enough literal hits in source sentences.
    title_concerns = _audit_title_evidence(title, source_sentences)

    comp = CompetencyV2(
        id=f"comp_{uuid.uuid4().hex[:8]}",
        title=title,
        description=description,
        related_skills=matched_skill_texts,
        soft_skills_required=soft_required,
        soft_skills_description=soft_desc,
        rationale=rationale,
        batch_reasoning_id=batch_reasoning_id,
        contributing_item_ids=contributing_item_ids,
        source_job_ids=sorted(source_job_ids),
        source_sentence_ids=sorted(source_sentence_ids),
        source_sentences=list(source_sentences)[:20],  # cap for UI; ordering preserved
        title_evidence_concerns=title_concerns,
        future_relevance=future_relevance,
        future_weight=future_weight,
        empirical_trend=empirical_trend,
        kkni_level=kkni_level,
        grounding_score_preview=grounding_preview,
        grounding_flagged=(grounding_preview < 0.80),
        generated_at=datetime.now(timezone.utc).isoformat(),
        cluster_id=cluster.id,
        cluster_summary_label=cluster.summary_label,
    )

    violations.extend(
        comp.validate_provenance(
            min_rationale=config.min_rationale_chars,
            max_rationale=config.max_rationale_chars,
        )
    )
    if violations:
        return None, violations
    return comp, violations


# --------------------------------------------------------------------------- #
# Dedup with provenance merging
# --------------------------------------------------------------------------- #


def _norm_title_key(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {_canonical_text_key(x) for x in a}
    sb = {_canonical_text_key(x) for x in b}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _merge_into(survivor: CompetencyV2, absorbed: CompetencyV2) -> CompetencyV2:
    """Merge `absorbed` into `survivor` in place.

    Updated 2026-05-13 after the n1k corpus showed that unioning source_sentences
    on every merge breaks the title↔sentences alignment (the survivor's title
    represents one of N merged sub-competencies, so most of the absorbed
    sentences become "off-topic" under the Phase-2.5 grounding gate).

    Merge policy:
      - related_skills + contributing_item_ids + source_job_ids: UNION
        (broader provenance is honest — these jobs DID contribute extracted skills)
      - source_sentences: KEEP SURVIVOR'S ONLY
        (so sentence_relevance stays aligned with the surviving title)
      - source_sentence_ids: same — keep survivor's only
      - rationale + title: keep survivor's (it was the higher-ranked one by
        virtue of being first in iteration order)
    """
    survivor.related_skills = list(dict.fromkeys(survivor.related_skills + absorbed.related_skills))
    survivor.contributing_item_ids = list(
        dict.fromkeys(survivor.contributing_item_ids + absorbed.contributing_item_ids)
    )
    survivor.source_job_ids = sorted(set(survivor.source_job_ids) | set(absorbed.source_job_ids))
    # Sentences + sentence_ids: keep survivor's; do NOT union with absorbed.
    # (See merge-policy comment above.)
    survivor.soft_skills_required = list(
        dict.fromkeys(survivor.soft_skills_required + absorbed.soft_skills_required)
    )
    survivor.merged_from = sorted(set(survivor.merged_from + [absorbed.id]))
    survivor.future_weight = max(survivor.future_weight, absorbed.future_weight)
    return survivor


def _dedup_competencies(
    competencies: List[CompetencyV2],
    config: GeneratorConfig,
) -> List[CompetencyV2]:
    """Three-stage dedup, mirroring legacy but with provenance merging."""
    if len(competencies) <= 1:
        return competencies

    # Stage 1: exact-normalized-title match (fast win)
    by_norm_title: Dict[str, CompetencyV2] = {}
    for c in competencies:
        k = _norm_title_key(c.title)
        if k in by_norm_title:
            _merge_into(by_norm_title[k], c)
        else:
            by_norm_title[k] = c
    stage1 = list(by_norm_title.values())

    # Stage 2: skill-overlap (Jaccard) — pairwise greedy
    survivors: List[CompetencyV2] = []
    for c in stage1:
        absorbed = False
        for s in survivors:
            jac = _jaccard(c.related_skills, s.related_skills)
            if jac >= config.dedup_jaccard_skills_threshold:
                _merge_into(s, c)
                absorbed = True
                break
        if not absorbed:
            survivors.append(c)

    # Stage 3: semantic-title via SBERT. Defer to a lightweight check — only
    # try if sentence-transformers is importable; otherwise skip (Stage 2 is
    # sufficient in practice). This keeps the v2 generator usable without
    # SBERT at import time.
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        titles = [c.title for c in survivors]
        embs = model.encode(titles, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sim = embs @ embs.T
        absorbed_idx: set = set()
        for i in range(len(survivors)):
            if i in absorbed_idx:
                continue
            for j in range(i + 1, len(survivors)):
                if j in absorbed_idx:
                    continue
                if sim[i, j] >= config.dedup_semantic_title_threshold:
                    _merge_into(survivors[i], survivors[j])
                    absorbed_idx.add(j)
        survivors = [c for i, c in enumerate(survivors) if i not in absorbed_idx]
    except Exception as e:
        logger.info("SBERT semantic-title dedup skipped (%s); Stage 1+2 only", e)

    # Final cap
    if len(survivors) > config.final_competency_cap:
        # Keep the ones with highest grounding_score_preview, then most contributing items
        survivors.sort(
            key=lambda c: (c.grounding_score_preview, len(c.contributing_item_ids)),
            reverse=True,
        )
        survivors = survivors[: config.final_competency_cap]

    return survivors


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def generate_competencies_v2(
    clusters: Sequence,
    config: Optional[GeneratorConfig] = None,
    top_soft_skills: Optional[Sequence[str]] = None,
    future_weights: Optional[Dict[str, dict]] = None,
    apply_grounding_gate: bool = True,
    job_titles: Optional[Dict[str, str]] = None,
) -> Tuple[List[CompetencyV2], List[BatchReasoning]]:
    """Cluster-driven competency generation.

    Args:
        clusters: list of Phase 2.1 Cluster objects.
        config: GeneratorConfig (default: gpt-5.4-mini via Jatevo).
        top_soft_skills: optional list of soft-skill texts to inject as a
            reference list into every prompt.
        future_weights: optional dict[skill_text -> {future_weight, domain, trend}]
            to enrich the future-context block per cluster.
        apply_grounding_gate: if True, run Phase 2.5 evaluator and return ONLY
            competencies that pass the grounding gate. Failing competencies
            are still mutated with their grounding fields so the caller can
            audit. Default True (production). Set False for ablation runs.
        job_titles: optional {job_id: title} map. When provided, the prompt
            includes a per-cluster role-family distribution computed via
            `role_normalization.role_distribution`. v8 sprint Phase 2.A.

    Returns:
        (competencies, batch_reasonings)
    """
    if config is None:
        config = GeneratorConfig()

    if top_soft_skills:
        top_soft_skills = list(top_soft_skills)[:30]

    # Only process hard_plus_knowledge clusters by default. Soft-skill clusters
    # feed the top_soft_skills reference list (built by caller).
    hpk_clusters = [c for c in clusters if c.stream == "hard_plus_knowledge"]
    logger.info(
        "generating competencies for %d hard+knowledge clusters (of %d total)",
        len(hpk_clusters), len(clusters),
    )

    raw_competencies: List[CompetencyV2] = []
    batch_reasonings: List[BatchReasoning] = []
    dropped_audit: List[Tuple[str, str, List[str]]] = []  # (cluster_id, comp_title, violations)

    for cluster in hpk_clusters:
        # Build optional future-context block from future_weights
        future_context = None
        if future_weights:
            hits = []
            for it in cluster.items[:10]:
                t = getattr(it, "text", "")
                if t in future_weights:
                    fw = future_weights[t]
                    hits.append(f"  • {t}: weight={fw.get('future_weight', 0):.2f}, "
                                f"domain={fw.get('domain', '?')}, "
                                f"trend={fw.get('trend', '?')}")
            if hits:
                future_context = "\n".join(hits)

        prompt = build_cluster_prompt(
            cluster, config,
            top_soft_skills=top_soft_skills,
            future_context=future_context,
            job_titles=job_titles,
        )
        parsed, br = _call_llm_for_cluster(cluster, prompt, config)
        batch_reasonings.append(br)
        if parsed is None:
            logger.warning("cluster %s: LLM call failed or unparseable; skipping", cluster.id)
            continue

        for raw_comp in (parsed.get("competencies") or [])[: config.max_competencies_per_cluster]:
            comp, violations = _attach_provenance_and_validate(
                raw_comp, cluster, config, batch_reasoning_id=br.id
            )
            if comp is None:
                dropped_audit.append(
                    (cluster.id, str(raw_comp.get("title", ""))[:80], violations)
                )
                continue
            comp.provider = br.provider
            comp.model = br.model
            raw_competencies.append(comp)

    logger.info(
        "produced %d valid competencies pre-dedup (dropped %d for invariant violations)",
        len(raw_competencies), len(dropped_audit),
    )

    survivors = _dedup_competencies(raw_competencies, config)
    logger.info("after dedup: %d competencies", len(survivors))

    if apply_grounding_gate and survivors:
        from competency_evaluator import EvaluatorConfig, evaluate_competencies

        passing, failing, eval_report = evaluate_competencies(
            survivors, config=EvaluatorConfig()
        )
        logger.info(
            "Phase 2.5 grounding gate: %d/%d pass (mean=%.3f, min=%.3f)",
            eval_report.n_passed, eval_report.n_evaluated,
            eval_report.grounding_mean, eval_report.grounding_min,
        )
        survivors = passing  # canonical Req-7 contract: failing competencies are dropped

    return survivors, batch_reasonings
