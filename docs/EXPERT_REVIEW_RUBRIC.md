# Expert Review Rubric

This document provides objective criteria for reviewers to judge extracted **skills**, **knowledge**, and **competencies** in the expert review UI. Use these guidelines to make consistent, defensible judgments.

---

## 1. Skills Review

### Valid?

**Valid** = The item is a real, actionable skill or competence mentioned (or clearly implied) in the job posting text.

Choose **Valid** when:
- The skill text appears in or is directly supported by the job description
- The skill is something a candidate would need or use on the job (e.g., "design REST APIs", "communicate with stakeholders")
- The skill is a meaningful phrase (not a fragment or single vague word)
- You can point to specific wording in the job text that supports it

Choose **Invalid** when:
- The text is **garbage** or extraction noise (e.g., "and", "detail", "experience")
- The text is a **fragment** (e.g., "of the", "in a")
- The text is **not a skill** (e.g., a technology name alone without action, a degree requirement, a company name)
- The skill was **hallucinated**—not present or implied in the job text
- The skill is **too vague** to be useful (e.g., "skills", "abilities")

**When in doubt:** Use **Invalid** (conservative). It is better to exclude borderline items than to inflate the skill set with noise.

### Type (corrected)

- **Hard**: Technical, domain-specific, measurable (e.g., "implement CI/CD pipelines", "debug SQL")
- **Soft**: Interpersonal, transferable (e.g., "communicate effectively", "work in a team")
- **Both**: Hybrid (e.g., "technical communication", "explain complex concepts to non-technical stakeholders")

### ~~Bloom (corrected)~~

> **REMOVED in pipeline-redesign-v2 Phase 1.3 (Req 1).** Bloom-level
> correction is no longer collected from reviewers. Bloom-level decisions
> are returned to curriculum stakeholders per the requirements doc; the
> reviewer's job is now limited to validity + type correction. The
> downstream KKNI labeler (Phase 2.3) assigns KKNI level 1–9 post-hoc
> via SBERT match against Perpres 8/2012 descriptors and does not depend
> on a reviewer-supplied Bloom level.

---

## 2. Knowledge Review

### Valid?

**Valid** = The item is a real concept, technology, tool, or domain knowledge mentioned in the job posting.

Choose **Valid** when:
- The knowledge item appears in or is clearly implied by the job text
- It represents tools, technologies, platforms, or theoretical concepts (e.g., "Python", "React", "cloud computing")
- It is **not** a skill phrased as action (those belong in skills)
- It is **not** an educational degree (bachelor, master, PhD, diploma)

Choose **Invalid** when:
- Garbage or extraction noise
- Fragment or hallucination
- Actually a skill (verb phrase) misclassified as knowledge
- Degree or certification requirement
- Too vague (e.g., "systems", "solutions" with no context)

**Judgment focus:** Extraction correctness only. Ignore Domain, Trend, and Weight—those are for downstream analysis.

---

## 3. Competency Review

Competencies are curriculum-style statements generated from skills. Each has a **title**, **description**, **related skills**, **future relevance** note, and (Phase 2.2 onward) **provenance**: `contributing_item_ids` and `source_sentences` linking back to the job-posting sentences that produced it.

**Use the provenance click-through.** Before rating, expand the source sentences. The competency is supposed to be a faithful synthesis of those skills/knowledge items extracted from those sentences. If it isn't, the validity and relevance scores should reflect that.

### Four-dimensional rating (per competency)

| Dimension | Scale | Question |
|---|---|---|
| **Validity** | 1–5 Likert | "This is a well-defined competency that a software engineer would have." |
| **Relevance** | 1–5 Likert | "This competency is relevant for the target curriculum (SMK / D3 / D4 / undergraduate IT)." |
| **Specificity** | 1–5 Likert | "This competency is specific enough that learner progress against it can be assessed." |
| **Recommend** | Yes / No | "Would you recommend including this competency in the curriculum?" |

**Likert scale (1–5):** 1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree.

**Detailed anchors per Likert level:**

| Score | Validity | Relevance | Specificity |
|---|---|---|---|
| 1 | Incoherent, mislabeled, or fabricated | Off-domain or unrelated to SE/IT | Untestable, abstract, or aspirational |
| 2 | Partially well-formed but with significant issues | Tangentially related; weak fit | Vague — progress could not be measured |
| 3 | Acceptable; usable with minor revisions | Reasonable fit for the curriculum | Measurable but with some ambiguity |
| 4 | Well-formed and operational | Clear fit for the target curriculum | Clearly assessable |
| 5 | Exemplary — clear, integrative, curriculum-ready | Highly relevant and central to the curriculum | Precisely scoped — assessment design is obvious |

**Recommend (Yes/No):** holistic gate. Yes means you would actually advocate for including this in a curriculum (not just that it scored well on the Likert dimensions). No means despite any Likert scores, you would not include it.

### Inter-rater reliability protocol

When 3+ reviewers rate the same competencies:

- **Fleiss' Kappa** computed on the 1–5 Likert dimensions (3+ raters, ordinal data)
- **Cohen's Kappa pairwise** (averaged) for the Yes/No `Recommend` dimension
- **Free-marginal Kappa (Randolph's)** reported alongside Fleiss' when the marginal distributions are highly skewed (e.g., almost all "Recommend = Yes") to address the Cohen's Kappa paradox in unbalanced data

**Acceptance thresholds (Landis & Koch 1977 — standard benchmark in education research):**

| Kappa range | Interpretation | Action |
|---|---|---|
| < 0.20 | Slight | Rejected; revise rubric and re-rate |
| 0.21 – 0.40 | Fair | Marginal; flag in paper's limitations section |
| 0.41 – 0.60 | **Moderate** | **Minimum acceptable threshold** |
| 0.61 – 0.80 | Substantial | Target |
| 0.81 – 1.00 | Almost perfect | Excellent |

**Target: Fleiss' Kappa ≥ 0.60** (substantial agreement) on each Likert dimension. Below 0.41 triggers protocol revision and re-rating. Between 0.41 and 0.60 ships with caveat.

### Reviewer setup (default for v2 user-testing window)

- 3 reviewers
- Each reviewer rates **75 competencies**: 50 unique to that reviewer + 25 shared across all three (IRR computed on the 25 shared)
- Total unique competencies evaluated: 50 × 3 + 25 = **175**
- Workload estimate: ~30s per dimension per competency (with provenance click-through) → ~2.5 hours per reviewer, split across two sessions

### Legacy 2-dimensional rubric

The previous `human_quality` (1–5) + `human_relevant` (yes/no/partial) schema is still readable by `import_feedback.py`. New reviews should use the 4-dimensional schema. Existing 2-dim feedback is mapped: `human_quality → human_validity`, `human_relevant → human_recommend` (partial → no).

---

## 4. Summary: Quick Reference

| Item        | Field        | Valid / High                          | Invalid / Low                             |
|-------------|--------------|---------------------------------------|-------------------------------------------|
| Skill       | Valid?       | Real skill from job text              | Garbage, fragment, hallucination, not a skill |
| Knowledge   | Valid?       | Real concept/tool from job text       | Garbage, fragment, hallucination, degree |
| Competency  | Quality 1–5  | 4–5 = curriculum-ready                | 1–2 = unusable or weak                    |
| Competency  | Relevant?    | Yes = aligns with skills + curriculum | No = misaligned or not relevant           |

---

## 5. Notes Field

Use the **Notes** field for:
- Edge cases or borderline judgments
- Suggested corrections (e.g., better skill type, span boundary, or canonical wording)
- Context that affected your decision (for multi-reviewer consistency)
