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

Competencies are curriculum-style statements generated from skills. Each has a **title**, **description**, **related skills**, and **future relevance** note.

### Quality (1–5)

Rate how well the competency is written and how suitable it is for curriculum use.

| Score | Label     | Meaning |
|-------|-----------|---------|
| **1** | Poor      | Unusable: incoherent, wrong grouping, misleading, or trivial |
| **2** | Fair      | Weak: unclear, partly wrong, or too narrow to be useful |
| **3** | Good      | Acceptable: clear, reasonable grouping, usable with minor edits |
| **4** | Good      | Strong: clear, well-structured, good level for curriculum |
| **5** | Excellent | Outstanding: exemplary statement, integrative, curriculum-ready |

**When to choose each:**
- **1**: The competency does not make sense, groups unrelated skills, or is factually wrong.
- **2**: The competency is partially correct but has significant issues (vague, incomplete, or misaligned with skills).
- **3**: The competency is usable as-is. Minor wording improvements possible but not critical.
- **4**: The competency is well written and would fit directly into a curriculum.
- **5**: The competency is exemplary—clear, integrative, and clearly maps to the related skills.

### Relevant?

**Relevance = How well the competency aligns with its related skill set and the target curriculum domain.**

Choose **Yes** when:
- The competency **accurately groups and represents** its related skills
- The competency is **relevant to the curriculum domain** (e.g., Software & Game Development)
- A learner achieving this competency would legitimately possess the related skills
- The competency would be useful for curriculum design in the target domain

Choose **Partial** when:
- Some related skills fit; others do not
- The competency is relevant but too narrow or too broad for the skill set
- Minor misalignment between title/description and related skills

Choose **No** when:
- The competency **does not align** with its related skills (wrong grouping, mismatched scope)
- The competency is **not relevant** to the curriculum domain
- The competency would mislead curriculum designers about what the skills actually imply

**Key question:** *Is this competency a faithful and useful representation of these skills for curriculum purposes?*

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
