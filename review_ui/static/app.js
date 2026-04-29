const API = "/api";
let showUnreviewedOnly = false;
let reviewerId = "";

function getReviewerId() {
  if (reviewerId) return reviewerId;
  const params = new URLSearchParams(window.location.search);
  reviewerId = params.get("reviewer_id") || params.get("reviewer") || "default";
  return reviewerId;
}

function apiUrl(path) {
  const rid = getReviewerId();
  const sep = path.includes("?") ? "&" : "?";
  return `${API}${path}${sep}reviewer_id=${encodeURIComponent(rid)}`;
}

function showPanel(name) {
  document
    .querySelectorAll(".panel")
    .forEach((p) => p.classList.remove("active"));
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.remove("active"));
  document.getElementById(`${name}-panel`).classList.add("active");
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add("active");
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => showPanel(btn.dataset.tab));
});

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

async function postJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

// Delegate Show more / Show less clicks to a single handler so all panels
// (skills, knowledge) behave consistently and avoid duplicate handlers.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".toggle-job-text");
  if (!btn) return;
  const ctx = btn.closest(".job-context");
  if (!ctx) return;
  const expanded = ctx.classList.toggle("expanded");
  btn.textContent = expanded ? "Show less" : "Show more";
  btn.setAttribute("aria-expanded", expanded);
});

function isSkillReviewed(item) {
  return !!(
    item.human_valid ||
    item.human_type ||
    item.human_bloom ||
    item.human_notes
  );
}
function isKnowledgeReviewed(item) {
  return !!(item.human_valid || item.human_notes);
}
function isCompetencyReviewed(item) {
  return !!(
    item.human_quality ||
    item.human_relevant ||
    item.human_skill_focus ||
    item.human_notes
  );
}

// --- Debounced auto-save ---
const saveTimers = {};
function debouncedSave(key, fn, delay = 800) {
  if (saveTimers[key]) clearTimeout(saveTimers[key]);
  saveTimers[key] = setTimeout(fn, delay);
}

function showSaveIndicator(card, success) {
  let dot = card.querySelector(".save-dot");
  if (!dot) {
    dot = document.createElement("span");
    dot.className = "save-dot";
    card.prepend(dot);
  }
  dot.textContent = success ? "Saved" : "Error";
  dot.style.color = success ? "#16a34a" : "#dc2626";
  setTimeout(() => {
    dot.textContent = "";
  }, 2000);
}

// --- Skills ---
let skillsData = [];

async function autoSaveSkill(reviewId, card) {
  const hv =
    card.querySelector(`select[data-field="human_valid"]`)?.value || "";
  const ht = card.querySelector(`select[data-field="human_type"]`)?.value || "";
  const hb =
    card.querySelector(`select[data-field="human_bloom"]`)?.value || "";
  const hn =
    card.querySelector(`textarea[data-field="human_notes"]`)?.value || "";
  try {
    await postJson(`${API}/save_skill_feedback`, {
      review_id: reviewId,
      human_valid: hv,
      human_type: ht,
      human_bloom: hb,
      human_notes: hn,
      reviewer_id: getReviewerId(),
    });
    const item = skillsData.find((x) => x.review_id === reviewId);
    if (item) {
      item.human_valid = hv;
      item.human_type = ht;
      item.human_bloom = hb;
      item.human_notes = hn;
    }
    updateSkillProgress();
    card.classList.toggle("reviewed", !!(hv || hb || hn));
    showSaveIndicator(card, true);
    // Don't re-render on save when unreviewed-only: keep card visible so user can
    // add Type, Bloom, Notes before moving on (hide only on "Next Unreviewed").
  } catch (e) {
    showSaveIndicator(card, false);
  }
}

function updateSkillProgress() {
  const reviewed = skillsData.filter(isSkillReviewed).length;
  const pct = skillsData.length
    ? Math.round((reviewed / skillsData.length) * 100)
    : 0;
  document.getElementById("skills-progress").textContent =
    `${reviewed} / ${skillsData.length} (${pct}%)`;
  const bar = document.getElementById("skills-bar");
  if (bar) bar.style.width = pct + "%";
}

function renderSkills() {
  const list = document.getElementById("skills-list");
  list.innerHTML = "";
  const items = showUnreviewedOnly
    ? skillsData.filter((i) => !isSkillReviewed(i))
    : skillsData;
  items.forEach((item) => {
    const reviewed = isSkillReviewed(item);
    const card = document.createElement("div");
    card.className = "card" + (reviewed ? " reviewed" : "");

    // Build job context HTML with collapsible preview if text is long
    let jobHtml = "";
    if (item.job_text) {
      const text = item.job_text || "";
      if (text.length > 800) {
        jobHtml = `<div class="job-context"><strong>Job context:</strong><blockquote><span class="job-preview">${escapeHtml(
          text.slice(0, 800),
        )}</span><span class="ellipsis">…</span><span class="job-full">${escapeHtml(text)}</span><button type="button" class="toggle-job-text" aria-expanded="false">Show more</button></blockquote></div>`;
      } else {
        jobHtml = `<div class="job-context"><strong>Job context:</strong><blockquote>${escapeHtml(text)}</blockquote></div>`;
      }
    }

    card.innerHTML = `
      <div class="card-header">
        <div class="label">Skill</div>
        <div class="value">${escapeHtml(item.skill || "")}</div>
      </div>
      <div class="card-meta">
        <span class="meta-tag">Type: ${escapeHtml(item.type || "—")}</span>
        <span class="meta-tag">Bloom: ${escapeHtml(item.bloom || "—")}</span>
        <span class="meta-tag">Confidence: ${item.confidence_score || "—"}</span>
        <span class="meta-tag">Source: ${escapeHtml(item.source || "—")}</span>
      </div>
      ${jobHtml}
      <div class="row">
        <div class="col">
          <div class="label">Valid?</div>
          <select data-field="human_valid">
            <option value="">--</option>
            <option value="valid" ${item.human_valid === "valid" ? "selected" : ""}>Valid</option>
            <option value="invalid" ${item.human_valid === "invalid" ? "selected" : ""}>Invalid</option>
          </select>
        </div>
        <div class="col">
          <div class="label">Type (corrected)</div>
          <select data-field="human_type">
            <option value="">--</option>
            <option value="Hard" ${item.human_type === "Hard" ? "selected" : ""}>Hard</option>
            <option value="Soft" ${item.human_type === "Soft" ? "selected" : ""}>Soft</option>
            <option value="Both" ${item.human_type === "Both" ? "selected" : ""}>Both (hybrid)</option>
          </select>
        </div>
        <div class="col">
          <div class="label">Bloom (corrected)</div>
          <select data-field="human_bloom">
            <option value="">--</option>
            <option value="Remember" ${item.human_bloom === "Remember" ? "selected" : ""}>Remember</option>
            <option value="Understand" ${item.human_bloom === "Understand" ? "selected" : ""}>Understand</option>
            <option value="Apply" ${item.human_bloom === "Apply" ? "selected" : ""}>Apply</option>
            <option value="Analyze" ${item.human_bloom === "Analyze" ? "selected" : ""}>Analyze</option>
            <option value="Evaluate" ${item.human_bloom === "Evaluate" ? "selected" : ""}>Evaluate</option>
            <option value="Create" ${item.human_bloom === "Create" ? "selected" : ""}>Create</option>
            <option value="N/A" ${item.human_bloom === "N/A" ? "selected" : ""}>N/A</option>
          </select>
        </div>
      </div>
      <div class="label">Notes</div>
      <textarea data-field="human_notes" placeholder="Optional notes">${escapeHtml(item.human_notes || "")}</textarea>
      <button type="button" class="save-now">Save Skill Review</button>
      <span class="hint-inline">Changes auto-save. Click to save immediately.</span>
    `;
    const rid = item.review_id;
    card
      .querySelector(".save-now")
      ?.addEventListener("click", () => autoSaveSkill(rid, card));
    card.querySelectorAll("select").forEach((sel) => {
      sel.addEventListener("change", () =>
        debouncedSave(`skill-${rid}`, () => autoSaveSkill(rid, card), 300),
      );
    });
    card.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () =>
        debouncedSave(`skill-${rid}`, () => autoSaveSkill(rid, card)),
      );
    });
    list.appendChild(card);
    // Toggle handling is done via delegated listener (see above).
  });
  updateSkillProgress();
}

// --- Knowledge ---
let knowledgeData = [];

async function autoSaveKnowledge(reviewId, card) {
  const hv =
    card.querySelector(`select[data-field="human_valid"]`)?.value || "";
  const hn =
    card.querySelector(`textarea[data-field="human_notes"]`)?.value || "";
  try {
    await postJson(`${API}/save_knowledge_feedback`, {
      review_id: reviewId,
      human_valid: hv,
      human_notes: hn,
      reviewer_id: getReviewerId(),
    });
    const item = knowledgeData.find((x) => x.review_id === reviewId);
    if (item) {
      item.human_valid = hv;
      item.human_notes = hn;
    }
    updateKnowledgeProgress();
    card.classList.toggle("reviewed", !!(hv || hn));
    showSaveIndicator(card, true);
    // Don't re-render on save when unreviewed-only: keep card visible for notes.
  } catch (e) {
    showSaveIndicator(card, false);
  }
}

function updateKnowledgeProgress() {
  const reviewed = knowledgeData.filter(isKnowledgeReviewed).length;
  const pct = knowledgeData.length
    ? Math.round((reviewed / knowledgeData.length) * 100)
    : 0;
  document.getElementById("knowledge-progress").textContent =
    `${reviewed} / ${knowledgeData.length} (${pct}%)`;
  const bar = document.getElementById("knowledge-bar");
  if (bar) bar.style.width = pct + "%";
}

function renderKnowledge() {
  const list = document.getElementById("knowledge-list");
  list.innerHTML = "";
  const items = showUnreviewedOnly
    ? knowledgeData.filter((i) => !isKnowledgeReviewed(i))
    : knowledgeData;
  items.forEach((item) => {
    const reviewed = isKnowledgeReviewed(item);
    const card = document.createElement("div");
    card.className = "card" + (reviewed ? " reviewed" : "");
    card.innerHTML = `
      <div class="card-header">
        <div class="label">Knowledge</div>
        <div class="value">${escapeHtml(item.knowledge || "")}</div>
      </div>
      <div class="card-meta">
        <span class="meta-tag">Confidence: ${item.confidence_score != null && item.confidence_score !== "" ? Number(item.confidence_score).toFixed(2) : "—"}</span>
        <span class="meta-tag">Source: ${escapeHtml(item.source || "—")}</span>
        <span class="meta-tag">Domain: ${escapeHtml(item.best_future_domain || "—")}</span>
        <span class="meta-tag">Trend: ${escapeHtml(item.trend_label || "—")}</span>
        <span class="meta-tag">Weight: ${item.future_weight != null && item.future_weight !== "" ? Number(item.future_weight).toFixed(3) : "—"}</span>
      </div>
      ${(() => {
        let jh = "";
        if (item.job_text) {
          const t = item.job_text || "";
          if (t.length > 800) {
            jh = `<div class="job-context"><strong>Job context:</strong><blockquote><span class="job-preview">${escapeHtml(
              t.slice(0, 800),
            )}</span><span class="ellipsis">…</span><span class="job-full">${escapeHtml(t)}</span><button type="button" class="toggle-job-text" aria-expanded="false">Show more</button></blockquote></div>`;
          } else {
            jh = `<div class="job-context"><strong>Job context:</strong><blockquote>${escapeHtml(t)}</blockquote></div>`;
          }
        }
        return jh;
      })()}
      <div class="row">
        <div class="col">
          <div class="label">Valid?</div>
          <select data-field="human_valid">
            <option value="">--</option>
            <option value="valid" ${item.human_valid === "valid" ? "selected" : ""}>Valid</option>
            <option value="invalid" ${item.human_valid === "invalid" ? "selected" : ""}>Invalid</option>
          </select>
        </div>
      </div>
      <div class="label">Notes</div>
      <textarea data-field="human_notes" placeholder="Optional notes">${escapeHtml(item.human_notes || "")}</textarea>
      <button type="button" class="save-now">Save Knowledge Review</button>
      <span class="hint-inline">Changes auto-save. Click to save immediately.</span>
    `;
    const rid = item.review_id;
    card
      .querySelector(".save-now")
      ?.addEventListener("click", () => autoSaveKnowledge(rid, card));
    card.querySelectorAll("select").forEach((sel) => {
      sel.addEventListener("change", () =>
        debouncedSave(`know-${rid}`, () => autoSaveKnowledge(rid, card), 300),
      );
    });
    card.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () =>
        debouncedSave(`know-${rid}`, () => autoSaveKnowledge(rid, card)),
      );
    });
    list.appendChild(card);
  });
  updateKnowledgeProgress();
}

// --- Competencies ---
let competenciesData = [];

async function autoSaveCompetency(compId, card) {
  const hq =
    card.querySelector(`select[data-field="human_quality"]`)?.value || "";
  const hr =
    card.querySelector(`select[data-field="human_relevant"]`)?.value || "";
  const hsf =
    card.querySelector(`select[data-field="human_skill_focus"]`)?.value || "";
  const hn =
    card.querySelector(`textarea[data-field="human_notes"]`)?.value || "";
  try {
    await postJson(`${API}/save_competency_feedback`, {
      competency_id: compId,
      human_quality: hq,
      human_relevant: hr,
      human_skill_focus: hsf,
      human_notes: hn,
      reviewer_id: getReviewerId(),
    });
    const item = competenciesData.find((x) => x.competency_id === compId);
    if (item) {
      item.human_quality = hq;
      item.human_relevant = hr;
      item.human_skill_focus = hsf;
      item.human_notes = hn;
    }
    updateCompetencyProgress();
    card.classList.toggle("reviewed", !!(hq || hr || hsf || hn));
    showSaveIndicator(card, true);
    // Don't re-render on save when unreviewed-only: keep card visible for notes.
  } catch (e) {
    showSaveIndicator(card, false);
  }
}

function updateCompetencyProgress() {
  const reviewed = competenciesData.filter(isCompetencyReviewed).length;
  const pct = competenciesData.length
    ? Math.round((reviewed / competenciesData.length) * 100)
    : 0;
  document.getElementById("competencies-progress").textContent =
    `${reviewed} / ${competenciesData.length} (${pct}%)`;
  const bar = document.getElementById("competencies-bar");
  if (bar) bar.style.width = pct + "%";
}

function renderCompetencies() {
  const list = document.getElementById("competencies-list");
  list.innerHTML = "";
  const items = showUnreviewedOnly
    ? competenciesData.filter((i) => !isCompetencyReviewed(i))
    : competenciesData;
  items.forEach((item) => {
    const reviewed = isCompetencyReviewed(item);
    const card = document.createElement("div");
    card.className = "card" + (reviewed ? " reviewed" : "");
    const hv = item.all_skills_human_verified;
    const hasStatus = hv !== "" && hv != null && hv !== undefined;
    const isVerified = hasStatus && String(hv).toLowerCase() === "true";
    const verifiedTag = hasStatus
      ? isVerified
        ? '<span class="meta-tag verified">Skills verified</span>'
        : '<span class="meta-tag unverified">Skills not yet verified</span>'
      : "";
    card.innerHTML = `
      <div class="card-header">
        <div class="label">Title</div>
        <div class="value">${escapeHtml(item.title || "")}</div>
        ${verifiedTag ? `<div class="card-meta">${verifiedTag}</div>` : ""}
      </div>
      <div class="label">Description</div>
      <div class="value">${escapeHtml(item.description || "")}</div>
      <div class="card-meta">
        <span class="meta-tag">Competency ID: ${escapeHtml(item.competency_id || "—")}</span>
        <span class="meta-tag">Batch: ${escapeHtml(item.batch_id != null ? String(item.batch_id) : "—")}</span>
        ${item.kkni_level ? `<span class="meta-tag">KKNI ${escapeHtml(String(item.kkni_level))}</span>` : ""}
        <span class="meta-tag">Future relevance: ${escapeHtml((item.future_relevance || "").slice(0, 80))}${(item.future_relevance || "").length > 80 ? "…" : ""}</span>
      </div>
      <div class="label">Related skills (Bloom level per skill)</div>
      <div class="value small mono">${escapeHtml(item.related_skills_with_bloom || item.related_skills || "")}</div>
      ${item.soft_skills_required ? `<div class="label">Soft skills required</div>
      <div class="value small">${escapeHtml(item.soft_skills_required)}</div>` : ""}
      ${item.soft_skills_description ? `<div class="label">Soft skills (how they apply)</div>
      <div class="value small">${escapeHtml(item.soft_skills_description)}</div>` : ""}
      <div class="card-meta">
        ${item.skill_focus || "" ? `<span class="meta-tag">Skill focus (derived): ${escapeHtml(item.skill_focus)}</span>` : ""}
      </div>
      <div class="row">
        <div class="col">
          <div class="label">Skill focus</div>
          <select data-field="human_skill_focus">
            <option value="">--</option>
            <option value="Hard" ${item.human_skill_focus === "Hard" ? "selected" : ""}>Hard</option>
            <option value="Soft" ${item.human_skill_focus === "Soft" ? "selected" : ""}>Soft</option>
            <option value="Both" ${item.human_skill_focus === "Both" ? "selected" : ""}>Both</option>
          </select>
        </div>
        <div class="col">
          <div class="label">Quality (1–5)</div>
          <select data-field="human_quality">
            <option value="">--</option>
            <option value="1" ${item.human_quality === "1" || item.human_quality === "poor" ? "selected" : ""}>1 – Poor</option>
            <option value="2" ${item.human_quality === "2" || item.human_quality === "fair" ? "selected" : ""}>2 – Fair</option>
            <option value="3" ${item.human_quality === "3" || item.human_quality === "good" ? "selected" : ""}>3 – Good</option>
            <option value="4" ${item.human_quality === "4" ? "selected" : ""}>4 – Strong</option>
            <option value="5" ${item.human_quality === "5" || item.human_quality === "excellent" ? "selected" : ""}>5 – Excellent</option>
          </select>
        </div>
      </div>
      <div class="row">
        <div class="col">
          <div class="label">Relevant?</div>
          <select data-field="human_relevant">
            <option value="">--</option>
            <option value="yes" ${item.human_relevant === "yes" ? "selected" : ""}>Yes</option>
            <option value="partial" ${item.human_relevant === "partial" ? "selected" : ""}>Partial</option>
            <option value="no" ${item.human_relevant === "no" ? "selected" : ""}>No</option>
          </select>
        </div>
      </div>
      <div class="label">Notes</div>
      <textarea data-field="human_notes" placeholder="Optional notes">${escapeHtml(item.human_notes || "")}</textarea>
      <button type="button" class="save-now">Save Competency Review</button>
      <span class="hint-inline">Changes auto-save. Click to save immediately.</span>
    `;
    const cid = item.competency_id;
    card
      .querySelector(".save-now")
      ?.addEventListener("click", () => autoSaveCompetency(cid, card));
    card.querySelectorAll("select").forEach((sel) => {
      sel.addEventListener("change", () =>
        debouncedSave(`comp-${cid}`, () => autoSaveCompetency(cid, card), 300),
      );
    });
    card.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () =>
        debouncedSave(`comp-${cid}`, () => autoSaveCompetency(cid, card)),
      );
    });
    list.appendChild(card);
  });
  updateCompetencyProgress();
}

// --- Filter toggle ---
function toggleFilter() {
  showUnreviewedOnly = !showUnreviewedOnly;
  document.querySelectorAll(".btn-filter").forEach((b) => {
    b.textContent = showUnreviewedOnly ? "Show All" : "Show Unreviewed Only";
    b.classList.toggle("active-filter", showUnreviewedOnly);
  });
  renderSkills();
  renderKnowledge();
  renderCompetencies();
}

// --- Jump to next unreviewed ---
function jumpToNextUnreviewed(panelId, data, checkFn) {
  const panel = document.getElementById(panelId);
  const cards = panel.querySelectorAll(".card:not(.reviewed)");
  if (cards.length > 0) {
    cards[0].scrollIntoView({ behavior: "smooth", block: "center" });
    cards[0].classList.add("highlight");
    setTimeout(() => cards[0].classList.remove("highlight"), 1500);
  }
}

document.querySelectorAll(".btn-next").forEach((btn) => {
  btn.addEventListener("click", () => {
    const panel = btn.closest(".panel");
    jumpToNextUnreviewed(panel.id);
  });
});

// --- Load and init ---
async function load() {
  try {
    const [skillsRes, knowledgeRes, compRes] = await Promise.all([
      fetchJson(apiUrl("/skills")),
      fetchJson(apiUrl("/knowledge")),
      fetchJson(apiUrl("/competencies")),
    ]);
    skillsData = skillsRes.items || [];
    knowledgeData = knowledgeRes.items || [];
    competenciesData = compRes.items || [];
    const rid =
      skillsRes.reviewer_id ||
      knowledgeRes.reviewer_id ||
      compRes.reviewer_id ||
      getReviewerId();
    const el = document.getElementById("reviewer-display");
    if (el) el.textContent = rid;
    if (skillsData.length === 0 && skillsRes.hint) {
      document.getElementById("skills-list").innerHTML =
        `<p style="color:#b91c1c">${escapeHtml(skillsRes.hint)}. Run <code>python export_for_review.py</code> from project root first.</p>`;
    }
    renderSkills();
    renderKnowledge();
    renderCompetencies();
  } catch (e) {
    document.getElementById("skills-list").innerHTML =
      `<p style="color:#b91c1c">Failed to load: ${e.message}. Run export scripts first.</p>`;
  }
}

load();
