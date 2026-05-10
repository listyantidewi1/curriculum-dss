const API = "/api";
let showUnlabeledOnly = false;
let labelerId = "";
let domainOptions = [];

function getLabelerId() {
  if (labelerId) return labelerId;
  const params = new URLSearchParams(window.location.search);
  labelerId = params.get("labeler_id") || params.get("labeler") || "default";
  return labelerId;
}

function apiUrl(path) {
  const lid = getLabelerId();
  const sep = path.includes("?") ? "&" : "?";
  return `${API}${path}${sep}labeler_id=${encodeURIComponent(lid)}`;
}

function showPanel(name) {
  document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
  document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
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

function isSkillLabeled(item) {
  return !!(item.is_correct && (item.is_correct === "yes" || item.is_correct === "no"));
}
function isKnowledgeLabeled(item) {
  return !!(item.is_correct && (item.is_correct === "yes" || item.is_correct === "no"));
}
function isDomainLabeled(item) {
  return !!(item.true_domain_id && String(item.true_domain_id).trim());
}

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
  setTimeout(() => { dot.textContent = ""; }, 2000);
}

const OVERLAP_N = 20;

// --- Skills ---
let skillsData = [];

async function autoSaveSkill(goldId, card) {
  const ic = card.querySelector(`select[data-field="is_correct"]`)?.value || "";
  const tl = card.querySelector(`select[data-field="type_label"]`)?.value || "";
  const nn = card.querySelector(`textarea[data-field="notes"]`)?.value || "";
  try {
    await postJson(`${API}/save_skill_label`, {
      gold_id: goldId, is_correct: ic, type_label: tl, notes: nn,
      labeler_id: getLabelerId(),
    });
    const item = skillsData.find((x) => x.gold_id === goldId);
    if (item) { item.is_correct = ic; item.type_label = tl; item.notes = nn; }
    updateSkillProgress();
    card.classList.toggle("reviewed", !!(ic === "yes" || ic === "no"));
    showSaveIndicator(card, true);
  } catch (e) {
    showSaveIndicator(card, false);
  }
}

function updateSkillProgress() {
  const labeled = skillsData.filter(isSkillLabeled).length;
  const pct = skillsData.length ? Math.round(labeled / skillsData.length * 100) : 0;
  document.getElementById("skills-progress").textContent = `${labeled} / ${skillsData.length} (${pct}%)`;
  const bar = document.getElementById("skills-bar");
  if (bar) bar.style.width = pct + "%";
}

function renderSkills() {
  const list = document.getElementById("skills-list");
  list.innerHTML = "";
  const items = showUnlabeledOnly ? skillsData.filter((i) => !isSkillLabeled(i)) : skillsData;
  items.forEach((item, idx) => {
    const labeled = isSkillLabeled(item);
    const isOverlap = idx < OVERLAP_N;
    const card = document.createElement("div");
    card.className = "card" + (labeled ? " reviewed" : "") + (isOverlap ? " overlap" : "");
    card.innerHTML = `
      ${isOverlap ? '<span class="overlap-badge">Overlap (IRR)</span>' : ''}
      <div class="card-header">
        <div class="label">Skill</div>
        <div class="value">${escapeHtml(item.skill || "")}</div>
      </div>
      <div class="card-meta">
        <span class="meta-tag">Type: ${escapeHtml(item.type || "—")}</span>
        <span class="meta-tag">Confidence: ${item.confidence_score || "—"}</span>
        <span class="meta-tag">Source: ${escapeHtml(item.source || "—")}</span>
      </div>
      <div class="row">
        <div class="col">
          <div class="label">is_correct <span class="required">*</span></div>
          <select data-field="is_correct">
            <option value="">--</option>
            <option value="yes" ${item.is_correct === "yes" ? "selected" : ""}>yes</option>
            <option value="no" ${item.is_correct === "no" ? "selected" : ""}>no</option>
          </select>
        </div>
        <div class="col">
          <div class="label">type_label</div>
          <select data-field="type_label">
            <option value="">--</option>
            <option value="Hard" ${item.type_label === "Hard" ? "selected" : ""}>Hard</option>
            <option value="Soft" ${item.type_label === "Soft" ? "selected" : ""}>Soft</option>
            <option value="Both" ${item.type_label === "Both" ? "selected" : ""}>Both</option>
          </select>
        </div>
      </div>
      <div class="label">Notes</div>
      <textarea data-field="notes" placeholder="Optional">${escapeHtml(item.notes || "")}</textarea>
    `;
    const gid = item.gold_id;
    card.querySelectorAll("select").forEach((sel) => {
      sel.addEventListener("change", () => debouncedSave(`skill-${gid}`, () => autoSaveSkill(gid, card), 300));
    });
    card.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () => debouncedSave(`skill-${gid}`, () => autoSaveSkill(gid, card)));
    });
    list.appendChild(card);
  });
  updateSkillProgress();
}

// --- Knowledge ---
let knowledgeData = [];

async function autoSaveKnowledge(goldId, card) {
  const ic = card.querySelector(`select[data-field="is_correct"]`)?.value || "";
  const nn = card.querySelector(`textarea[data-field="notes"]`)?.value || "";
  try {
    await postJson(`${API}/save_knowledge_label`, {
      gold_id: goldId, is_correct: ic, notes: nn,
      labeler_id: getLabelerId(),
    });
    const item = knowledgeData.find((x) => x.gold_id === goldId);
    if (item) { item.is_correct = ic; item.notes = nn; }
    updateKnowledgeProgress();
    card.classList.toggle("reviewed", !!(ic === "yes" || ic === "no"));
    showSaveIndicator(card, true);
  } catch (e) {
    showSaveIndicator(card, false);
  }
}

function updateKnowledgeProgress() {
  const labeled = knowledgeData.filter(isKnowledgeLabeled).length;
  const pct = knowledgeData.length ? Math.round(labeled / knowledgeData.length * 100) : 0;
  document.getElementById("knowledge-progress").textContent = `${labeled} / ${knowledgeData.length} (${pct}%)`;
  const bar = document.getElementById("knowledge-bar");
  if (bar) bar.style.width = pct + "%";
}

function renderKnowledge() {
  const list = document.getElementById("knowledge-list");
  list.innerHTML = "";
  const items = showUnlabeledOnly ? knowledgeData.filter((i) => !isKnowledgeLabeled(i)) : knowledgeData;
  items.forEach((item, idx) => {
    const labeled = isKnowledgeLabeled(item);
    const isOverlap = idx < OVERLAP_N;
    const card = document.createElement("div");
    card.className = "card" + (labeled ? " reviewed" : "") + (isOverlap ? " overlap" : "");
    card.innerHTML = `
      ${isOverlap ? '<span class="overlap-badge">Overlap (IRR)</span>' : ''}
      <div class="card-header">
        <div class="label">Knowledge</div>
        <div class="value">${escapeHtml(item.knowledge || "")}</div>
      </div>
      <div class="card-meta">
        <span class="meta-tag">Confidence: ${item.confidence_score || "—"}</span>
        <span class="meta-tag">Source: ${escapeHtml(item.source || "—")}</span>
      </div>
      <div class="row">
        <div class="col">
          <div class="label">is_correct <span class="required">*</span></div>
          <select data-field="is_correct">
            <option value="">--</option>
            <option value="yes" ${item.is_correct === "yes" ? "selected" : ""}>yes</option>
            <option value="no" ${item.is_correct === "no" ? "selected" : ""}>no</option>
          </select>
        </div>
      </div>
      <div class="label">Notes</div>
      <textarea data-field="notes" placeholder="Optional">${escapeHtml(item.notes || "")}</textarea>
    `;
    const gid = item.gold_id;
    card.querySelectorAll("select").forEach((sel) => {
      sel.addEventListener("change", () => debouncedSave(`know-${gid}`, () => autoSaveKnowledge(gid, card), 300));
    });
    card.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () => debouncedSave(`know-${gid}`, () => autoSaveKnowledge(gid, card)));
    });
    list.appendChild(card);
  });
  updateKnowledgeProgress();
}

// --- Domain ---
let domainData = [];

async function autoSaveDomain(goldId, card) {
  const td = card.querySelector(`select[data-field="true_domain_id"]`)?.value || "";
  const nn = card.querySelector(`textarea[data-field="notes"]`)?.value || "";
  try {
    await postJson(`${API}/save_domain_label`, {
      gold_id: goldId, true_domain_id: td, notes: nn,
      labeler_id: getLabelerId(),
    });
    const item = domainData.find((x) => x.gold_id === goldId);
    if (item) { item.true_domain_id = td; item.notes = nn; }
    updateDomainProgress();
    card.classList.toggle("reviewed", !!td);
    showSaveIndicator(card, true);
  } catch (e) {
    showSaveIndicator(card, false);
  }
}

function updateDomainProgress() {
  const labeled = domainData.filter(isDomainLabeled).length;
  const pct = domainData.length ? Math.round(labeled / domainData.length * 100) : 0;
  document.getElementById("domain-progress").textContent = `${labeled} / ${domainData.length} (${pct}%)`;
  const bar = document.getElementById("domain-bar");
  if (bar) bar.style.width = pct + "%";
}

function domainSelectOptions(selected) {
  const knownIds = new Set(domainOptions.map(d => d.id));
  const opts = [...domainOptions];
  if (selected && !knownIds.has(selected)) {
    opts.push({ id: selected, name: selected });
  }
  return opts.map(d => {
    const label = d.id === d.name || !d.name ? d.id : `${d.id} — ${d.name}`;
    return `<option value="${escapeHtml(d.id)}">${escapeHtml(label)}</option>`;
  }).join("");
}

function renderDomain() {
  const list = document.getElementById("domain-list");
  list.innerHTML = "";
  const items = showUnlabeledOnly ? domainData.filter((i) => !isDomainLabeled(i)) : domainData;
  items.forEach((item, idx) => {
    const labeled = isDomainLabeled(item);
    const isOverlap = idx < OVERLAP_N;
    const selected = String(item.true_domain_id || "").trim();
    const opts = domainSelectOptions(selected);
    const card = document.createElement("div");
    card.className = "card" + (labeled ? " reviewed" : "") + (isOverlap ? " overlap" : "");
    card.innerHTML = `
      ${isOverlap ? '<span class="overlap-badge">Overlap (IRR)</span>' : ''}
      <div class="card-header">
        <div class="label">Item</div>
        <div class="value">${escapeHtml(item.item_text || "")}</div>
      </div>
      <div class="card-meta">
        <span class="meta-tag">Pipeline domain: ${escapeHtml(item.pipeline_domain || "—")}</span>
        <span class="meta-tag">Type: ${escapeHtml(item.item_type || "—")}</span>
      </div>
      <div class="row">
        <div class="col">
          <div class="label">true_domain_id <span class="required">*</span></div>
          <select data-field="true_domain_id">
            <option value="">--</option>
            ${opts}
          </select>
        </div>
      </div>
      <div class="label">Notes</div>
      <textarea data-field="notes" placeholder="Optional">${escapeHtml(item.notes || "")}</textarea>
    `;
    const sel = card.querySelector(`select[data-field="true_domain_id"]`);
    if (sel && selected) sel.value = selected;
    const gid = item.gold_id;
    card.querySelectorAll("select").forEach((s) => {
      s.addEventListener("change", () => debouncedSave(`dom-${gid}`, () => autoSaveDomain(gid, card), 300));
    });
    card.querySelectorAll("textarea").forEach((ta) => {
      ta.addEventListener("input", () => debouncedSave(`dom-${gid}`, () => autoSaveDomain(gid, card)));
    });
    list.appendChild(card);
  });
  updateDomainProgress();
}

// --- Filter ---
function toggleFilter() {
  showUnlabeledOnly = !showUnlabeledOnly;
  document.querySelectorAll(".btn-filter").forEach((b) => {
    b.textContent = showUnlabeledOnly ? "Show All" : "Show Unlabeled Only";
    b.classList.toggle("active-filter", showUnlabeledOnly);
  });
  renderSkills();
  renderKnowledge();
  renderDomain();
}

function jumpToNextUnlabeled(panelId) {
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
    jumpToNextUnlabeled(panel.id);
  });
});

// --- Load ---
async function load() {
  try {
    const domainsRes = await fetchJson(`${API}/domains`);
    const raw = domainsRes.domains || domainsRes.fallback || [];
    domainOptions = raw.map(d => (typeof d === "string") ? { id: d, name: d } : d);

    const [skillsRes, knowledgeRes, domainRes] = await Promise.all([
      fetchJson(apiUrl("/skills")),
      fetchJson(apiUrl("/knowledge")),
      fetchJson(apiUrl("/domain")),
    ]);

    skillsData = skillsRes.items || [];
    knowledgeData = knowledgeRes.items || [];
    domainData = domainRes.items || [];

    const lid = skillsRes.labeler_id || knowledgeRes.labeler_id || domainRes.labeler_id || getLabelerId();
    const el = document.getElementById("labeler-display");
    if (el) el.textContent = lid;

    if (skillsData.length === 0 && skillsRes.hint) {
      document.getElementById("skills-list").innerHTML =
        `<p style="color:#b91c1c">${escapeHtml(skillsRes.hint)}</p>`;
    }

    renderSkills();
    renderKnowledge();
    renderDomain();
  } catch (e) {
    document.getElementById("skills-list").innerHTML =
      `<p style="color:#b91c1c">Failed to load: ${escapeHtml(e.message)}</p>`;
  }
}

load();
