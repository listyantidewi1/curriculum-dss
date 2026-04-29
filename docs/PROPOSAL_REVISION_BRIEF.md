# Comprehensive Pipeline Summary — for Proposal Revision

> **Purpose**: hand this document to another Claude session along with the
> dissertation proposal so the proposal can be updated to reflect the current
> system. Each section calls out which proposal sections need attention and
> what specifically needs to change.

---

## 1. Identitas dan Pivot Sistem

**Nama produk (terkini):** Sistem Rekomendasi Kompetensi (Competency Recommendation
System) berbasis KKNI, ruang lingkup terbatas pada Software Engineering & Game
Development.

**Pergeseran scope dari proposal asli:**

- ~~"Sistem rekomendasi kurikulum SMK konsentrasi RPL"~~ → **"Sistem rekomendasi
  kompetensi untuk seluruh jenjang pendidikan (SMA, SMK, D1–D3, S1, Profesi, S2,
  S3) di bidang Software/Game Development"**.
- Audiens berkembang dari guru/koordinator kurikulum SMK menjadi **publik luas**
  (siswa, guru, dosen, pengembang kurikulum, perusahaan, peneliti).
- Sekolah **tidak lagi** mengunggah data lowongan kerja sendiri. Seluruh akuisisi
  data dilakukan oleh **admin tunggal (peneliti)**, hasil dipublikasikan satu
  kali ke public surface.
- Sekolah hanya mengunggah **kurikulumnya sendiri** untuk analisis cakupan
  (coverage analysis), opsional dengan akun ringan untuk menyimpan riwayat.
- Domain **dikunci** pada Bidang Keahlian 4 (Teknologi Informasi) per Kepmen
  244/M/2024. Konsentrasi RPL dan PG diperlakukan sebagai dua spektrum terpisah.

**Implikasi proposal:**

- **Judul** mungkin perlu diperluas: "...untuk SMK dan Pendidikan Tinggi Bidang
  Software & Game Development" atau diganti framing menjadi "Pengembangan
  Sistem Rekomendasi Kompetensi Berbasis KKNI...".
- **BAB I §1.1 (Latar Belakang)** masih relevan untuk SMK; perlu paragraf
  tambahan yang menjelaskan mengapa sistem dibuka untuk seluruh jenjang KKNI
  1–9, bukan hanya KKNI 2–3 (SMK).
- **BAB I §1.4 (Manfaat)** poin 2 perlu mencerminkan audiens publik, bukan hanya
  guru SMK.

---

## 2. Audiens & Surface

| Audiens | Akses | Auth |
|---------|-------|------|
| **Publik** (siapa pun) | `/`, `/competencies`, `/competencies/{id}`, `/coverage`, `/about` | Tidak perlu login |
| **Pemegang akun ringan** | `/me/coverage`, `/coverage` (dengan simpan) | Email + password (`role='public'`) |
| **Admin** (peneliti) | `/dashboard/admin/*` termasuk **`/dashboard/admin/publish`** | Session login (`role='admin'`) |
| ~~School user~~ | Legacy `/dashboard/school/*` masih ada untuk akun lama, **tidak ada signup baru** | — |

**Implikasi proposal:**

- **BAB III §3.4 (Populasi dan Sampel)** perlu menambahkan kategori responden
  publik (siswa, guru, dosen, peneliti) di luar guru SMK.
- **BAB III §3.3 (Deskripsi Produk)** mockup dashboard yang ada di Gambar 3.5
  menggambarkan tampilan school-internal (ranking + curriculum gap per
  komponen). Mockup ini sekarang **lebih sesuai untuk admin internal**; mockup
  public-facing baru perlu dideskripsikan: hero page, stage chips
  (SMA / SMK / D1–D3 / S1 / Profesi / S2 / S3), browse + filter, detail page,
  coverage analyzer.

---

## 3. KKNI Integration (Fitur Baru — Belum Ada di Proposal)

**Perubahan paling besar yang belum tercakup di proposal sama sekali.** Sistem
sekarang:

1. **Mengintegrasikan Perpres 8/2012 KKNI** sebagai sumber kebenaran tunggal
   untuk level kualifikasi 1–9.
2. Setiap **competency carries `kkni_level`** (1–9) yang dideterminasi oleh:
   - **Floor deterministik** dari Bloom level tertinggi di `related_skills`:
     - Remember → KKNI 1, Understand → 2, Apply → 4, Analyze → 5,
       Evaluate → 6, Create → 7
   - **LLM diberi reference table KKNI 1–9** dan diminta memilih level dalam
     band `[floor, floor+1]`. Out-of-band values dijepit (clamped) kembali ke
     floor oleh post-validator.
3. **Mapping stage-friendly ke KKNI levels** untuk filter UI:
   - SMA → [2], SMK → [2,3], D1 → [3], D2 → [4], D3 → [5], D4/S1 → [6],
     Profesi → [7], S2 → [7,8], S3 → [9]
4. **Modul `kkni.py`** menyimpan deskriptor verbatim Perpres 8/2012
   (`KKNI_LEVELS[1..9]`) + helper `kkni_floor_for_competency()`,
   `clamp_llm_kkni()`, `kkni_reference_for_prompt()`.

**Implikasi proposal:**

- **BAB II perlu seksi baru §2.X "Kerangka Kualifikasi Nasional Indonesia
  (KKNI)"** menjelaskan Perpres 8/2012, struktur 9 jenjang, kelompok jabatan
  (Operator/Teknisi-Analis/Ahli), dan justifikasi pemilihan KKNI sebagai tulang
  punggung pemetaan jenjang kompetensi.
- **BAB II §2.4 (Desain Kurikulum)** sebaiknya menambahkan Tabel pemetaan
  Bloom → KKNI sebagai jembatan teoretis antara CBET dan KKNI.
- **BAB III §3.2.3 (Develop Product)** perlu menambahkan langkah "Hybrid KKNI
  assignment: deterministic floor + LLM refinement within clamped band" sebagai
  komponen produk.
- **Tabel 3.1 (Pemetaan Metrik Evaluasi)** perlu metrik baru: KKNI assignment
  accuracy oleh expert (target ≥ 0.70 Top-1 setelah clamp).

---

## 4. Education-Level Extraction (Fitur Baru — Belum Ada di Proposal)

Sistem sekarang **memparse persyaratan pendidikan dari teks lowongan kerja** dan
memetakannya ke level KKNI:

- **Modul `education_level_extractor.py`** dengan regex English-first
  (Indonesian-ready):
  - "Bachelor's", "S1", "Sarjana" → KKNI 6
  - "D3", "Diploma 3", "Associate" → KKNI 5
  - "SMK", "vocational" → KKNI 3
  - "Master's", "S2", "Magister" → KKNI 8
  - "PhD", "S3", "Doktor" → KKNI 9
- **Output:** `jobs_metadata.csv` mendapat kolom `min_education_kkni` dan
  `education_labels`.
- **Modul `compute_education_demand.py`** mengagregasi **per-skill demand by
  KKNI level**: berapa banyak lowongan pada KKNI 3 yang menuntut skill X,
  berapa pada KKNI 6, dst.
- Output: `skill_education_demand.csv` (long format) +
  `skill_education_summary.csv` (per-skill: dominant_kkni + distribusi JSON).
- Public UI menampilkan **education-demand chart per kompetensi** di halaman
  detail.

**Implikasi proposal:**

- **BAB II perlu paragraf** di §2.7 atau §2.8 yang menjelaskan ekstraksi syarat
  pendidikan dari teks lowongan sebagai tahap NLP tambahan.
- **BAB III §3.2.3** menambahkan langkah ekstraksi level pendidikan ke dalam
  pipeline.
- Catatan keterbatasan: korpus saat ini bahasa Inggris, regex Indonesia siap
  pakai tapi belum diuji dengan data Indonesia.

---

## 5. Pipeline Saat Ini — Struktur Lengkap

### Phase 1: `run.bat` (18 langkah, resume-aware via `--resume`)

| # | Langkah | Output kunci |
|---|---------|--------------|
| 1 | `run_with_job_scraping.py --dedupe` (default mode = **llm_only**) | `advanced_skills.csv`, `advanced_knowledge.csv`, `run_metadata.json` (dengan prompt SHA-256 hashes); `jobs_metadata.csv` sudah berisi `min_education_kkni` dari preprocess |
| 2 | `run_with_job_scraping.py --extraction-mode hybrid --output_dir results/hybrid` (RQ1 ablation) | `results/hybrid/advanced_skills.csv` |
| 3 | `plot_generator.py` | figures/ |
| 4 | `verify_skills.py` | `verified_skills.csv` |
| 5 | `future_weight_mapping.py` (knowledge) | `future_skill_weights_dummy.csv` |
| 6 | `future_weight_mapping.py --input_type skills` | `future_skill_weights.csv` (with margin) |
| 7 | `enrich_with_dates.py` | `advanced_skills_with_dates.csv` |
| 8 | `skill_time_trend_analysis.py --only_hard --stability` | `skill_time_trends.csv` (FDR q-values + Durbin-Watson) |
| **8b** | **`compute_education_demand.py`** (BARU) | `skill_education_demand.csv`, `skill_education_summary.csv` |
| 9 | `generate_competencies.py` (hard-skills only, KKNI-aware, ≤12 per batch) | `competency_proposals.json` |
| 10 | `recommendations.py --ablation --sensitivity --coverage-ablation` | `recommendations.csv`, `coverage_ablation_report.json` |
| 11–14 | export gold set, export for review, export competencies for review, export recommendations for review (RQ5 IRR) | `DATA/labels/*.csv` |
| — | **Gold Labeling UI** (interaktif) | `DATA/labels/gold_labels/*.csv` |
| 15 | `merge_gold_labels.py` | `DATA/labels/gold_*_merged.csv` |
| 16 | `evaluate_extraction.py --llmonly-labels-dir results/hybrid/DATA/labels` | `extraction_evaluation_report.json` (RQ1: LLM-only primary vs hybrid ablation) |
| 17 | `evaluate_future_mapping.py` | `future_mapping_evaluation_report.json` |
| 18 | `plot_scientific_analysis.py` | scientific figures |
| — | **Expert Review UI** (interaktif) | `feedback_store/*.csv` |

### Phase 2: `run_phase_2.bat` (18 langkah, resume-aware)

Pasca-review: import_feedback → apply_feedback → validate_parameters (5-fold CV,
calibrated threshold) → re-verify_skills → merge_gold_labels → re-future_weight
(with tier sensitivity) → generate_competencies --comprehensive → re-trend →
**9b: re-compute education demand** → recommendations (with --baseline) →
re-plots → evaluate_extraction → evaluate_future_mapping →
evaluate_competency_generation → scientific plots → log_run_metadata →
weight_sensitivity_extraction → **17b: skill_trend_holdout_validation** (RQ3
holdout) → export_recommendations_for_review (RQ5 IRR refresh).

**Implikasi proposal:**

- **BAB III §3.3 (Deskripsi Produk)** masih menyebut "Fase 1 18 langkah, Fase 2
  4 langkah". **Ini sudah usang.** Phase 2 sekarang **18 langkah** termasuk
  `compute_education_demand` (9b), `weight_sensitivity_extraction` (17), dan
  `skill_trend_holdout_validation` (17b).
- **Gambar 3.4 (Perhitungan Skala Prioritas)** masih akurat
  (priority_score = 0.40 × demand + 0.30 × trend + 0.30 × future,
  coverage = 0.0). Tapi sekarang ada **coverage ablation** yang membuktikan
  secara empiris bahwa w_coverage = 0.0 lebih baik (Jaccard ≥ 0.80 dibanding
  baseline).
- **Gambar 3.2 (Pipeline Direction A)** **harus diganti**: default sekarang
  LLM-only, hybrid hanya path ablation. Gambar baru: dua jalur paralel
  (Primary: LLM-only; Ablation: Hybrid) → fusi hanya di mode hybrid.

---

## 6. Perubahan pada Ekstraksi (BAB II §2.7–§2.10 dan BAB III §3.2.3)

**Default mode berubah dari `hybrid` → `llm_only`.** BERT (JobBERT) tetap
dipertahankan **sebagai jalur ablation** (`--extraction-mode hybrid`), bukan
path utama lagi.

| Aspek | Status Lama (Proposal) | Status Sekarang |
|-------|------------------------|-----------------|
| **Default extraction** | Hybrid Direction A (BERT+LLM fusion) | **LLM-only** pada full document |
| **Skills** | BERT+LLM fusion via SBERT cosine | **LLM-only** (BERT path tetap ada untuk `--extraction-mode hybrid`) |
| **Knowledge** | LLM-only (proposal already correct) | LLM-only (unchanged) |
| **BERT knowledge → LLM** | Anti-hallucination context | Hanya aktif di mode hybrid |
| **Type (Hard/Soft)** | LLM | LLM (unchanged) |
| **Bloom** | Two-stage (SBERT exemplars + LLM fallback) | Same |
| **Domain (future)** | Post-extraction SBERT (`future_weight_mapping.py`) | Same — **tidak** dipindah ke LLM |
| **KKNI Level (BARU)** | — | Hybrid: deterministic Bloom-floor + LLM refinement, clamped to [floor, floor+1] |

**Implikasi proposal:**

- **§2.7, §2.9, §2.10** masih relevan teoretis tapi posisi BERT digeser dari
  "tulang punggung" menjadi "jalur ablation untuk RQ1".
- **§2.10 paragraf "Direction A"** perlu rewrite: arsitektur primer sekarang
  adalah LLM-only on full text dengan reasoning langsung; hybrid menjadi
  ablation untuk membuktikan apakah BERT memberikan lift.
- **RQ1 di BAB I** perlu reframing: bukan lagi "Apakah hybrid lebih baik dari
  komponen?" melainkan **"Apakah LLM-only sudah cukup untuk competency
  recommendation, dan apakah hybrid (+BERT) memberikan lift presisi yang
  material?"**

---

## 7. Perubahan pada Generasi Kompetensi (Major — banyak yang baru)

| Aspek | Status Lama (Proposal) | Status Sekarang |
|-------|------------------------|-----------------|
| **Input** | Semua skill (hard + soft) | **Hanya hard skills** (filter type ∈ {Hard, Both}); soft skills ditangani terpisah |
| **Soft skills** | Bercampur di `related_skills` | **Field terpisah: `soft_skills_required` (list 3–6)** + `soft_skills_description` (1 kalimat). Hybrid: top-N extracted soft skills as prompt context, LLM may expand |
| **Jumlah per batch** | "Aim for 8–20" (no enforcement) | **Hard cap 12 per batch**; LLM diminta coalesce overlapping themes |
| **Grouping** | Domain-based batching | Domain-based **+ sub-clustering of "Uncertain"/"Unmapped" via SBERT** (mengatasi keluhan: skill yang tidak terkait dilumpukkan jadi satu) |
| **Deduplication** | Title normalization + Jaccard merge | **Three-stage:** (1) normalized-title exact, (2) **SBERT semantic title similarity ≥ 0.85 AND skill Jaccard ≥ 0.40**, (3) optional related-skill Jaccard merge |
| **KKNI level** | — | **Required schema field**, hybrid floor + LLM refinement |
| **Skema final** | id, title, description, related_skills, future_relevance | + **soft_skills_required, soft_skills_description, kkni_level, kkni_floor, kkni_descriptor** |

**Implikasi proposal:**

- **BAB III §3.2.3 poin 4 (Modul generasi kompetensi)** perlu rewrite
  mencerminkan: hard-only input, soft-skills sebagai field terpisah, KKNI
  assignment, hard cap 12, sub-clustering Uncertain.
- Tambahkan **paragraf di §2.4** menjelaskan rasional konsultasi mentor:
  "kompetensi sebelumnya terlalu banyak dan kelompok skill kadang tidak terkait
  — diatasi dengan hard cap, sub-clustering, dan dedup semantic-title."

---

## 8. Coverage Analysis (Public-Facing, Belum Ada di Proposal)

Sistem sekarang menyediakan **endpoint upload kurikulum publik**:

- **`POST /coverage`** (anonymous) atau dengan akun ringan untuk simpan analisis.
- Parse CSV dengan kolom (`phrase`, `objective`, `content`, `topic`, `name`,
  `title`, `description`) atau JSON tree walking.
- **Output report:**
  - `coverage_pct` (% kompetensi yang sudah covered di kurikulum sekolah)
  - `n_competencies_covered` / `n_competencies_uncovered`
  - `covered_competencies` (daftar)
  - `uncovered_competencies` (daftar — gap analysis langsung untuk Tyler/Taba step 1)
  - `missing_high_priority_skills` (top-20 priority skills yang belum ada di
    kurikulum, dengan future_domain dan priority_score)
- Saved analyses tersimpan di tabel `coverage_analyses` untuk akun yang login.

**Implikasi proposal:**

- **BAB I §1.4 (Manfaat poin 2)** sekarang lebih kuat: produk "siap digunakan"
  tidak hanya oleh guru SMK tapi juga public viewer karena ada self-service
  coverage.
- **§2.4 Model Taba — diagnosis kebutuhan (langkah 1)** dapat ditegaskan sebagai
  **dioperasionalisasi langsung** lewat endpoint `/coverage`, bukan hanya via
  dashboard sekolah.
- **BAB III §3.2.10 (Diseminasi)** dapat menyoroti bahwa sistem dapat
  dimanfaatkan publik luas tanpa pelatihan.

---

## 9. Publish Workflow (Belum Ada di Proposal)

Pipeline run dipisahkan dari public consumption:

1. Admin (peneliti) menjalankan `run.bat` → `results/`.
2. Admin login ke `/dashboard/admin/publish`, mengisi version label,
   vocational_field, spektrum_code, notes.
3. Tabel `published_runs(is_active=1)` dimutakhirkan; cache public surface
   di-invalidasi.
4. Public site auto-pick latest published run; jika belum ada publish, fallback
   ke project-level `results/`.

Memberi admin kontrol penuh kapan rekomendasi go-live, mencegah flicker selama
re-runs.

**Implikasi proposal:**

- **BAB III §3.2.10 (Diseminasi)** dapat ditambahkan paragraf: "Mekanisme
  publish-gate memungkinkan admin mengontrol versi rekomendasi yang ditampilkan
  ke publik, mendukung tahap diseminasi bertahap dan A/B testing untuk evaluasi
  efektivitas."

---

## 10. Improvement Lain yang Sudah Ada di Proposal Tapi Sudah Diimplementasikan

Hal-hal di proposal yang **sudah benar dan terimplementasi** (tidak perlu
diubah):

| Proposal Section | Sudah ada? |
|------------------|------------|
| Skill normalization (SBERT clustering) | ✓ `skill_normalizer.py` (threshold 0.82) |
| Job deduplication (MD5 fingerprint) | ✓ `--dedupe` flag |
| Indonesian postings support | ✓ `--include-indonesian` (regex Indonesia siap, data belum) |
| FDR Benjamini-Hochberg | ✓ `skill_time_trend_analysis.py` |
| Durbin-Watson autocorrelation | ✓ |
| Wilson Score CI | ✓ `evaluate_extraction.py` |
| AUC-ROC + Brier Score (5-fold CV) | ✓ `validate_parameters.py` |
| Cohen's / Fleiss' Kappa | ✓ `import_feedback.py` |
| MRR + NDCG@20 | ✓ `evaluate_future_mapping.py`, `recommendations.py --evaluate` |
| Jaccard stability | ✓ trend_stability_report.json |
| Weight sensitivity analysis | ✓ |
| Trend holdout validation (RQ3) | ✓ `skill_trend_holdout_validation.py` |
| Coverage ablation report | ✓ |
| LLM prompt versioning (SHA-256) | ✓ `log_run_metadata.py` |
| Checkpoint/resume in pipeline | ✓ `run.bat --resume`, `run_phase_2.bat --resume`, `pipeline_orchestrator.py resume=True` |
| Printable PDF report | ✓ `/dashboard/school/report` (admin/legacy) |
| Trend sparklines | ✓ `/dashboard/api/sparklines` |
| Score explainability | ✓ `/dashboard/api/explain_score` |
| Spektrum Keahlian integration | ✓ tapi **dilock ke Bidang 4** sekarang |

---

## 11. Daftar Bagian Proposal yang Perlu Direvisi

### High priority (factual mismatch)

1. **Judul** — pertimbangkan penambahan "Berbasis KKNI" dan/atau perluasan ke
   "...untuk Pendidikan Software & Game Development" (tidak hanya SMK RPL).
2. **BAB I §1.2 (RQ)** — RQ1 reframing: LLM-only primary, hybrid sebagai
   ablation. Tambah RQ baru tentang KKNI assignment quality dan/atau competency
   count quality.
3. **BAB II — seksi baru §2.X "KKNI"** dengan Perpres 8/2012, 9 jenjang,
   mapping Bloom→KKNI.
4. **BAB II §2.7–§2.10** — rewrite framing: LLM-first, BERT optional ablation.
5. **BAB III §3.2.3** — Develop Preliminary Product perlu mencerminkan:
   - Default LLM-only
   - Hard-skill input + soft-skill schema fields
   - KKNI hybrid assignment
   - Sub-clustering Uncertain
   - Three-stage dedup
   - Hard cap 12 per batch
   - Education-level extraction
   - Public-facing surface (landing, browse, detail, coverage)
   - Publish workflow
6. **BAB III §3.3 (Deskripsi Produk)** — full rewrite. Sekarang dua komponen
   utama:
   - **Backend**: pipeline 18+18 langkah resume-aware
   - **Frontend**: public surface (anonymous browse + coverage upload, light
     signup) + admin surface (data ops + publish)
7. **Gambar 3.2 (Pipeline Direction A)** — gambar baru: "Default LLM-only,
   Hybrid sebagai ablation".
8. **Gambar 3.5 (Mockup Dashboard)** — tambah mockup public-facing (landing
   dengan stage chips, browse cards, detail page).

### Medium priority (additions)

9. **BAB III §3.4 (Sampel)** — perluas ke responden publik (siswa, guru SMA,
   dosen D3/S1, peneliti).
10. **BAB III §3.6.3 (Kelayakan Produk)** — tambah metrik:
    - KKNI assignment accuracy (Top-1 vs expert label)
    - Competency count materiality (target: ~30–40% reduction vs baseline)
    - Per-batch competency count (≤12)
    - Coverage analyzer accuracy (presisi gap detection)
11. **Tabel 3.1** — tambah baris untuk evaluasi KKNI assignment dan coverage
    analyzer.

### Low priority (documentation)

12. **Daftar Rujukan** — tambah:
    - Perpres 8 Tahun 2012 tentang KKNI
    - Anderson & Krathwohl (2001) Bloom revisi (sudah disebut di teks tapi
      belum di rujukan)
13. **§2.6** — sebut bahwa "rekomendasi tingkat kurikulum institusional"
    sekarang juga **tersedia untuk publik luas**, bukan hanya internal sekolah.

---

## 12. Open Items / Limitasi yang Layak Disebut

- **Korpus saat ini bahasa Inggris** — Indonesian regex siap, data belum
  dikumpulkan.
- **KKNI assignment belum tervalidasi expert** — perlu metode Delphi pada tahap
  6 untuk validasi clamp band [floor, floor+1].
- **Sub-clustering threshold** untuk Uncertain batch (target sub-cluster size
  10) belum dieksperimentasi pada data realistis.
- **Coverage analyzer** menggunakan substring + normalized-token matching,
  bukan deep semantic matching — kurikulum yang ditulis dalam abreviasi
  ("OOP") mungkin missing.
- **Bidang 4 lockdown** mengasumsikan Kepmen 244/M/2024; jika user ingin RPL/PG
  terpisah (Kepmen 130/2017), pakai field `vocational_field` manual.

---

## 13. Snapshot Konfigurasi Final (untuk Lampiran Proposal)

```
config.py                            # PROJECT_ROOT, OUTPUT_DIR, RANDOM_SEED=42
kkni.py                              # NEW — KKNI 1-9 source of truth, Bloom→floor map
education_level_extractor.py         # NEW — regex KKNI extractor
compute_education_demand.py          # NEW — per-skill education demand aggregator
pipeline.py                          # default EXTRACTION_MODE = "llm_only"
generate_competencies.py             # hard-only input, KKNI fields, cap 12, sub-cluster
domain_batching.py                   # subcluster_unrelated_skills() for Uncertain/Unmapped
recommendations.py                   # priority = 0.40d + 0.30t + 0.30f, coverage ablation
skill_trend_holdout_validation.py    # RQ3 holdout
skill_normalizer.py                  # SBERT clustering 0.82
log_run_metadata.py                  # prompt SHA-256
preprocess_jobs_pipeline.py          # MD5 dedup + KKNI annotation
run.bat / run_phase_2.bat            # 18+18 steps, --resume aware
dashboard/app.py                     # FastAPI; admin + public + light signup
dashboard/api_public.py              # /api/* read-only public router
dashboard/publish.py                 # publish_results_dir + active_results_dir
dashboard/db.py                      # +published_runs, +coverage_analyses, +public role
dashboard/templates/public/*.html    # landing, browse, detail, coverage, about, signup, my_coverage
docs/PUBLIC_UI.md                    # NEW — surface architecture
```

---

## 14. Hand-off Instructions for the Other Claude Session

When you paste this brief into another Claude session along with the proposal
PDF, ask it to:

1. Update the title to reflect the broader scope (KKNI + multi-jenjang).
2. Reframe RQ1 (LLM-only primary, hybrid ablation) and add an RQ on competency
   quality / KKNI assignment.
3. Add a new section §2.X covering KKNI (Perpres 8/2012) as a theoretical anchor.
4. Rewrite §2.7–§2.10 framing (LLM-first, BERT for ablation).
5. Rewrite §3.2.3 and §3.3 to reflect current product (public + admin surface,
   KKNI module, education extraction, hard-skill competencies + soft-skill
   fields, hard cap 12, three-stage dedup, sub-clustering, publish workflow).
6. Replace Figure 3.2 (Direction A) and add a new Figure for the public-facing
   UI.
7. Expand §3.4 sampel to public users.
8. Update Table 3.1 with KKNI accuracy and competency count metrics.
