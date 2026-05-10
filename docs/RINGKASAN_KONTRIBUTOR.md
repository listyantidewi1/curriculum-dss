# Ringkasan Proyek untuk Kontributor

> Dokumen onboarding singkat untuk pengembang yang baru bergabung. Setelah
> membaca ini, Anda seharusnya tahu **apa** yang kami bangun, **mengapa**,
> **bagaimana arsitekturnya**, **status pengembangan saat ini**, dan
> **dari mana mulai berkontribusi**.
>
> Dokumen lain yang relevan:
> - [`PENJELASAN_UMUM.md`](PENJELASAN_UMUM.md) — untuk audiens non-teknis (siswa, guru, orang tua, pemangku kebijakan).
> - [`KAJIAN_AKADEMIK.md`](KAJIAN_AKADEMIK.md) — refleksi akademik untuk diskusi dengan promotor.
> - [`PROPOSAL_REVISION_BRIEF.md`](PROPOSAL_REVISION_BRIEF.md) — log keputusan teknis terbaru.
> - [`../PIPELINE.md`](../PIPELINE.md) — peta lengkap pipeline produksi (file per file, output per output).

---

## 1. Apa yang Kami Bangun

**Sistem Rekomendasi Kompetensi berbasis Data Pasar Kerja** untuk reformasi
kurikulum SMK di Indonesia. Sistem ini:

1. **Mengambil ribuan iklan lowongan kerja** dari berbagai platform (LinkedIn,
   Indeed, JobStreet, dll.).
2. **Mengekstrak skill, knowledge, dan kompetensi** yang diminta industri,
   memisahkan keterampilan teknis (hard skills) dari keterampilan
   non-teknis (soft skills) dan tools/teknologi (knowledge).
3. **Memetakan setiap skill ke domain pekerjaan masa depan** berbasis
   referensi WEF, O\*NET, dan McKinsey (mis. AI, Cloud, Data Analytics,
   Human–AI Collaboration).
4. **Mendeteksi tren** — skill yang naik (emerging), turun (declining),
   atau stabil — dengan analisis time-series ber-FDR.
5. **Menghasilkan kompetensi-statemen** yang dapat dipakai langsung untuk
   menyusun KOSP (Kurikulum Operasional Satuan Pendidikan), lengkap dengan
   label KKNI 1–9 dan level pendidikan tipikal yang diminta industri.
6. **Membandingkan kurikulum sekolah** terhadap kebutuhan pasar untuk
   mengidentifikasi *coverage gap*.
7. **Menyajikan semuanya** lewat dashboard publik dan admin yang
   menyediakan provenance lengkap — setiap kompetensi bisa ditelusuri
   sampai ke kalimat asli di iklan lowongan.

Output akhir: **rekomendasi kompetensi yang ter-rangking** berdasarkan
formula `0.40·demand + 0.30·trend + 0.30·future_weight`, siap dipakai
guru, ketua program keahlian, atau dinas pendidikan.

---

## 2. Mengapa Proyek Ini

Tiga motivasi utama:

**(a) Akademik — disertasi.** Proyek ini adalah implementasi disertasi
tentang sistem rekomendasi kompetensi berbasis NLP. Mengikuti metodologi
Borg & Gall; saat ini berada di tahap implementasi → uji lapangan.

**(b) Praktis — kesenjangan kurikulum SMK.** Industri perangkat lunak (dan
sektor lain) berubah jauh lebih cepat daripada siklus revisi kurikulum
SMK. Akibatnya, lulusan sering tidak menguasai skill yang sebenarnya
diminta. Sistem ini memberi sinyal kontinu dan obyektif tentang apa yang
sedang dibutuhkan pasar.

**(c) Kebijakan — KKNI alignment.** Setiap kompetensi yang dihasilkan
dilabeli level KKNI (Perpres 8/2012). Ini memungkinkan dinas pendidikan
membandingkan curriculum target SMK (umumnya KKNI 2–3) dengan kompetensi
real-world yang muncul di iklan lowongan untuk peran setara.

---

## 3. Arsitektur Tingkat Tinggi

```
[Job postings] → preprocess → relevance filter → extraction (hybrid) →
                                                        ↓
                                fusion → future-domain mapping → trend analysis
                                                        ↓
                                clustering → competency generation → KKNI labeler
                                                        ↓
                            recommendation ranking → dashboard publik + admin
```

### Komponen utama

| Tahap | Tanggung jawab | Status |
|---|---|---|
| **Preprocess** | Split sentence, deteksi bahasa, terjemahan ke EN, deduplikasi job-level | Stabil |
| **Sentence Relevance Filter** (Phase 1.2) | LLM zero-shot drop kalimat yang bukan skill (deskripsi perusahaan, benefit, lokasi, dll). Cache SHA-256 untuk biaya nol di re-run | Selesai |
| **Hybrid Extraction** | (i) BERT path: per-kalimat, deterministik. (ii) LLM path: full-posting, kontekstual. (iii) Fusion: rekonsiliasi keduanya | BERT path sedang di-upgrade dari JobBERT → Skill-LLM (LoRA LLaMA 3.1 8B). Lihat §4 |
| **Future-domain Mapping** | SBERT cosine sim antara skill dengan deskriptor domain WEF/O\*NET/McKinsey | Stabil |
| **Trend Detection** | Linear regression + FDR (Benjamini-Hochberg) terhadap frekuensi skill per bulan | Stabil |
| **Skill Clustering** (Phase 2.1) | HDBSCAN + agglomerative, pilih winner per batch berdasarkan SBERT cohesion. Gantikan domain-batching lama | Belum mulai |
| **Competency Generator** (Phase 2.2) | LLM (DeepSeek-V3 / GPT-5 / dll) sintesis kompetensi dari cluster + provenance | Akan ditulis ulang post-clustering |
| **KKNI Labeler** (Phase 2.3) | SBERT-based mapping kompetensi ke level KKNI 1–9 (descriptor Perpres 8/2012) | Belum mulai |
| **Education-Level Extractor** (Phase 2.4) | Tarik requirement pendidikan dari teks lowongan, agregasi per kompetensi | Sebagian sudah ada |
| **Competency Evaluator** (Phase 2.5) | Grounding score, coherence score, coverage score + expert rating | Belum mulai |
| **Dashboard** | Public (anonim) + School (login) + Admin (publish gate) | Stabil; akan di-extend dengan provenance UI di Phase 2.6 |

### Provenance — invariansi penting

Setiap skill yang diekstrak membawa `(job_id, sentence_id, sentence_text,
extractor_source)`. Setiap kompetensi yang dihasilkan membawa daftar
`contributing_item_ids` + `source_sentences`. Akibatnya, **setiap angka
di dashboard bisa ditelusuri sampai kalimat aslinya di lowongan kerja**.
Ini adalah requirement Req 6 dari mentor (dokumen
[.kiro/specs/pipeline-redesign-v2/requirements.md](../.kiro/specs/pipeline-redesign-v2/requirements.md))
— sistem tidak boleh "kotak hitam".

### Verb-vs-Noun discrimination

Penting: sistem membedakan secara eksplisit:

- **Skill (verb-led):** "designing UI/UX", "implementing CI/CD"
- **Knowledge (noun):** "UI/UX", "Python", "Docker"
- **Soft skill (noun/adjektif satu kata juga valid):** "passion", "empathetic",
  "self-starter"

Banyak skill extractor populer keliru memetakan "Python" sebagai skill,
padahal itu knowledge. Pipeline kami menjaga distinksi ini sepanjang stack
karena tahap downstream (competency generator, KKNI labeler) bergantung
pada verb sebagai sinyal cognitive level.

---

## 4. Status Pengembangan Saat Ini

Kami sedang mengerjakan **pipeline-redesign-v2**, perombakan besar
berdasarkan masukan mentor. Progres Phase 1 (data layer):

| Sub-phase | Pekerjaan | Status |
|---|---|---|
| **1.1** | Provenance per kalimat di seluruh pipeline | ✅ Selesai |
| **1.2** | Sentence relevance filter (zero-shot LLM) | ✅ Selesai |
| **1.3** | Hapus klasifikasi Bloom dari pipeline (kembalikan ke pemangku kepentingan) | ✅ Selesai |
| **1.4** | Replikasi `jjzha/jobbert_skill_extraction` sebagai BERT baseline | ✅ Selesai (lihat [REPLICATION_REPORT.md](../baseline_versions/jjzha_replicate/REPLICATION_REPORT.md)) |
| **1.5** | LoRA fine-tune Skill-LLM (LLaMA 3.1 8B) sebagai BERT-replacement | 🔄 Sedang training |

Phase 2 (pipeline reflow) belum dimulai; akan menggantikan domain-batching
dengan clustering, menulis ulang competency generator agar cluster-driven
dan provenance-aware, lalu menambah KKNI labeler post-hoc dan competency
evaluator.

### Hasil Phase 1.4 — temuan yang penting diketahui

Replikasi JobBERT yang dipublikasikan ternyata mendapat F1 yang setara
dengan vanilla BERT (skill 0.519, knowledge 0.653 pada SkillSpan test).
Target awal Phase 1.4 (skill ≥ 0.70, knowledge ≥ 0.80) tidak realistis —
literatur SOTA sendiri (Skill-LLM, Herandi et al. 2024) hanya mencapai
0.543 / 0.742. Target di-revisi menjadi:

- Skill F1 ≥ 0.54 (matches SOTA)
- Knowledge F1 ≥ 0.74 (matches SOTA)
- Total span F1 ≥ 0.65

Inilah alasan kami pivot ke Skill-LLM-style LoRA fine-tune di Phase 1.5.

---

## 5. Stack Teknis

| Lapisan | Teknologi |
|---|---|
| Bahasa utama | Python 3.10+ |
| ML / NLP | `transformers`, `peft`, `bitsandbytes`, `sentence-transformers`, `seqeval` |
| LLM API | OpenRouter (DeepSeek-V3 default; GPT-5 / Gemini / Claude bisa swap) |
| Web framework | FastAPI + Jinja2 (dashboard publik & admin) |
| Database | SQLite (dev), Postgres (prod) |
| Backbone extractor | JobBERT (legacy) → Skill-LLM LoRA LLaMA 3.1 8B (Phase 1.5+) |
| Statistik / FDR | `scipy.stats`, Benjamini-Hochberg |
| Plotting | matplotlib, seaborn |
| Repro | `git` + per-phase commit log; semua hyperparameter di `config.py` per-package |

Training berat (LoRA fine-tune) dilakukan di Kaggle (free P100/T4 16 GB);
inference berjalan di CPU/GPU lokal atau di Kaggle untuk batch besar.

---

## 6. Bagaimana Mulai Berkontribusi

Tergantung minat — empat entry points utama:

### (A) Pipeline / Extraction (Python, NLP)

Cocok jika Anda ingin: NER, fine-tuning, evaluasi model.

- Mulai dari [`PIPELINE.md`](../PIPELINE.md) untuk memahami alur file-per-file.
- Baca [`baseline_versions/jjzha_replicate/AUDIT.md`](../baseline_versions/jjzha_replicate/AUDIT.md)
  dan [`REPLICATION_REPORT.md`](../baseline_versions/jjzha_replicate/REPLICATION_REPORT.md) untuk
  memahami status BERT extractor.
- Baca [`baseline_versions/skill_llm/AUDIT.md`](../baseline_versions/skill_llm/AUDIT.md) untuk
  memahami arah Phase 1.5.
- Issue-issue berikutnya: integrasi Skill-LLM ke `pipeline.py`, ESCO
  normalization untuk knowledge items, perbaikan span boundary.

### (B) Clustering / Competency Generation (Python, embeddings, prompt engineering)

Cocok jika Anda ingin: unsupervised clustering, LLM prompt design,
competency synthesis.

- Mulai dari [`generate_competencies.py`](../generate_competencies.py) — generator saat ini.
- Phase 2.1 (HDBSCAN + agglomerative clustering) belum mulai. Ada
  prototype slot di plan.
- Phase 2.2 (rewrite generator agar cluster-driven dan provenance-aware)
  akan jadi PR besar setelah clustering siap.

### (C) Dashboard / Public UI (FastAPI, Jinja2, vanilla JS)

Cocok jika Anda ingin: backend web, UX research, accessibility.

- Mulai dari [`dashboard/`](../dashboard/) — sudah berjalan.
- Phase 2.6: tambahkan provenance UI ("Why this competency?"), filter
  KKNI level + education stage, coverage analyzer untuk curriculum upload.
- Lihat [`PUBLIC_UI.md`](PUBLIC_UI.md) untuk arsitektur public surface.

### (D) Evaluasi / Penelitian (statistik, IRR, expert review)

Cocok jika Anda ingin: scientific methodology, reliability statistics,
expert-rater coordination.

- Phase 2.5: implementasi grounding score, coherence score, coverage
  score, dan inter-rater reliability (Cohen's / Fleiss' Kappa).
- Sistem human-in-the-loop sudah ada di [`gold_labeling_ui/`](../gold_labeling_ui/) dan
  [`review_ui/`](../review_ui/) — perlu protokol review yang lebih ketat.
- Lihat [`SCIENTIFIC_METHODOLOGY.md`](../SCIENTIFIC_METHODOLOGY.md) (jika ada) atau
  [`KAJIAN_AKADEMIK.md`](KAJIAN_AKADEMIK.md) untuk konteks akademik.

---

## 7. Bacaan Wajib Sebelum PR Pertama

Urutan baca yang direkomendasikan:

1. [`PENJELASAN_UMUM.md`](PENJELASAN_UMUM.md) — 10 menit, konteks domain.
2. Ringkasan ini (`RINGKASAN_KONTRIBUTOR.md`) — sudah Anda baca.
3. [`PIPELINE.md`](../PIPELINE.md) — peta lengkap file → output, ~30 menit.
4. [`.kiro/specs/pipeline-redesign-v2/requirements.md`](../.kiro/specs/pipeline-redesign-v2/requirements.md) —
   sumber otoritas untuk semua keputusan arsitektur saat ini.
5. Kode yang relevan dengan area kontribusi Anda (Bagian 6).

Setelah itu, koordinasi area pekerjaan dengan tim untuk menghindari
tabrakan dengan PR yang sedang berjalan.

---

## 8. Konvensi Pengembangan

- **Bahasa commit message:** Inggris (commit log harus tetap konsisten
  dengan tooling internasional).
- **Bahasa dokumentasi:** Indonesia untuk dokumen publik / non-teknis;
  Inggris untuk dokumen teknis dalam-package (`AUDIT.md`, `README.md`,
  `REPLICATION_REPORT.md`).
- **Branching:** trunk-based; commit langsung ke `main` setelah review
  ringan untuk perubahan kecil. PR untuk perubahan besar atau lintas-area.
- **Hyperparameter:** semuanya di `config.py` per-package, jangan
  hard-code di train/eval scripts.
- **Provenance invariant:** setiap perubahan di pipeline harus tetap
  membawa `sentence_id` + `sentence_text` di output. Lihat
  [pipeline.py SkillItem dataclass](../pipeline.py).
- **Verb-noun invariant:** jangan kompromi distinksi skill (verb-led)
  vs knowledge (noun) di tahap ekstraksi maupun fusion.
- **Bloom taxonomy:** **DILARANG** kembali ke pipeline. Telah dihapus
  per Req 1 (Bloom diserahkan ke pemangku kepentingan kurikulum).

Selamat datang. 🚀
