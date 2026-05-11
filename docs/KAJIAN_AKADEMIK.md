# Kajian Akademik Sistem Rekomendasi Kompetensi

> Dokumen reflektif untuk diskusi dengan promotor/pembimbing.
> Disusun setelah tahap implementasi rampung, sebelum tahap uji lapangan
> (Borg & Gall Tahap 4 ke atas) dimulai. Dimaksudkan sebagai bahan
> percakapan, bukan section disertasi yang sudah final.

---

## Pengantar dan Tujuan

Sistem rekomendasi kompetensi yang dikembangkan dalam disertasi ini telah
mencapai bentuk fungsional. Sistem mencakup pipeline ekstraksi-keterampilan
dua-fase berbasis pendekatan hibrida JobBERT+LLM, integrasi Kerangka
Kualifikasi Nasional Indonesia (KKNI) sesuai Perpres 8/2012, ekstraksi
level pendidikan dari teks lowongan, mekanisme publikasi-terkurasi
(*publish-gate*) untuk laman publik, serta dashboard yang memisahkan akses
publik anonim, akun ringan, dan administrator. Implementasi dilakukan
dalam beberapa iterasi termasuk satu kali pivot besar berdasarkan
masukan promotor (perubahan default ekstraksi dari hibrida menjadi
LLM-only, dengan jalur hibrida dipertahankan sebagai ablation untuk RQ1).

Dokumen ini berfungsi sebagai bahan refleksi sebelum tahap uji lapangan
dimulai. Setiap keputusan desain dibedah dari empat sudut pandang:
**(1) justifikasi metodologis** terhadap akar teoretisnya,
**(2) pemetaan kontribusi** relatif terhadap literatur yang menjadi
rujukan proposal, **(3) ancaman validitas dan reliabilitas** yang
terbawa, dan **(4) refleksi praktis serta arah lanjutan** yang terbuka.
Tone yang dipilih adalah lugas dan boleh self-critical—tujuannya
mengundang diskusi mendalam, bukan membela diri.

Pembaca dapat menelusuri dokumen ini secara linear, atau langsung loncat
ke bagian tertentu sesuai fokus diskusi. Bagian §5 memuat poin-poin
spesifik yang diharapkan menjadi agenda diskusi.

---

## §1 Justifikasi Metodologis Tiap Keputusan Desain

Tabel berikut merangkum lima belas keputusan desain kunci, akar
teoretisnya, alternatif yang ditolak, dan trade-off yang diterima.
Beberapa keputusan kritis di-elaborasi lebih lanjut setelah tabel.

| No | Keputusan | Akar Teori | Alternatif Ditolak | Trade-off |
|----|-----------|------------|--------------------|-----------|
| 1 | Default ekstraksi `llm_only`, hibrida untuk ablation | Senger et al. (2024) Direction A; konsultasi promotor | Hibrida sebagai default | Presisi BERT-spesifik vs reasoning kontekstual LLM |
| 2 | Pemetaan domain *post-extraction* via SBERT | Reimers & Gurevych (2019) | Klasifikasi domain di dalam panggilan LLM | Independensi evaluasi vs satu LLM call |
| 3 | Filter input kompetensi: hanya hard-skills | Konsultasi promotor; CBET (Mulder et al. 2007) | Input campur hard+soft | Simplifikasi vs hilangnya konteks soft langsung |
| 4 | Soft-skills sebagai field terpisah (hybrid sourcing) | Galster et al. (2023): 82% iklan minta soft skills | Soft-skill mengikuti related_skills | Dua sumber kebenaran vs konteks lebih kaya |
| 5 | Hard cap 12 kompetensi per *batch* | Pragmatic post-hoc enforcement | Prompt-only "aim for 8–20" yang LLM abaikan | Kontrol output vs natural LLM judgment |
| 6 | Sub-clustering "Uncertain"/"Unmapped" via SBERT agglomerative | Keluhan promotor: skill tidak terkait dilumpukkan | *Drop* skill berkonfidensi rendah | Menjaga *recall* vs menambah kompleksitas |
| 7 | *Three-stage deduplication* (normalized + semantic + Jaccard) | Literatur clustering | *Title-only matching* | Kompleksitas vs presisi dedup |
| 8 | ~~KKNI assignment hibrida (Bloom-floor + LLM clamp band)~~ → **direvisi: SBERT post-hoc labeler (Phase 2.3)** | Perpres 8/2012 deskriptor resmi | LLM-only assignment; rule-only assignment | Reliabilitas SBERT match vs ground-truth ahli KKNI |
| 9 | Ekstraksi level pendidikan dari teks lowongan | ILO (2022) job-posting analysis | Asumsikan satu level untuk seluruh korpus | Regex tidak menangkap variasi natural |
| 10 | Bobot prioritas 0.40 demand + 0.30 trend + 0.30 future, w_coverage=0 | *Design intent* "curriculum reform tool"; coverage ablation report | Coverage masuk bobot prioritas | Reformasi kurikulum vs *compliance audit* |
| 11 | *Public-first dashboard* (anonymous browse + light signup) | Pragmatik penyebarluasan; di luar literatur | Akses penuh hanya via login | Aksesibilitas vs analitik perilaku |
| 12 | *Domain lockdown* ke Bidang 4 (Teknologi Informasi) | Kepmen 244/M/2024; *scope decision* | Multi-bidang sejak awal | Kedalaman vs keluasan |
| 13 | *Publish workflow* (snapshot canonical run, `is_active`) | *Engineering decision*: cegah *flicker* saat re-run | Public membaca live `results/` | Kontrol versi vs latensi update |
| 14 | *Resume-aware checkpointed pipeline* | *Operational pragmatism* | Rerun penuh setiap kali | Robustness operasional vs kompleksitas state |
| 15 | *Bundle* evaluasi multi-metrik | Brown et al. (2001), Murphy (1973), Voorhees (1999), Reimers & Gurevych (2019) | Single metric (Accuracy / F1) | Rigor tinggi vs laporan kompleks |

### 1.1 Mengapa LLM-only menjadi default

Pivot dari hibrida ke LLM-only dipicu oleh konsultasi promotor pada
pertengahan implementasi. Senger et al. (2024) memang menyarankan
pendekatan hibrida (Direction A), namun argumen mereka berasumsi
keterbatasan *context window* LLM yang berlaku pada saat publikasi.
Dengan model generasi terkini yang mampu memproses dokumen utuh,
peran JobBERT sebagai *anti-hallucination context* menjadi kurang
mendesak. JobBERT tetap dipertahankan sebagai jalur ablation
(`--extraction-mode hybrid`) sehingga RQ1 tetap dapat menguji apakah
penambahan BERT memberikan *lift* presisi yang material—justru
diuji secara empiris alih-alih diasumsikan.

### 1.2 Mengapa pemetaan domain tetap di luar LLM

Promotor menyarankan agar LLM "mengklasifikasi domain" sekaligus dengan
ekstraksi. Saran ini dipertimbangkan namun tidak diadopsi penuh.
Pemetaan domain via SBERT cosine ke `future_domains.csv` tetap
dipertahankan karena tiga alasan: pertama, sumber taksonomi domain
(WEF/O\*NET/McKinsey) bersifat eksternal sehingga lebih cocok diolah
sebagai *retrieval problem*, bukan *generation problem*. Kedua, hasil
SBERT dapat dievaluasi independen terhadap `gold_future_domain.csv`
(RQ4) sehingga akurasi pemetaan terdokumentasi terpisah dari
akurasi ekstraksi. Ketiga, pendekatan ini menghindari ketergantungan
hasil terhadap satu *prompt template* yang dapat menjadi *single point
of failure*.

### 1.3 Mengapa hard cap 12 kompetensi

Selama eksperimen pra-pivot, *prompt template* meminta LLM "aim for 8–20
competencies per batch". Dalam praktik, LLM kerap menghasilkan 25–35
kompetensi per *batch* dan menyebabkan keluhan promotor bahwa
"kompetensi terlalu banyak dan kelompok skill kadang tidak terkait".
Solusi *prompt-only* tidak konsisten; oleh karena itu ditambahkan
mekanisme *post-hoc enforcement*: setelah LLM merespons, sistem memilih
12 kompetensi paling ter-anchor (terbanyak `related_skills`-nya). Pilihan
12 bukan hasil eksperimen formal, melainkan kompromi antara kerapatan
informasi dan kelelahan kognitif pembaca. Eksperimen sensitivitas
terhadap *cap* (8 vs 12 vs 16) belum dilakukan dan disebutkan sebagai
arah lanjutan di §4.3.

### 1.4 Mengapa SBERT post-hoc untuk KKNI assignment (revisi May 2026)

**Catatan revisi:** subseksi ini awalnya mendokumentasikan pendekatan
*hybrid* (Bloom-floor + LLM clamp band) yang sempat diimplementasikan di
[kkni.py](../kkni.py). Pendekatan tersebut **dihapus dalam
pipeline-redesign-v2 Phase 1.3 (Mei 2026)** seiring dengan penghapusan
klasifikasi Bloom dari pipeline secara menyeluruh per Req 1 (lihat
[`.kiro/specs/pipeline-redesign-v2/requirements.md`](../.kiro/specs/pipeline-redesign-v2/requirements.md)).

**Pendekatan asli (yang dihapus):** sistem menghitung *floor*
deterministik dari level Bloom tertinggi di antara `related_skills`
(Apply→4, Analyze→5, Evaluate→6, Create→7), lalu meminta LLM memilih
level KKNI dalam *band* [floor, floor+1]. Output di luar *band*
di-*clamp* kembali ke *floor*. Argumen pendukung: menggabungkan
reliabilitas aturan eksplisit dengan kemampuan LLM membedakan
kompleksitas yang sulit dirumuskan sebagai aturan kaku.

**Mengapa direvisi.** Tiga alasan, dalam urutan kepentingan:

1. **Bloom sebagai layer pipeline tidak konsisten dengan brief promotor.**
   Promotor meminta agar keputusan Bloom diserahkan kepada pemangku
   kepentingan kurikulum (SMK / dinas pendidikan), bukan dipaksakan oleh
   pipeline ekstraksi. Bloom-floor sebagai jembatan deterministik
   ke KKNI berarti pipeline tetap mengambil keputusan Bloom internal
   yang kemudian *propagate* ke output—berlawanan dengan brief.
2. **Heuristik [floor, floor+1] tidak tervalidasi.** Sebagaimana diakui
   pada §3.1, *band* tersebut belum diuji terhadap penilaian ahli KKNI.
   Pendekatan post-hoc SBERT yang baru menghindari heuristik ini
   sepenuhnya: kompetensi dibandingkan langsung terhadap deskriptor
   level KKNI 1–9 dari Perpres 8/2012, dengan skor cosine similarity
   sebagai *evidence trail* yang bisa diaudit.
3. **Separasi yang lebih bersih.** Dengan pendekatan baru, keputusan
   "level kompetensi" sepenuhnya menjadi *retrieval problem* (cocokkan
   teks kompetensi ke deskriptor level), bukan *chained reasoning*
   (Bloom→KKNI band→LLM clamp). Ini lebih mudah dievaluasi secara
   independen melalui `gold_kkni.csv` di Tahap 6.

**Pendekatan baru (Phase 2.3, queued).** File `kkni_labeler.py` (akan
ditulis): encode setiap deskriptor KKNI 1–9 dari Perpres 8/2012 sebagai
vektor SBERT; encode `title + description` setiap kompetensi yang
dihasilkan; assign level dengan cosine similarity tertinggi; record
top-3 candidate levels untuk transparansi. Field hasil:
`kkni_level`, `kkni_level_top3`, `kkni_descriptor`,
`kkni_match_similarity`. Bersifat *informational*, tidak masuk ranking
priority (Req 2.5).

---

## §2 Pemetaan Kontribusi vs State-of-the-Art

Tabel berikut memetakan setiap rujukan literatur kunci dari proposal
disertasi terhadap posisi yang diambil sistem ini.

| Literatur Rujukan | Posisi | Justifikasi |
|-------------------|--------|-------------|
| Senger et al. (2024)—Direction A hybrid | **Adaptasi** | Default LLM-only; hibrida menjadi jalur ablation untuk RQ1 |
| Zhang et al. (2022)—SkillSpan + JobBERT | **Adopsi** | Dipakai sebagai komponen ekstraksi (mode hibrida) tanpa retraining |
| Reimers & Gurevych (2019)—SBERT cosine | **Adopsi langsung** | Dipakai untuk pemetaan domain, sub-clustering Uncertain, dan dedup semantic-title |
| Hassan et al. (2023)—Norwegian curriculum recommender | **Perluasan** | Ditambahkan KKNI alignment, public-first architecture, dan ekstraksi level pendidikan |
| Wang et al. (2021); Chen & Zhong (2024)—MOOC GCN/GNN | **Kontras** | Unit analisis berbeda: kurikulum institusional, bukan personal learning path |
| Norton (1997)—DACUM | **Adaptasi modern** | Filosofi DACUM dioperasikan secara digital pada ribuan iklan |
| Tyler (1949); Taba (1962)—desain kurikulum | **Operasionalisasi** | Tahap pertama Taba ("diagnosis kebutuhan") direalisasikan langsung via endpoint `/coverage` |
| Brown et al. (2001)—Wilson Score CI | **Adopsi** | Estimasi presisi pada data ekstraksi yang *sparse* |
| Murphy (1973)—dekomposisi Brier | **Adopsi** | Evaluasi kalibrasi probabilitas modul *confidence scoring* |
| Anderson & Krathwohl (2001)—taksonomi Bloom revisi | ~~**Adopsi struktural**~~ → **Tidak dipakai (revisi v2)** | Awalnya digunakan untuk mapping Bloom→KKNI floor; dihapus dalam pipeline-redesign-v2 Phase 1.3. KKNI assignment kini melalui SBERT match terhadap deskriptor Perpres 8/2012 (Phase 2.3) tanpa perantara Bloom. |
| Perpres 8/2012—KKNI | **Sumber kebenaran tunggal** | Seluruh kerangka jenjang kompetensi mengikuti deskriptor resmi KKNI 1–9 |

### 2.1 Identifikasi gap yang diisi

Beberapa kontribusi yang tidak ditemukan dalam literatur yang dirujuk:

1. **Penyandingan rekomendasi kompetensi dengan KKNI Indonesia.**
   Hassan et al. (2023) membahas universitas Norwegia tanpa kerangka
   kualifikasi nasional yang spesifik. Penelitian ini menjadi yang
   pertama mengintegrasikan rekomendasi berbasis data lowongan kerja
   dengan struktur 9 jenjang KKNI (Perpres 8/2012). Hal ini secara
   khusus relevan untuk konteks pendidikan vokasi Indonesia karena
   memungkinkan *filter* kompetensi yang sesuai dengan jenjang sekolah
   (SMA, SMK, D3, S1, dst.) tanpa intervensi ahli kurikulum tambahan.

2. **Public-facing competency browser yang KKNI-aware.**
   Sebagian besar sistem rekomendasi pendidikan dalam literatur bersifat
   internal (untuk lembaga atau kelas tertentu). Sistem ini membuka
   akses langsung ke publik luas—siswa, guru, dosen, peneliti, praktisi
   industri—tanpa kewajiban login. Implikasinya: rekomendasi menjadi
   *common good* yang dapat dimanfaatkan oleh siapa pun untuk berbagai
   tujuan (lihat [docs/PUBLIC_UI.md](docs/PUBLIC_UI.md)).

3. **Pemisahan hard/soft di skema output kompetensi.**
   Galster et al. (2023) menunjukkan bahwa 82% iklan lowongan kerja
   secara eksplisit menyebut soft skills, namun literatur tentang
   ekstraksi keterampilan jarang memisahkan hard dan soft di tingkat
   *output* kompetensi. Sistem ini memisahkan keduanya secara struktural:
   hard skills sebagai *related_skills*, soft skills sebagai
   *soft_skills_required* dan *soft_skills_description* per kompetensi.
   Pemisahan ini mendukung praktik CBET (Mulder et al. 2007) yang
   menempatkan keterampilan teknis sebagai inti kompetensi sekaligus
   mengakui bahwa konteks kerja menuntut keterampilan non-teknis.

4. **Penggunaan bundle multi-metrik komprehensif.**
   Banyak studi NLP untuk ekstraksi keterampilan hanya melaporkan
   F1 (Senger et al. 2024) atau Top-1 accuracy. Sistem ini mengadopsi
   *bundle* yang mencakup Wilson Score CI (presisi), Brier dengan
   dekomposisi Murphy (kalibrasi probabilitas), AUC-ROC (diskriminasi),
   FDR Benjamini-Hochberg (deteksi tren), Cohen's/Fleiss' Kappa
   (reliabilitas reviewer), Jaccard (stabilitas algoritmik), Durbin-Watson
   (autokorelasi pada deret waktu), serta MRR dan NDCG@20 (kualitas
   pemeringkatan). Konsekuensi: rigor empiris meningkat namun laporan
   menjadi lebih panjang dan kompleks.

---

## §3 Ancaman Validitas dan Reliabilitas

### 3.1 Validitas Internal

**Ketergantungan pada LLM eksternal (DeepSeek).**
Sistem mendelegasikan ekstraksi dan generasi kompetensi ke LLM yang
diakses lewat OpenRouter API. Walaupun *temperature* di-set ke 0
dan *seed* terkontrol, LLM eksternal dapat menerima *update model*
oleh penyedia layanan tanpa pemberitahuan, sehingga reproduksibilitas
absolut tidak terjamin. Mitigasi yang sudah dipasang adalah
*prompt versioning* dengan SHA-256 ([log_run_metadata.py](log_run_metadata.py)),
yang memungkinkan deteksi bila *prompt* berubah, tetapi tidak
mendeteksi perubahan *internal* model.

**Sensitivitas prompt engineering.**
Kualitas hasil LLM sangat bergantung pada formulasi *prompt*. Belum
ada *ablation study* formal yang mengukur *delta* output relatif terhadap
varian prompt. (Catatan revisi: rule "BLOOM ALIGNMENT" yang sempat ada
di prompt `generate_competencies.py` dihapus dalam pipeline-redesign-v2
Phase 1.3 bersama dengan seluruh logika Bloom.)

**~~KKNI clamp band heuristik belum tervalidasi expert.~~ *(tidak lagi
relevan setelah revisi v2)*.**
Pemilihan *band* [floor, floor+1] yang dulu ada di `kkni.py` sebagai
heuristik Bloom→KKNI dihapus dalam pipeline-redesign-v2 Phase 1.3 dan
digantikan oleh SBERT post-hoc labeler (lihat §1.4 revisi). Ancaman
validitas yang asli (band tidak tervalidasi expert) telah hilang
karena pendekatan band itu sendiri tidak lagi ada. Ancaman baru yang
relevan: kualitas SBERT match terhadap deskriptor Perpres 8/2012—akan
divalidasi via metode Delphi pada Tahap 6 Borg & Gall melalui
`gold_kkni.csv` (rencana Phase 2.5 evaluator).

**Sub-clustering threshold belum dieksperimentasi.**
[domain_batching.py:113](domain_batching.py#L113) mendefinisikan
`_SUBCLUSTER_THRESHOLD = 15` (sub-cluster hanya dilakukan jika n>15)
dan `_SUBCLUSTER_TARGET_SIZE = 10` (target ukuran tiap sub-cluster).
Kedua angka ini ditetapkan berdasarkan intuisi, bukan hasil eksperimen
formal. Konsekuensinya: kelompok skill yang seharusnya dipecah lebih
halus mungkin tidak dipecah, atau sebaliknya.

### 3.2 Validitas Eksternal

**Korpus saat ini berbahasa Inggris.**
Iklan lowongan yang dipakai untuk prototipe diambil dari sumber
berbahasa Inggris. Generalisasi ke konteks Indonesia bergantung pada
asumsi bahwa kebutuhan industri perangkat lunak global mencerminkan
kebutuhan industri Indonesia—asumsi yang masuk akal untuk SE/GD
namun belum tervalidasi empiris. Dukungan untuk korpus Indonesia
sudah disiapkan ([education_level_extractor.py](education_level_extractor.py)
memiliki *regex* siap untuk istilah Indonesia seperti "S1 Teknik
Informatika"), namun datanya belum dikumpulkan.

**Single domain (Software & Game Development).**
Sistem dikunci pada Bidang 4 Teknologi Informasi (Kepmen 244/M/2024).
Generalisasi ke bidang lain (akuntansi, pariwisata, kuliner) belum
diuji. Beberapa komponen sistem—khususnya pemetaan ke
`future_domains.csv` dan ekstraksi level pendidikan—mungkin
memerlukan adaptasi domain-spesifik.

**Audiens yang lebih luas dari SMK.**
Pasca-konsultasi promotor Mei 2026, mission sistem diperluas: bukan hanya
SMK Indonesia, melainkan **siapa saja yang merancang kurikulum software
engineering** — SMK, universitas, fakultas vokasi (D3/D4), lembaga
pendidikan tinggi, dinas pendidikan, atau pemerintah. Cohort uji pengguna
pertama tetap SMK + fakultas vokasi universitas di Indonesia (KKNI 2–6
relevant), namun arsitektur sistem (provenance per kalimat, hybrid
extraction Layer 1 + Layer 2, grounding gate untuk anti-hallucination)
sengaja generik. Implikasi: KKNI labeler tetap menjadi *informational
metadata*, bukan filter wajib — konsumen institusional yang tidak
beroperasi dalam kerangka KKNI dapat mengabaikannya tanpa kehilangan
fungsionalitas. *Education-level extractor* memiliki regex Indonesia-
spesifik (S1/D3/D4/SMK) namun akan diperluas dengan padanan ISCED untuk
audiens internasional.

**Sample size dan representativitas industri.**
Default `SAMPLE_SIZE` adalah 1000 lowongan, dengan rentang produksi
hingga 10.000 selama 12 bulan. Apakah sampel ini representatif terhadap
*spectrum* industri perangkat lunak Indonesia (perusahaan kecil, BUMN,
*startup*, konsultan, dst.)? Belum ada *stratified sampling design*
formal—sampel diambil dari portal lowongan publik berdasarkan
ketersediaan, bukan stratifikasi sengaja.

### 3.3 Validitas Konstruk

**Definisi "kompetensi" sebagai cluster of related skills.**
CBET (Mulder et al. 2007) mendefinisikan kompetensi sebagai
"kemampuan menerapkan pengetahuan, keterampilan, dan sikap secara
terpadu dalam konteks kerja yang nyata". Sistem ini menyederhanakan
definisi tersebut menjadi *cluster* keterampilan teknis yang relevan
plus daftar soft skills. Dimensi "sikap" tidak terwakili eksplisit;
"pengetahuan" tertangkap parsial via ekstraksi `advanced_knowledge.csv`
namun tidak masuk skema kompetensi final. Kesenjangan ini perlu didiskusikan
dengan promotor: apakah kompetensi sistem adalah kompetensi CBET yang
disederhanakan, atau konsep yang berbeda namanya saja sama?

**Tren sebagai regresi linier frekuensi bulanan.**
[skill_time_trend_analysis.py](skill_time_trend_analysis.py)
mengoperasionalkan "tren" sebagai *slope* regresi linier dari frekuensi
kemunculan skill per bulan, dengan koreksi FDR Benjamini-Hochberg.
Apakah ini *proxy* yang valid untuk "kebutuhan masa depan"? Tidak
sepenuhnya—frekuensi iklan adalah *lagging indicator*; perusahaan
mempublikasikan lowongan setelah memutuskan bahwa kebutuhan ada.
Untuk benar-benar memprediksi masa depan, dibutuhkan sumber lain
(laporan WEF, O\*NET, McKinsey) yang sudah diintegrasikan via
`future_domains.csv` namun bobotnya tetap 0.30 (lebih kecil dari demand
empiris 0.40).

**Coverage analysis pakai substring + normalized-token match.**
Endpoint `/coverage` di [dashboard/api_public.py](dashboard/api_public.py)
membandingkan kurikulum yang diunggah dengan kompetensi sistem
menggunakan *substring matching* setelah normalisasi token. Pendekatan
ini sederhana dan dapat mendeteksi sebagian besar *match* literal,
tetapi gagal pada kasus singkatan (misalnya kurikulum menulis "OOP",
sistem menulis "Object-Oriented Programming"). *Measurement validity*
terbatas; perbaikan menggunakan SBERT semantic match adalah arah
lanjutan.

### 3.4 Reliabilitas

**Output LLM bersifat stochastic.**
Walaupun *temperature*=0 dan *seed* terkontrol, LLM via API tetap
dapat menghasilkan output yang sedikit berbeda antar pemanggilan,
khususnya jika layanan API melakukan *load balancing* antar replika.
Mitigasi: setiap *run* mendokumentasikan SHA-256 dari *prompt*
([log_run_metadata.py](log_run_metadata.py)) dan *commit hash* git,
sehingga perubahan dapat dirunut.

**Threshold SBERT clustering tanpa sensitivity analysis eksplisit.**
Sistem menggunakan tiga *threshold* SBERT yang berbeda:
- `0.82` untuk *skill normalization* di [skill_normalizer.py](skill_normalizer.py)
- `0.85` untuk *semantic title dedup* di [generate_competencies.py:567](generate_competencies.py#L567)
- `0.45` untuk *domain assignment confidence* di
  [generate_competencies.py:702](generate_competencies.py#L702)

Setiap *threshold* dipilih berdasarkan rekomendasi konvensi atau intuisi.
*Sensitivity analysis* yang sistematis (misalnya: *sweep* 0.75–0.95
dengan *step* 0.05 dan ukur *delta* output) belum dilakukan.

**FDR threshold q=0.05.**
Konvensi statistik yang lazim, namun tetap arbitrer. Di domain dengan
ribuan skill, q=0.05 berarti 5% dari skill yang dilaporkan sebagai
"emerging" mungkin *false positive*. Untuk konteks rekomendasi
kurikulum, apakah ini cukup ketat? Atau perlu q=0.01 yang lebih
konservatif tetapi mengurangi *power*?

---

## §4 Refleksi Praktis dan Arah Lanjutan

### 4.1 Pelajaran dari Pengembangan

**Pivot mentor sebagai bukti pentingnya keluwesan arsitektur.**
Pivot dari hibrida ke LLM-only dilakukan setelah kode hibrida sudah
berjalan penuh. Karena arsitektur memisahkan ekstraksi (pipeline.py),
verifikasi (verify_skills.py), pemetaan domain (future_weight_mapping.py),
analisis tren (skill_time_trend_analysis.py), dan generasi kompetensi
(generate_competencies.py), pivot dapat dieksekusi tanpa *rewrite*
besar—hanya mengubah *default flag* dan menambahkan jalur ablation.
Pelajaran: investasi awal pada *modular pipeline* terbayar saat
kebutuhan riset berubah.

**Hard cap 12 sebagai bukti bahwa instruksi prompt saja tidak cukup.**
*Prompt template* yang menyatakan "aim for 8–20 competencies per batch"
secara eksplisit, lengkap dengan justifikasi pedagogis, tetap diabaikan
oleh LLM dalam ~30% pemanggilan. Hal ini mengkonfirmasi temuan literatur
*prompt engineering* bahwa LLM cenderung mengikuti pola distribusi
output dari *training data* alih-alih mematuhi instruksi spesifik.
*Post-hoc enforcement* (memilih top-12 berdasarkan jumlah related_skills)
adalah solusi pragmatis namun mengangkat pertanyaan: berapa banyak
informasi yang hilang akibat *cap* ini?

**Public surface lahir dari permintaan praktis, bukan literatur.**
Keputusan untuk membuka akses publik anonim tidak datang dari kerangka
teoretis manapun—muncul dari pertanyaan praktis pengguna ("siapa yang
bisa pakai ini?"). Ini adalah fenomena umum dalam *software development*
namun jarang dibahas dalam disertasi. Apakah hal seperti ini layak
masuk ke disertasi sebagai "kontribusi praktis" yang berbeda dari
"kontribusi teoretis"? Layak didiskusikan dengan promotor.

### 4.2 Trade-off yang Diterima

1. **Reproducibility vs flexibility.** LLM eksternal dipilih meskipun
   mengorbankan reproducibility absolut, demi kemampuan reasoning yang
   tidak tersedia pada model lokal saat ini. Mitigasi: *prompt versioning*
   dan dokumentasi seed.

2. **Coverage vs Precision di formula prioritas.**
   `w_coverage=0.0` memaksa sistem fokus pada *gap reform* alih-alih
   *compliance audit*. Sekolah yang ingin mempertahankan kompetensi
   yang sudah ada di kurikulum tidak mendapat dukungan langsung dari
   skor prioritas, tetapi tetap dapat melihatnya via dashboard *insights*.

3. **Anonymous browse vs analytics.** Public surface tanpa login
   memaksimalkan aksesibilitas namun mengorbankan kemampuan menelusuri
   perilaku pengguna untuk *iterative improvement*. Akun ringan mengisi
   sebagian celah, namun adopsi sukarela menyebabkan *self-selection bias*.

4. **Single-source vs multi-source data.** Hanya menggunakan iklan
   lowongan kerja sebagai sinyal kebutuhan industri. Sumber alternatif
   (survei industri, *job description* internal perusahaan, sertifikasi
   industri) tidak diintegrasikan. Trade-off: fokus eksperimen
   vs lengkap-tetapi-kompleks.

5. **Hard cap vs natural LLM judgment.** Memaksa maksimal 12 kompetensi
   per *batch* mengorbankan kemampuan LLM membedakan situasi di mana
   suatu *batch* benar-benar memerlukan lebih banyak kompetensi
   (misalnya batch dengan skill sangat heterogen). Belum ada mekanisme
   adaptif yang menyesuaikan *cap* dengan keragaman skill input.

### 4.3 Arah Lanjutan Terbuka

1. **Validasi expert KKNI assignment via metode Delphi.**
   Tahap 6 Borg & Gall direncanakan melibatkan panel ahli (≥5 orang
   dari kurikulum vokasi dan industri perangkat lunak) untuk memvalidasi
   level KKNI yang ditetapkan sistem. Target konsensus ≥75% setelah
   2 ronde. Hasil validasi akan menjadi *ground truth* untuk mengevaluasi
   seberapa sering *clamp band* [floor, floor+1] sudah cukup vs perlu
   diperluas.

2. **Korpus Indonesia + multi-bidang.**
   Setelah prototipe Inggris+SE/GD divalidasi, langkah berikutnya adalah
   mengumpulkan korpus berbahasa Indonesia (target 5.000–10.000 lowongan)
   dan memperluas ke bidang lain (akuntansi, pariwisata, kuliner). Hal ini
   memerlukan adaptasi `future_domains.csv` dan kemungkinan *fine-tuning*
   model bahasa untuk domain spesifik.

3. **Eksperimen formal sub-clustering threshold.**
   *Sweep* parameter `_SUBCLUSTER_THRESHOLD` (5, 10, 15, 20) dan
   `_SUBCLUSTER_TARGET_SIZE` (5, 10, 15) dengan ukuran metrik kohesi
   *batch* (rata-rata SBERT cosine intra-batch) dan kepuasan reviewer.

4. **Studi komparatif: rekomendasi sistem vs panel ahli DACUM
   tradisional.** Penelitian terpisah yang membandingkan kompetensi
   rekomendasi sistem dengan kompetensi yang dihasilkan workshop DACUM
   manual selama 2–3 hari. Jika hasilnya konvergen pada >70% kompetensi
   inti, ini akan menjadi argumen kuat bagi sistem sebagai pengganti
   workshop DACUM.

5. **Tracking longitudinal efek implementasi.**
   Apakah sekolah yang menggunakan sistem benar-benar mengubah KOSP-nya?
   Apakah perubahan tersebut menghasilkan lulusan yang lebih cepat
   terserap industri? Studi *quasi-experimental* dengan kelompok
   kontrol (sekolah tanpa akses sistem) selama 2–3 tahun ajaran
   diperlukan untuk klaim efektivitas yang kuat.

---

## §5 Penutup

Sistem yang dikembangkan dalam disertasi ini diposisikan sebagai
**instrumen sistematis untuk diagnosis kebutuhan kurikulum**, bukan
sebagai pengganti pertimbangan profesional ahli kurikulum. Dalam
*framework* Taba (1962), sistem mengoperasikan *Langkah 1: Diagnosis
Kebutuhan*; *Langkah 2–7* (perumusan tujuan, pemilihan konten,
pengorganisasian, dst.) tetap menjadi domain ahli. Sistem juga
memodernisasi filosofi DACUM (Norton 1997) dengan menggantikan
*workshop* tertutup berskala kecil dengan analisis berskala besar
terhadap ribuan iklan lowongan. Validasi expert tetap diperlukan;
peran ahli berubah dari *sumber data* menjadi *validator*.

### Pesan kepada Promotor

Beberapa poin yang diharapkan menjadi agenda diskusi:

1. **Validasi KKNI clamp band.** Apakah *band* [floor, floor+1] cukup
   ketat? Bagaimana metode Delphi yang paling efektif untuk validasi—
   ronde tertutup, atau forum terbuka? Berapa panelis yang ideal?

2. **Reframing RQ1.** RQ1 di proposal lama berbunyi "Apakah hibrida
   lebih baik dari komponen?". Pasca-pivot, framing yang diusulkan
   adalah "Apakah LLM-only sudah cukup untuk competency recommendation,
   dan apakah hibrida memberikan *lift* presisi yang material?"
   Apakah promotor setuju dengan reframing ini?

3. **Scope expansion.** Sistem saat ini *locked* pada SE/GD. Apakah
   disertasi harus tetap pada satu bidang demi kedalaman, atau perluas
   ke beberapa bidang demi keluasan kontribusi? Trade-off rigor vs
   *generalizability*.

4. **Strategi diseminasi.** Sistem sudah *deployable* sebagai *public
   service*. Apakah diseminasi sebaiknya dilakukan setelah disertasi
   selesai, atau dapat dimulai paralel sebagai bagian dari Tahap 10
   Borg & Gall? Bagaimana koordinasi dengan BSKAP dan Direktorat SMK?

5. **Posisi *public surface* di disertasi.** *Public-facing competency
   browser* lahir dari pertimbangan praktis, bukan literatur. Apakah ini
   layak menjadi kontribusi terpisah dalam disertasi (BAB IV bagian
   tersendiri), atau cukup disebut sebagai *implementation detail*
   pada bagian metode?

Dokumen ini diharapkan menjadi pijakan diskusi yang produktif. Setiap
keputusan yang dibahas masih dapat direvisi sebelum tahap uji lapangan
operasional dimulai.
