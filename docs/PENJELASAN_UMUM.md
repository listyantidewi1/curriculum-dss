# Pengenalan Sistem Rekomendasi Kompetensi

> Dokumen ini disusun untuk pembaca umum dari berbagai latar belakang —
> siswa, guru, kepala sekolah, orang tua, hingga pemangku kebijakan
> pendidikan. Tidak diperlukan pengetahuan teknis untuk memahami isinya.

---

## 1. Latar Belakang Permasalahan

Setiap tahun, ribuan lulusan Sekolah Menengah Kejuruan (SMK) di Indonesia
memasuki dunia kerja. Namun, banyak di antara mereka mengalami kesulitan
saat melamar pekerjaan. Pemberi kerja kerap menyampaikan kekhawatiran
seperti berikut:

```
   ┌──────────────────────────────────────────────┐
   │  "Pelamar belum menguasai perangkat ini..."  │
   │  "Kami membutuhkan kemampuan komputasi awan" │
   │  "Kemampuan kerja sama tim masih terbatas"   │
   └──────────────────────────────────────────────┘
```

Kondisi ini menunjukkan adanya **kesenjangan** antara kompetensi yang
dibekalkan oleh sekolah dengan kompetensi yang dibutuhkan oleh dunia
industri. Permasalahan ini bukan kesalahan satu pihak. Industri perangkat
lunak berkembang sangat pesat, sementara peninjauan kurikulum sekolah
umumnya dilakukan secara berkala dengan rentang waktu yang panjang.
Akibatnya, ketika kurikulum selesai diperbarui, kebutuhan industri
seringkali sudah berubah kembali.

Dampak nyata dari kondisi tersebut tercermin dalam data
ketenagakerjaan nasional:

```
                    Lulusan SMK     Lulusan SMA
                    ───────────    ───────────
   Pengangguran       9,01%           7,05%
                       ↑↑↑
   Tingkat pengangguran lulusan SMK justru
   lebih tinggi daripada lulusan SMA.
```

*Sumber: Badan Pusat Statistik, Agustus 2024.*

Padahal, pendidikan vokasi seharusnya memberikan keunggulan komparatif
bagi lulusannya untuk lebih siap memasuki dunia kerja. Diperlukan suatu
mekanisme yang memungkinkan kurikulum sekolah berjalan beriringan dengan
dinamika kebutuhan industri secara aktual.

---

## 2. Solusi yang Ditawarkan

Penelitian ini mengembangkan **sistem otomatis** yang membaca ribuan
iklan lowongan kerja, kemudian menyusun **rekomendasi kompetensi** yang
sebaiknya tercakup dalam kurikulum sekolah.

Sistem ini berperan sebagai alat bantu bagi pengembang kurikulum dengan
karakteristik berikut:

- Beroperasi secara berkelanjutan tanpa terikat jam kerja konvensional;
- Memproses data lowongan kerja dalam jumlah besar yang sulit ditangani
  secara manual;
- Mengidentifikasi tren kebutuhan kompetensi secara objektif berdasarkan
  data; dan
- Memberikan rekomendasi yang dapat ditelusuri sumbernya.

```
   ┌─────────────────┐
   │  Iklan Kerja    │ ──┐
   │  (ribuan)       │   │
   └─────────────────┘   │
                         ▼
                  ┌─────────────────┐
                  │  Sistem Analisis│
                  │  Kompetensi     │
                  └─────────────────┘
                         │
                         ▼
   ┌─────────────────────────────────────────┐
   │  Daftar Kompetensi yang Direkomendasikan│
   │  untuk masuk ke kurikulum sekolah       │
   └─────────────────────────────────────────┘
```

---

## 3. Bagaimana Sistem Bekerja

Berikut adalah perjalanan data dari iklan lowongan kerja hingga menjadi
rekomendasi kompetensi yang siap digunakan oleh sekolah.

### Tahap 1 — Pengumpulan Data Lowongan Kerja

Sistem mengumpulkan iklan lowongan kerja dari berbagai portal karier
daring. Proses ini berlangsung secara berkala sehingga data yang diperoleh
selalu mencerminkan kondisi terkini.

```
   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   │iklan│ │iklan│ │iklan│ │iklan│ │iklan│ │iklan│  ...ribuan...
   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

### Tahap 2 — Pembacaan dan Penyaringan Informasi

Setiap iklan lowongan kerja dibaca dan dianalisis secara terstruktur.
Sistem mengekstrak tiga jenis informasi utama dari setiap iklan:

```
   ╔════════════════════════════════════════════════════╗
   ║  KETERAMPILAN TEKNIS (Hard Skills)                 ║
   ║      Contoh: penguasaan bahasa pemrograman Python, ║
   ║      pengelolaan basis data SQL, dan sebagainya    ║
   ║                                                    ║
   ║  KETERAMPILAN NON-TEKNIS (Soft Skills)             ║
   ║      Contoh: kemampuan komunikasi, kerja sama tim, ║
   ║      manajemen waktu, dan sebagainya               ║
   ║                                                    ║
   ║  PERSYARATAN PENDIDIKAN                            ║
   ║      Contoh: minimal lulusan D3, Sarjana bidang    ║
   ║      Informatika, dan sebagainya                   ║
   ╚════════════════════════════════════════════════════╝
```

### Tahap 3 — Analisis Frekuensi Kebutuhan

Sistem menghitung seberapa sering setiap kompetensi muncul dalam
keseluruhan iklan. Apabila suatu kompetensi muncul pada sebagian besar
iklan, hal tersebut menunjukkan tingginya permintaan industri. Sebaliknya,
kompetensi yang jarang muncul mengindikasikan menurunnya relevansi.

```
   Python              ████████████████████████████████  800 iklan
   JavaScript          ███████████████████████████       650 iklan
   Komputasi Awan      ████████████████████              520 iklan
   Keamanan Siber      █████████████████                 410 iklan
   Visual Basic 6      █                                   2 iklan
                       └────────────────────────────────┘
                       0                                 1000
```

### Tahap 4 — Analisis Tren dari Waktu ke Waktu

Selain frekuensi, sistem juga menganalisis perubahan kebutuhan kompetensi
dari bulan ke bulan. Hal ini memungkinkan identifikasi kompetensi yang
sedang berkembang, stabil, atau menurun.

```
   Kecerdasan Buatan            Bahasa Pemrograman PHP Klasik
   ───────────────────────      ───────────────────────
       ▁▂▃▄▅▆▇█  Meningkat          █▇▆▅▄▃▂▁  Menurun

   Direkomendasikan untuk        Perlu pertimbangan ulang
   masuk kurikulum               sebelum diajarkan
```

### Tahap 5 — Penyusunan Kompetensi

Daftar keterampilan mentah yang sangat banyak tidak dapat langsung
diterjemahkan menjadi mata pelajaran. Sistem mengelompokkan keterampilan
yang saling terkait menjadi satu kompetensi yang lebih komprehensif.

```
   ┌─────────────────────────────────────────────────┐
   │  Daftar keterampilan yang serumpun:             │
   │   • Docker                                      │
   │   • Kubernetes                                  │
   │   • Komputasi Awan (AWS)                        │
   │   • Penerapan Aplikasi Berbasis Awan            │
   │   • Pipeline Integrasi Berkelanjutan            │
   │                                                 │
   │            ↓  Dirumuskan menjadi  ↓             │
   │                                                 │
   │  KOMPETENSI:                                    │
   │     "Mengembangkan dan menerapkan aplikasi      │
   │      berbasis komputasi awan secara aman dan    │
   │      efisien"                                   │
   └─────────────────────────────────────────────────┘
```

Setiap kompetensi yang dihasilkan dilengkapi dengan informasi sebagai
berikut:

- Judul kompetensi yang siap digunakan dalam dokumen kurikulum;
- Penjelasan ringkas dalam satu kalimat;
- Daftar keterampilan teknis yang relevan;
- Daftar keterampilan non-teknis yang dibutuhkan; dan
- Tingkatan kualifikasi sesuai KKNI (akan dijelaskan pada bagian
  berikutnya).

### Tahap 6 — Penentuan Prioritas

Tidak seluruh kompetensi memiliki tingkat urgensi yang sama. Sistem
menghitung skor prioritas untuk setiap kompetensi berdasarkan tiga faktor:

```
   Prioritas =  40% Tingkat Permintaan  (banyaknya iklan
                                          yang membutuhkan)
              + 30% Tren                 (sedang meningkat
                                          atau menurun)
              + 30% Relevansi Masa Depan (proyeksi
                                          pertumbuhan bidang)
```

Hasil perhitungan tersebut kemudian digunakan untuk menyusun urutan
prioritas:

```
   #1  Komputasi Awan                     skor 88,3  ★★★
   #2  Pengembangan Antarmuka API         skor 85,1  ★★★
   #3  Pipeline Integrasi Berkelanjutan   skor 84,7  ★★★
   #4  Arsitektur Microservices           skor 80,6  ★★
   #5  Basis Data NoSQL                   skor 76,9  ★★
   ...
```

---

## 4. Tentang Kerangka Kualifikasi Nasional Indonesia (KKNI)

KKNI merupakan kerangka acuan resmi yang ditetapkan Pemerintah Republik
Indonesia melalui **Peraturan Presiden Nomor 8 Tahun 2012**. Kerangka ini
membagi tingkat kualifikasi keahlian di Indonesia menjadi sembilan
jenjang, dimulai dari jenjang paling dasar hingga jenjang paling tinggi.

```
   Jenjang 9  ████████████████████  S3 (Doktor)
   Jenjang 8  ██████████████████    S2 (Magister)
   Jenjang 7  ████████████████      Pendidikan Profesi
   Jenjang 6  ██████████████        S1 / D4 (Sarjana)
   Jenjang 5  ████████████          D3
   Jenjang 4  ██████████            D2
   Jenjang 3  ████████              SMK / D1
   Jenjang 2  ██████                SMA
   Jenjang 1  ████                  SMP
              └──────────────┘
              Semakin tinggi jenjang, semakin kompleks
              kompetensi yang diharapkan
```

Pengelompokan ini penting karena setiap jenjang pendidikan memiliki
ekspektasi kompetensi yang berbeda. Lulusan SMK tidak ditargetkan untuk
menguasai kompetensi yang sama dengan lulusan S1, demikian pula
sebaliknya. Sistem yang dikembangkan **menyematkan jenjang KKNI** pada
setiap kompetensi yang direkomendasikan, sehingga pengguna dapat memilih
kompetensi yang sesuai dengan jenjang pendidikan yang dimaksud.

Sebagai contoh, seorang pengembang kurikulum SMK dapat memfilter sistem
agar hanya menampilkan kompetensi yang relevan dengan KKNI Jenjang 2 dan
3, sesuai dengan posisi SMK dalam kerangka KKNI.

---

## 5. Perbandingan dengan Pendekatan Konvensional

### Pendekatan Konvensional

```
   ┌──────────────────────────────────────────────────┐
   │  • Tim pengembang kurikulum mengadakan rapat     │
   │  • Konsultasi dengan rekan dari industri         │
   │  • Memperkirakan tren berdasarkan pengalaman     │
   │  • Menyusun kurikulum                            │
   │  • Peninjauan dilakukan dua tahun kemudian       │
   │  • Saat itu, tren industri sudah berubah         │
   └──────────────────────────────────────────────────┘
```

### Pendekatan Berbasis Sistem Rekomendasi

```
   ┌──────────────────────────────────────────────────┐
   │  • Pengguna mengakses laman daring sistem        │
   │  • Memilih jenjang pendidikan yang relevan       │
   │  • Mendapatkan daftar kompetensi terurut         │
   │    berdasarkan prioritas                         │
   │  • Mengunggah kurikulum yang sedang berlaku      │
   │  • Memperoleh laporan: "60% kompetensi sudah     │
   │    tercakup, 40% memerlukan penambahan"          │
   │  • Daftar kekurangan dijadikan dasar pembaruan   │
   │    kurikulum                                     │
   └──────────────────────────────────────────────────┘
```

---

## 6. Pengguna yang Dapat Memanfaatkan Sistem

Sistem ini terbuka untuk masyarakat umum tanpa biaya. Berikut adalah
gambaran pemanfaatan sistem oleh berbagai kelompok pengguna:

```
   ┌─────────────────┐  →  Mengetahui kompetensi yang relevan
   │  Siswa          │     dengan jurusan yang dipilih
   └─────────────────┘

   ┌─────────────────┐  →  Mengevaluasi kurikulum mata pelajaran
   │  Guru           │     yang sedang diampu
   └─────────────────┘

   ┌─────────────────┐  →  Menyusun KOSP berbasis data pasar kerja
   │  Kepala Program │
   │  Keahlian       │
   └─────────────────┘

   ┌─────────────────┐  →  Melakukan kajian akademik dan analisis
   │  Dosen/Peneliti │     dinamika ketenagakerjaan
   └─────────────────┘

   ┌─────────────────┐  →  Memperoleh gambaran profil lulusan ideal
   │  Praktisi/HR    │     untuk perekrutan
   └─────────────────┘
```

Untuk fitur dasar (penjelajahan kompetensi dan analisis kurikulum
sekali pakai), pengguna tidak perlu mendaftar. Apabila pengguna ingin
**menyimpan riwayat analisis** kurikulum untuk ditinjau kembali, sistem
menyediakan fasilitas pendaftaran akun ringan yang hanya membutuhkan
alamat surel dan kata sandi.

---

## 7. Fitur-Fitur Utama Sistem

Sistem menyediakan empat halaman utama yang dapat diakses oleh pengguna:

```
   ╔═══════════════════════════════════════════════════════╗
   ║                                                       ║
   ║   BERANDA                                             ║
   ║      Memuat pengantar singkat dan tautan cepat menuju ║
   ║      seluruh fitur sistem.                            ║
   ║                                                       ║
   ║   PENJELAJAHAN KOMPETENSI                             ║
   ║      Pengguna dapat menyaring berdasarkan jenjang     ║
   ║      pendidikan, bidang masa depan, dan kata kunci.   ║
   ║      Setiap kartu kompetensi dapat diklik untuk       ║
   ║      melihat detail lengkap.                          ║
   ║                                                       ║
   ║   ANALISIS KURIKULUM                                  ║
   ║      Pengguna mengunggah berkas kurikulum (CSV/JSON). ║
   ║      Sistem menghasilkan laporan persentase cakupan,  ║
   ║      daftar kompetensi yang sudah dan belum tercakup, ║
   ║      serta keterampilan prioritas yang masih kurang.  ║
   ║                                                       ║
   ║   TENTANG SISTEM                                      ║
   ║      Berisi penjelasan metodologi, referensi KKNI     ║
   ║      lengkap, dan catatan tentang sumber data.        ║
   ║                                                       ║
   ╚═══════════════════════════════════════════════════════╝
```

---

## 8. Ilustrasi Penggunaan: Studi Kasus

Untuk memberikan gambaran yang lebih konkret, berikut adalah ilustrasi
penggunaan sistem oleh seorang Ketua Program Keahlian Rekayasa Perangkat
Lunak di sebuah SMK. Yang bersangkutan ditugaskan untuk menyusun KOSP
(Kurikulum Operasional Satuan Pendidikan) untuk tahun ajaran berikutnya.

```
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TAHAP 1   Pengguna mengakses laman utama sistem
             dan memilih menu "Penjelajahan Kompetensi".
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TAHAP 2   Pengguna memilih penyaring "SMK" pada
             bagian atas halaman. Sistem secara otomatis
             menampilkan kompetensi yang sesuai untuk
             jenjang SMK (KKNI Jenjang 2 dan 3).
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ┌─────────────────────────────────────────────────────┐
   │  [ KKNI 3 ]  [ Pengembangan Aplikasi Web ]          │
   │                                                     │
   │  Membangun dan memelihara aplikasi web sederhana    │
   │  menggunakan HTML, CSS, JavaScript, dan basis data. │
   │                                                     │
   │  Keterampilan non-teknis yang dibutuhkan:           │
   │  Komunikasi, Kerja Sama Tim, Ketelitian             │
   └─────────────────────────────────────────────────────┘

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TAHAP 3   Pengguna mengeklik kartu kompetensi untuk
             memperoleh informasi lebih rinci, meliputi:
             daftar keterampilan teknis terkait,
             keterampilan non-teknis yang dibutuhkan,
             jumlah lowongan kerja yang membutuhkannya,
             dan kecenderungan tren.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TAHAP 4   Pengguna berpindah ke menu "Analisis
             Kurikulum" dan mengunggah berkas kurikulum
             yang sedang berlaku di sekolah.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                     ┌─────────────────┐
                     │  RPL_2025.csv   │
                     └────────┬────────┘
                              │ unggah
                              ▼
                  ┌──────────────────────┐
                  │  Sedang dianalisis   │
                  └──────────────────────┘

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TAHAP 5   Sistem menyajikan hasil analisis sebagai
             berikut:
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   ┌─────────────────────────────────────────────────────┐
   │           CAKUPAN KURIKULUM SMK                     │
   │                                                     │
   │                    62%                              │
   │           ████████████████░░░░░░░░░                 │
   │                                                     │
   │   31 dari 50 kompetensi telah tercakup              │
   │   19 kompetensi belum tercakup                      │
   │                                                     │
   │   Kompetensi prioritas yang direkomendasikan:       │
   │      1. Pengembangan Antarmuka API                  │
   │      2. Dasar-Dasar Keamanan Aplikasi Web           │
   │      3. Pengelolaan Versi Kode (Git)                │
   │      4. Pengujian Otomatis                          │
   │      5. Dasar Komputasi Awan                        │
   └─────────────────────────────────────────────────────┘

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TAHAP 6   Pengguna memperoleh daftar konkret yang
             dapat dijadikan bahan diskusi dalam rapat
             penyusunan KOSP berikutnya, dilengkapi
             dengan justifikasi berbasis data.
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 9. Hal-Hal yang Perlu Diperhatikan

Sistem ini dirancang sebagai **alat bantu**, bukan sebagai pengganti
peran guru atau ahli kurikulum. Pertimbangan profesional, pengalaman
lapangan, dan kearifan lokal tetap menjadi faktor utama dalam pengambilan
keputusan kurikulum.

```
   ┌────────────────────────────────────────────────────┐
   │  PEMANFAATAN YANG DISARANKAN:                      │
   │                                                    │
   │     • Bahan diskusi dalam rapat pengembangan       │
   │       kurikulum                                    │
   │     • Daftar awal kompetensi yang perlu ditelaah   │
   │     • Validasi terhadap intuisi profesional        │
   │     • Pemantauan dinamika pasar tenaga kerja       │
   │                                                    │
   │  PEMANFAATAN YANG TIDAK DISARANKAN:                │
   │                                                    │
   │     • Penggantian sepenuhnya keputusan ahli        │
   │       kurikulum                                    │
   │     • Sumber kebenaran tunggal tanpa verifikasi    │
   │     • Pengukuran kemampuan siswa secara individu   │
   │     • Penentuan jurusan tanpa konsultasi dengan    │
   │       pihak terkait                                │
   │                                                    │
   └────────────────────────────────────────────────────┘
```

Beberapa keterbatasan sistem yang perlu disampaikan secara transparan:

- Sumber data lowongan kerja yang digunakan saat ini sebagian besar
  berbahasa Inggris. Pengembangan untuk data berbahasa Indonesia akan
  dilaksanakan pada tahap selanjutnya. Untuk sementara, sistem
  berfungsi sebagai jembatan yang menghubungkan tren industri global
  dengan konteks pendidikan di Indonesia.
- Cakupan bidang yang ditangani sistem masih terbatas pada Software
  Engineering dan Game Development. Pengembangan untuk bidang lain
  (akuntansi, pariwisata, kuliner, dan sebagainya) dapat dilakukan
  dengan pendekatan serupa pada penelitian lanjutan.
- Fitur analisis kurikulum menggunakan metode pencocokan kata kunci.
  Penggunaan singkatan pada kurikulum (misalnya "OOP") dapat tidak
  terdeteksi apabila kompetensi sistem ditulis dengan istilah lengkap
  ("Object-Oriented Programming"). Disarankan untuk menggunakan
  istilah yang konsisten dan lengkap.

---

## 10. Penutup

Penelitian ini didasarkan pada keyakinan bahwa lulusan pendidikan vokasi
Indonesia berhak mendapatkan kurikulum yang relevan dengan kebutuhan
industri secara aktual. Dengan pemanfaatan teknologi analisis data,
diharapkan kesenjangan antara dunia pendidikan dan dunia kerja dapat
diperkecil secara berkelanjutan.

```
   ╔═══════════════════════════════════════════════════════╗
   ║                                                       ║
   ║   Tujuan jangka panjang penelitian ini adalah:        ║
   ║                                                       ║
   ║   • Lulusan SMK memiliki kompetensi yang sesuai       ║
   ║     dengan kebutuhan industri.                        ║
   ║                                                       ║
   ║   • Tingkat kesenjangan kompetensi (mismatch)         ║
   ║     menurun secara signifikan.                        ║
   ║                                                       ║
   ║   • Kurikulum sekolah dapat diperbarui secara         ║
   ║     berkelanjutan berbasis data.                      ║
   ║                                                       ║
   ╚═══════════════════════════════════════════════════════╝
```

Pertanyaan, masukan, atau permohonan kerja sama dapat disampaikan melalui
laman **Tentang Sistem** pada situs web, atau melalui surel resmi
peneliti.

Terima kasih atas perhatian Anda.

---

*Dokumen ini disusun untuk pembaca umum. Apabila terdapat bagian yang
kurang jelas, kami mohon maaf atas keterbatasan penyajian, dan kami
sangat menghargai masukan untuk perbaikan dokumen pada penyusunan
berikutnya.*
