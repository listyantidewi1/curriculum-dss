# Spektrum Keahlian SMK/MAK (Kepmen 244/M/2024)

Reference taxonomy for Indonesian vocational high school (SMK/MAK) competencies.

## Source

- **Legal basis:** Keputusan Menteri Pendidikan, Kebudayaan, Riset, dan Teknologi Republik Indonesia Nomor **244/M/2024** tentang Spektrum Keahlian dan Konversi Spektrum Keahlian Sekolah Menengah Kejuruan/Madrasah Aliyah Kejuruan pada Kurikulum Merdeka
- **Effective date:** 10 Juni 2024
- **Official link:** [kurikulum.kemendikdasmen.go.id](https://kurikulum.kemendikdasmen.go.id/file/1718354948_manage_file.pdf)
- **JDIH:** [jdih.kemdikbud.go.id/detail_peraturan?main=3402](https://jdih.kemdikbud.go.id/detail_peraturan?main=3402)

## Structure

Spektrum has three levels:

1. **Bidang Keahlian** (field of expertise) — 10 fields, e.g. Teknologi Informasi, Bisnis dan Manajemen
2. **Program Keahlian** (program of expertise) — e.g. Rekayasa Perangkat Lunak
3. **Konsentrasi Keahlian** (concentration) — most specific, e.g. 4.1.1 Rekayasa Perangkat Lunak

## Files

| File | Purpose |
|------|---------|
| `spektrum_keahlian.json` | Full hierarchy: Bidang → Program → Konsentrasi with codes |
| `spektrum_mapping.csv` | Mapping from Spektrum code to `future_domains.csv` domain_ids |

## Extraction Method

- **spektrum_keahlian.json:** Manually extracted from Lampiran I of Kepmen 244/M/2024
- **spektrum_mapping.csv:** Manually defined; maps IT programs (4.x) to WEF/ONET domains; non-IT to ESCO/MCK; `*` = fallback to all domains

## Version

- **Extraction date:** 2025-02-25
- **Data version:** 1.0 (initial)

## Notes

- Full Lampiran I/II from the PDF should be consulted for authoritative codes; this JSON covers major programs, especially Teknologi Informasi (Bidang 4)
- Mapping to `future_domains.csv` is IT-biased; non-IT Bidang (Agribisnis, Pariwisata, etc.) may need domain expansion
