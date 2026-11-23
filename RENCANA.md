# Rancangan Proyek: Prediksi Kualifikasi F1 dengan OpenF1

> **Status:** draft rancangan, belum dieksekusi  
> **Sumber data:** [OpenF1 API](https://openf1.org/) — data historis 2023–2026  
> **Fokus studi kasus:** Monza (`circuit_key = 39`)  
> **Cut-off prediksi:** sebelum sesi Qualifying dimulai (setelah FP3 selesai)  
> **Target:** posisi grid kualifikasi (`starting_grid.position`)

---

## 1. Ringkasan

Proyek ini membangun **satu model** yang memprediksi posisi grid kualifikasi tiap pembalap:

**Model — Qualifying Position Regression / Classification**

Menggunakan data FP1–FP3 (pace, kecepatan, cuaca), histori kualifikasi, dan identitas pembalap/tim untuk menebak urutan grid sebelum kualifikasi dimulai.

Hasil akhir berupa **leaderboard prediksi grid**:

```text
Driver | Prediksi Grid | P(top3) | P(top10)
```

---

## 2. Pertanyaan yang ingin dijawab

1. Siapa yang paling mungkin dapat pole position?
2. Siapa 3 pembalap terdepan di grid (top-3)?
3. Siapa 10 pembalap terdepan di grid (top-10)?
4. Fitur apa yang paling berpengaruh: pace FP, sejarah quali, atau cuaca?

---

## 3. Keputusan desain

| Topik | Keputusan | Alasan |
|---|---|---|
| Sumber data | OpenF1 API historis | Gratis, tanpa API key |
| Rentang data | 2023–2026 | Data historis OpenF1 mulai 2023 |
| Sirkuut studi kasus | Monza `circuit_key=39` | 4 weekend konsisten, tanpa sprint |
| Sesi yang dipakai | FP1, FP2, FP3, Qualifying | Data sebelum cut-off = FP1–FP3; target = Qualifying |
| Target | `starting_grid.position` | Posisi grid dari hasil kualifikasi |
| Training | Semua sirkuit 2023–2026 | Menambah jumlah sampel |
| Evaluasi | Monza 2024–2026 | Fokus studi kasus, time-based |
| Cut-off | Sebelum Qualifying dimulai | Semua fitur dari FP1–FP3 atau histori |
| Model | Regresi posisi + klasifikasi top-3/top-10 | Target posisi kontinu & klasifikasi biner |
| Validasi | Expanding-window time split | Menghindari leakage temporal |
| Format cache | Parquet | Ringan, cepat, kolom bertipe jelas |

---

## 4. Data

### 4.1 Endpoint yang dipakai

| Endpoint | Kegunaan | Level |
|---|---|---|
| `sessions` | Daftar sesi, tanggal, tipe, `session_key` | Weekend |
| `meetings` | Info GP, sirkuit, tanggal | Weekend |
| `drivers` | Identitas pembalap & tim | Sesi |
| `laps` | Waktu lap FP1–FP3 & Qualifying | Lap |
| `starting_grid` | Hasil kualifikasi / posisi grid (target) | Qualifying |
| `weather` | Suhu trek/udara, kelembapan, angin, hujan | Sesi |
| `stints` | Compound, lap start/end (FP) | Sesi |

### 4.2 Endpoint yang **tidak** dipakai (dihapus)

| Endpoint | Alasan dihapus |
|---|---|
| `session_result` | Hasil finish race — bukan target qualifying |
| `pit` | Pit stop race — terjadi setelah qualifying |
| `race_control` | Insiden race — tidak relevan untuk quali |
| `championship_drivers` | Klasemen race — bukan fitur pre-quali |
| `championship_teams` | Klasemen tim — bukan fitur pre-quali |
| `car_data` | Terlalu berat, tidak diperlukan pre-quali |
| `location` | Telemetry lokasi, tidak diperlukan |
| `team_radio` | Cakupan tidak reliabel |
| `overtakes` | Insiden race, bukan pre-quali |

### 4.3 Cakupan & volume

| Item | Jumlah |
|---|---:|
| Weekend Monza 2023–2026 | 4 weekend |
| Sesi FP + Quali per weekend | 4 sesi |
| Baris `driver-race` (20 pembalap × 4) | ±80 |
| Baris `laps` (FP + Quali) | ±10 ribu per weekend |

---

## 5. Definisi target & cut-off

### 5.1 Garis waktu prediksi

```text
FP1 ── FP2 ── FP3 ── [CUT-OFF] ── Qualifying
                       │
                       └── semua fitur harus berasal dari titik ini atau sebelumnya
```

**Aturan utama:** tidak ada data Qualifying atau Race yang boleh masuk sebagai fitur. Hanya FP1–FP3 + histori sebelum cut-off.

### 5.2 Target

| Target | Tipe | Definisi |
|---|---|---|
| `grid_position` | Regresi | `starting_grid.position` |
| `is_top3` | Klasifikasi biner | `1` jika `position <= 3` |
| `is_top10` | Klasifikasi biner | `1` jika `position <= 10` |

---

## 6. Arsitektur model

```text
┌─────────────────────────────────────────────────────────────┐
│ Data sebelum qualifying: FP1–FP3, weather, histori,         │
│ driver & tim info                                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │  Model              │
                  │  Grid Position      │
                  │  Regression         │
                  └──────────┬──────────┘
                             │ prediksi posisi grid
                             ▼
                  ┌─────────────────────┐
                  │  Leaderboard        │
                  │  Driver | Grid |    │
                  │  Pred | P3 | P10   │
                  └─────────────────────┘
```

### Fitur utama

- Best lap FP1, FP2, FP3
- Median clean lap FP1–FP3
- Rata-rata speed trap (st_speed) FP
- Gap ke tercepat di FP
- `starting_grid` histori (grid & posisi finish tahun lalu di Monza)
- Cuaca FP (track_temperature, air_temperature, humidity, wind_speed, rainfall)
- `team_name`, `driver_number`

### Output

```text
driver_number | predicted_grid_position | p_top3 | p_top10
```

**Metrik:** MAE posisi grid, Spearman rank correlation, ROC-AUC untuk top-3/top-10.

**Baseline:** “Grid = finish race sebelumnya” atau “Grid = grid FP3 terbaik”.

---

## 7. Feature engineering

### 7.1 Tabel `driver_quali`

Satu baris per pembalap per qualifying session.

| Kolom | Sumber | Keterangan |
|---|---|---|
| `year` | `sessions` | Musim |
| `meeting_key` | `sessions` | ID weekend |
| `session_key_quali` | `sessions` | ID sesi Qualifying |
| `circuit_key` | `sessions` | 39 untuk Monza |
| `driver_number` | `drivers` | Nomor pembalap |
| `team_name` | `drivers` | Tim |
| `fp1_best_lap` | `laps` FP1 | Best lap FP1 |
| `fp2_best_lap` | `laps` FP2 | Best lap FP2 |
| `fp3_best_lap` | `laps` FP3 | Best lap FP3 |
| `fp_median_clean_lap` | `laps` FP | Median lap bersih semua FP |
| `fp_avg_st_speed` | `laps` FP | Rata-rata speed trap |
| `fp_gap_to_fastest` | turunan | Selisih ke lap tercepat FP |
| `quali_hist_grid` | histori | Grid Monza tahun sebelumnya |
| `quali_hist_finish` | histori | Finish Monza tahun sebelumnya |
| `track_temperature` | `weather` FP | Cuaca FP |
| `air_temperature` | `weather` FP | Cuaca FP |
| `humidity` | `weather` FP | Cuaca FP |
| `wind_speed` | `weather` FP | Cuaca FP |
| `rainfall` | `weather` FP | Cuaca FP |
| `grid_position` | `starting_grid` | **Target** |
| `is_top3` | turunan | **Target** |
| `is_top10` | turunan | **Target** |

### 7.2 Tabel `quali_lap`

Satu baris per lap Qualifying (untuk analisis pacing kualifikasi).

| Kolom | Sumber | Keterangan |
|---|---|---|
| `session_key` | `laps` | ID sesi Qualifying |
| `driver_number` | `laps` | Pembalap |
| `lap_number` | `laps` | Nomor lap |
| `lap_duration` | `laps` | Waktu lap |
| `i1_speed`, `i2_speed`, `st_speed` | `laps` | Kecepatan |
| `is_pit_out_lap` | `laps` | Flag pit out |

### 7.3 Aturan anti-leakage

| Data | Boleh sebagai fitur? | Catatan |
|---|---|---|
| FP1–FP3 | ✅ | Semua sebelum cut-off |
| Qualifying laps | ❌ | Merupakan target itu sendiri |
| `starting_grid` | ❌ | Hanya sebagai target |
| Race data | ❌ | Terjadi setelah qualifying |
| `championship_*` | ❌ | Berdasarkan hasil race |
| `pit` / `race_control` | ❌ | Terjadi selama/ setelah race |

---

## 8. Pipeline

### Stage 1 — Fetch & cache

1. Ambil semua sesi 2023–2026 (FP1, FP2, FP3, Qualifying).
2. Filter sirkuit Monza (`circuit_key=39`).
3. Simpan metadata sesi ke `data/raw/sessions/circuit_39.parquet`.
4. Untuk tiap sesi: `laps`, `stints`, `weather`, `drivers`.
5. Untuk sesi Qualifying: `starting_grid` (diambil dari `session_key` Qualifying).

### Stage 2 — Bangun `quali_lap` & `driver_quali`

- `quali_lap`: lap Qualifying yang bersih.
- `driver_quali`: satu baris per pembalap per qualifying, fitur dari FP + target grid.

### Stage 3 — Train model

- Split time-based (2024 test pada 2023, 2025 test pada 2023–2024, dst.).
- Train regresi posisi grid + klasifikasi top-3/top-10.
- Evaluasi MAE, Spearman, ROC-AUC.
- Bandingkan dengan baseline.

### Stage 4 — Laporan

- Leaderboard grid Monza 2024–2026.
- Prediksi vs aktual grid position.
- Feature importance.

---

## 9. Struktur repository

```
prediksi-openf1/
├── RENCANA.md
├── README.md
├── config.yaml
├── requirements.txt
├── data/
│   ├── raw/               # respons API mentah per endpoint/session
│   ├── driver_quali.csv   # satu baris per pembalap per qualifying
│   └── quali_lap.csv      # lap qualifying
├── notebooks/
│   ├── 00_fetch.ipynb     # API -> data/raw/
│   └── 01_tables.ipynb    # data/raw/ -> tabel quali
└── outputs/
    └── laporan.md
```

---

## 10. Milestone

| Tahap | Deliverable | Estimasi |
|---|---|---|
| 1 | Fetch & cache 2023–2026 | 0,5–1 hari |
| 2 | `quali_lap` & `driver_quali` | 0,5 hari |
| 3 | Model grid position + evaluasi | 1 hari |
| 4 | Laporan & visualisasi | 0,5 hari |
| 5 | Refactor & README | 0,5 hari |

Total: **±2–3 hari**.

---

## 11. Definition of done

- [x] Pipeline bisa dijalankan ulang dari cache tanpa fetch ulang.
- [x] Tidak ada leakage (data Qualifying/Race tidak masuk sebagai fitur).
- [x] Model mengalahkan baseline grid sederhana. (Ridge: MAE rata-rata per weekend **3.237** vs baseline 3.272; Spearman **0.734** vs 0.705. Alpha & model dipilih di validasi; 2026 holdout.)
- [x] Leaderboard grid per weekend (termasuk Monza 2024–2026) tersedia di `outputs/predictions.csv`.
- [x] Feature importance tersedia.
- [x] Laporan menjelaskan keterbatasan.
- [x] README berisi cara menjalankan.
