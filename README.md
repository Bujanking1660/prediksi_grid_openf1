# Prediksi Posisi Grid Kualifikasi F1

Memprediksi **posisi grid (`grid_position`) seorang pembalap di kualifikasi F1**
menggunakan data historis [OpenF1](https://openf1.org/) — **hanya data yang tersedia
sebelum kualifikasi dimulai** (latihan bebas FP1–FP3), sehingga tidak ada leakage.

**Hasil terakhir:** model **Ridge mengalahkan baseline** pada metrik rata-rata per weekend.

| Model | MAE ↓ | Spearman ↑ | Kendall ↑ | NDCG@3 ↑ |
|---|---|---|---|---|
| **Ridge** | **3.237** | **0.734** | **0.566** | **0.906** |
| Baseline (aturan sederhana) | 3.272 | 0.705 | 0.553 | 0.883 |
| XGBoost | 3.600 | 0.680 | 0.513 | 0.886 |
| HistGB | 3.847 | 0.634 | 0.469 | 0.872 |
| RandomForest | 3.868 | 0.636 | 0.471 | 0.876 |

Model dan `alpha` Ridge dipilih **di validasi** (2024–2025); **2026 disisihkan sebagai
holdout final** dan tidak dipakai untuk tuning.

---

## 1. Masalah & pendekatan

```text
INPUT (sebelum kualifikasi)              OUTPUT
──────────────────────────────           ─────────────────
Latihan bebas FP1–FP3          ─┐
Riwayat grid pembalap          ─┼──── model ────>  grid_position = 4
(musim berjalan, weekend lalu) ─┘                  (prediksi: grid ke-4)
```

- **Target tunggal:** `grid_position` (regresi). `is_top3` / `is_top10` **bukan model
  terpisah** — hanya turunan dari angka prediksi.
- **Baseline bukan model ML**, melainkan aturan: urutkan pembalap menurut lap FP
  tercepat dalam weekend yang sama. Tugas model: **mengalahkan aturan ini**.
- **Fitur `team_name` tidak dipakai** — performa mobil sudah tercermin di lap FP.

### Kenapa fiturnya berupa "selisih (gap/rank)", bukan angka mentah?

Target itu **relatif dalam 1 weekend**, sedangkan lap mentah **absolut** dan bervariasi
antar sirkuit (Monza ±79 detik, Spa ±106 detik). Model yang belajar angka absolut justru
belajar **mengenali sirkuit**, bukan pembalap tercepat.

Bukti dari data (`fp_best_lap` vs `grid_position`):

| Cara hitung | Korelasi |
|---|---|
| Digabung semua sirkuit (cara lama) | **0.061** — nyaris nol |
| Diukur per weekend (cara baseline) | **0.657** — kuat |

Maka fiturnya dibuat relatif per weekend, ditambah **fitur histori** (rata-rata grid
pembalap di weekend sebelumnya) untuk menangkap skill yang tidak terlihat dari 1 weekend latihan.

---

## 2. Struktur proyek

```text
config.yaml            # tahun, sirkuit, path, parameter model
scripts/fetch_all.py   # fetch semua sirkuit ke cache (resumable)
notebooks/
  00_fetch.ipynb       # pengenalan endpoint + fetch & cache
  01_tables.ipynb      # cache -> driver_quali.csv + quali_lap.csv (tempat fitur dibuat)
  02_model.ipynb       # baca CSV -> tuning + baseline + 4 model + evaluasi
data/
  raw/                 # cache mentah API (Parquet/JSON): sessions, laps, stints,
                       #   weather, drivers, starting_grid
  driver_quali.csv     # tabel model: 1 baris = 1 pembalap per weekend
  quali_lap.csv        # 1 baris = 1 lap kualifikasi
outputs/               # model_metrics, model_summary, weekend_metrics, predictions,
                       #   topk_metrics, feature_importance, tuning_results,
                       #   experiment_config.json, laporan.md
RENCANA.md             # rancangan lengkap proyek
```

**Prinsip:** semua pengolahan (mentah → fitur) ada di `01_tables.ipynb`;
`02_model.ipynb` **hanya membaca CSV** dan memilih kolom mana yang jadi fitur.

---

## 3. Setup

```powershell
python -m venv venvf1
.\venvf1\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

Kalau aktivasi PowerShell diblokir:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venvf1\Scripts\Activate.ps1
```

Jalankan notebook dari terminal (alternatif Run All):

```powershell
$env:PYTHONIOENCODING = "utf-8"
.\venvf1\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_tables.ipynb
.\venvf1\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\02_model.ipynb
```

---

## 4. Pipeline — dari fetch sampai evaluasi

### Tahap 1 — Fetch data (`scripts/fetch_all.py` atau `00_fetch.ipynb`)

Endpoint yang dipakai (hanya prakualifikasi):

| Endpoint | Dipakai untuk |
|---|---|
| `sessions` | daftar sesi FP1–FP3 & Qualifying, tahun, sirkuit |
| `laps` | lap FP (fitur pace) + lap kualifikasi (referensi) |
| `stints` | filter lap valid |
| `weather` | konteks cuaca (tidak dipakai sebagai fitur model) |
| `drivers` | identitas pembalap |
| `starting_grid` | **target** `grid_position` + `quali_lap_duration` |

Endpoint `session_result`, `pit`, `race_control`, `championship_*` **tidak dipakai**
(alasannya dijelaskan di `00_fetch.ipynb`).

```powershell
.\venvf1\Scripts\python.exe scripts\fetch_all.py
```

- **Resumable:** cache per `session_key` tidak diambil ulang — jalankan ulang saja bila terputus.
- Rate limit gratis: **30 request/menit** → fetch penuh ±46 menit (93 weekend, 1389 request).

### Tahap 2 — Bangun tabel (`01_tables.ipynb`)

**Run All.** Mengubah cache mentah menjadi dua CSV:

```text
data/driver_quali.csv   1.974 baris (1.834 berlabel), 91 weekend, 26 sirkuit, 2023–2026
data/quali_lap.csv      19.346 lap kualifikasi
```

Di sinilah **fitur dibuat**:

| Kelompok | Fitur | Cara dibuat |
|---|---|---|
| **Relatif per weekend** | `fp_best_gap` | `fp_best_lap − min(fp_best_lap)` dalam 1 `meeting_key` (0 = tercepat) |
| | `fp_best_rank` | `rank(fp_best_lap)` dalam 1 `meeting_key` (1 = tercepat) = informasi baseline |
| | `fp_median_gap` | selisih pace konsisten (median) ke tercepat |
| **Histori (masa lalu)** | `hist_grid_season` | `expanding().mean().shift(1)` per pembalap |
| | `hist_n` | `expanding().count().shift(1)` — jumlah weekend terlewati |
| **Dibuang** | `fp_best_lap`, `fp_median_clean_lap`, `fp_avg_st_speed` | angka absolut lintas sirkuit (korelasi pool nyaris nol) |
| | `air_temperature`, `track_temperature`, `humidity`, `wind_speed`, `rainfall` | **sama untuk semua pembalap** dalam 1 weekend → tak bisa menjelaskan urutan grid |

**Anti-leakage:**
- `starting_grid.lap_duration` (hasil kualifikasi) hanya referensi, **bukan fitur**.
- Fitur histori dihitung setelah sort chronologis + `shift(1)` → tiap baris hanya memakai
  weekend yang **sudah lewat**. (Diverifikasi: rata-rata histori = rata-rata grid murni masa lalu.)

### Tahap 3 — Modeling (`02_model.ipynb`)

**Run All.** Isi berurutan:

1. **Cek data & label** — jumlah baris, sebaran `grid_position`.
2. **EDA** — korelasi Spearman tiap fitur vs `grid_position`.
3. **Baseline** — ranking `fp_best_lap` per weekend, dievaluasi dengan **fungsi metrik
   dan weekend yang sama** dengan model.
4. **Split — expanding-window time split** (tanpa bocor waktu):

   ```text
   train 2023          → test 2024
   train 2023–2024     → test 2025
   train 2023–2025     → test 2026 (holdout final)
   ```

5. **Tuning di validasi (2024–2025)** — cari `alpha` Ridge terbaik dari
   `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`, lalu bandingkan 4 kandidat memakai alpha itu:
   `Ridge`, `RandomForest`, `HistGradientBoosting`, **`XGBoost`**.
6. **Training & prediksi** — prediksi skor kontinu diubah jadi **peringkat unik 1..N
   per `meeting_key`** (bukan `rint` independen yang bisa bikin posisi duplikat).
7. **Evaluasi** — MAE, Spearman, Kendall's Tau, NDCG@3; dihitung **per weekend** lalu
   dirata-ratakan (tidak di-pool lintas weekend).
8. **Turunan top-3 / top-10** — ROC-AUC dari **skor kontinu** per weekend (bukan model klasifikasi).
9. **Feature importance** — fitur mana yang paling menentukan.
10. **Simpan** ke `outputs/`.

### Tahap 4 — Evaluasi & hasil

```text
outputs/model_metrics.csv       MAE, Spearman, Kendall, NDCG@3 per model per tahun test
outputs/model_summary.csv       rata-rata metrik antar-test year
outputs/weekend_metrics.csv     metrik per weekend (bisa diaudit per sirkuit)
outputs/predictions.csv         pred_score (kontinu) + y_pred (ranking unik) + y_true
outputs/topk_metrics.csv        ROC-AUC & accuracy top-3/top-10 per weekend
outputs/feature_importance.csv  permutation importance
outputs/tuning_results.csv      pencarian alpha Ridge di validasi
outputs/experiment_config.json  fitur, seed, tahun split, alpha & model terpilih
outputs/laporan.md              ringkasan laporan
```

**Rata-rata 3 tahun test (metrik per weekend):**

| Model | MAE ↓ | Spearman ↑ | Kendall ↑ | NDCG@3 ↑ |
|---|---|---|---|---|
| **Ridge** | **3.237** | **0.734** | **0.566** | **0.906** |
| Baseline | 3.272 | 0.705 | 0.553 | 0.883 |
| XGBoost | 3.600 | 0.680 | 0.513 | 0.886 |
| HistGB | 3.847 | 0.634 | 0.469 | 0.872 |
| RandomForest | 3.868 | 0.636 | 0.471 | 0.876 |

**Per tahun test (MAE):**

| Tahun | Baseline | Ridge | XGBoost | Pemenang |
|---|---|---|---|---|
| 2024 | 3.442 | **3.146** | 3.301 | Ridge |
| 2025 | 3.604 | **3.363** | 3.413 | Ridge |
| 2026 (holdout) | **2.770** | 3.201 | 4.085 | Baseline |

**Tuning validasi:** `alpha = 0.001` (MAE validasi 3.254) dan **Ridge** terpilih sebagai
model terbaik (XGBoost 3.357, RandomForest 3.661, HistGB 3.673).

**Feature importance (Ridge):** `fp_best_rank` (1.55) dan `hist_grid_season` (1.02)
mendominasi — dua faktor yang membuat model menang.

**Syarat:** `driver_quali.csv` harus punya ≥ 2 tahun supaya split time-based bisa dibuat.

---

## 5. Batasan (jujur)

- **Kemenangan tidak merata di semua tahun** — pada holdout 2026 baseline masih jauh
  lebih baik (2.770 vs 3.201 Ridge). Rata-rata menang karena 2024–2025.
- **Selisih rata-rata kini sangat tipis** (0.035 MAE, dulu 0.094). Setelah revisi,
  test set tidak lagi dipakai untuk memilih alpha/model, tetapi 2024–2025 berperan
  ganda sebagai validasi sekaligus test — klaim kemenangan sebaiknya dibaca sebagai
  indikasi, bukan bukti kuat.
- **Dua bug evaluasi lama sudah diperbaiki:** (1) `rint` + clip bisa membuat posisi
  duplikat; (2) Spearman di-pool lintas weekend menyesatkan. Sekarang ranking unik
  per weekend dan metrik dirata-ratakan per weekend.
- **Faktor kualifikasi yang tidak terlihat dari FP:** sistem knock-out Q1/Q2/Q3,
  track evolution, tow/slipstream, beban bahan bakar & mode mesin.
- **Model terbaik = Ridge (linear, 5 fitur)** — model berkapasitas tinggi (RandomForest,
  HistGB, XGBoost) justru lebih buruk, menandakan data tidak cukup untuk kompleksitas itu.
- **LightGBM Ranker belum dicoba** — library sudah terpasang, tapi eksperimen ranking
  langsung ditunda sebagai fase berikutnya.

---

## 6. Catatan API OpenF1

- Rate limit gratis: **30 request/menit**.
- Saat sesi F1 **live**, OpenF1 membatasi akses historis dan mengembalikan
  `401 Unauthorized`. Jalankan fetch ulang setelah sesi selesai (fetch resumable).
- Data terbaru menunggu sesi selesai; `grid_position` 2026 bisa berubah seiring musim berjalan.

---

## 7. Dokumen lain

- `RENCANA.md` — rancangan lengkap proyek (endpoint, variabel, tahapan, definition of done).
- `notebooks/00_fetch.ipynb` — pengenalan endpoint (termasuk yang tidak dipakai).
- `notebooks/01_tables.ipynb` — pengenalan variabel & tabel.
- `notebooks/02_model.ipynb` — penjelasan fitur 1b (gap/rank) & 1c (histori), tuning, evaluasi.
# prediksi_grid_openf1
