# Laporan Modeling — Prediksi Grid Kualifikasi

- Jumlah baris data: 1834
- Tahun: [2023, 2024, 2025, 2026]
- Weekend unik: 90
- Fitur: fp_best_gap, fp_best_rank, fp_median_gap, hist_grid_season, hist_n
- Target: grid_position
- Alpha Ridge terpilih (validasi): 0.001
- Model terpilih (validasi): Ridge
- Holdout final: 2026
- Evaluasi: metrik per weekend lalu dirata-ratakan; ranking unik per meeting_key

## Rata-rata metrik per weekend

| Model | MAE | Spearman | Kendall | NDCG@3 |
|---|---|---|---|---|
| Ridge | 3.237 | 0.734 | 0.566 | 0.906 |
| Baseline | 3.272 | 0.705 | 0.553 | 0.883 |
| XGBoost | 3.600 | 0.680 | 0.513 | 0.886 |
| HistGB | 3.847 | 0.634 | 0.469 | 0.872 |
| RandomForest | 3.868 | 0.636 | 0.471 | 0.876 |

MAE terbaik: **Ridge** (3.237 vs baseline 3.272)
Model terpilih berdasarkan VALIDASI: **Ridge**