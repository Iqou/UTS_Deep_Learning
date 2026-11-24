# Tugas 2 - Prediksi sisa umur mesin

Tugas ini memprediksi remaining useful life (RUL) pada siklus terakhir setiap mesin uji, dalam rentang 0–150 siklus. Metrik penilaian adalah RMSE dan skor asimetris.

## Yang dikerjakan notebook

- Menghitung label dari siklus gagal dan membuat split terpisah untuk mesin latih, dev, kalibrasi, dan evaluasi.
- Menganalisis pergeseran sensor antarrezim, kanal dengan korelasi lemah, pembacaan hilang, serta spike. Statistik normalisasi dipelajari dari mesin latih, dan isian hilang hanya memakai sejarah mesin yang tersedia.
- Membentuk jendela hingga 40 siklus dengan padding dan mask. Histogram gradient boosting menjadi baseline, lalu dibandingkan dengan GRU dan CNN 1D.
- Memilih ensemble neural pada split dev, menghitung RMSE dan skor asimetris per rentang, lalu membangun interval split conformal pada kelompok kalibrasi.

## Menjalankan

Buka [Prediksi_RUL.ipynb](Prediksi_RUL.ipynb). Sel 1 membaca `rul_train.csv.gz` dan `rul_test.csv.gz`. Jalankan Sel 1–4, lalu panggil `jalankan_tugas2()` di Sel 5. Sel 6 mengekspor hasil.

Checkpoint CPU setiap seed disimpan di `model/` untuk checkout yang memiliki konfigurasi cocok. Notebook bisa melatih ulang jaringan dari riwayat mesin. Konfigurasi referensi memakai CPU tiga thread.

## Keluaran

- [prediksi_rul.csv](../../prediksi_rul.csv) memuat kolom `id_unit,rul` dengan satu baris per mesin uji.
- [Bagian laporan 02](../../laporan_bagian/02_prediksi_rul.pdf) membahas preprocessing, pembandingan model, galat per rentang, dan interval.
- Bukti eksperimen dan `interval_uji.csv` tersedia di `artifacts/tugas2/` pada direktori hasil notebook.
