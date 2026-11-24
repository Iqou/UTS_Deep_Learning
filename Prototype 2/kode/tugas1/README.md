# Tugas 1 - Klasifikasi cacat produk

Tugas ini memprediksi satu dari tujuh kelas cacat untuk setiap unit uji menggunakan citra permukaan dan tiga kanal sinyal mesin. Metrik penilaian adalah macro-F1.

## Yang dikerjakan notebook

- Memeriksa distribusi kelas, citra dan sinyal per kelas, kanal hilang, serta perbedaan lini antara latih dan uji.
- Melatih CNN citra, CNN sensor, fusi fitur, dan fusi logit menggunakan split stratifikasi kelas dan lini.
- Membandingkan label smoothing serta augmentasi. Audit label menandai ketidaksesuaian berprobabilitas tinggi sebagai kandidat pemeriksaan.
- Mengukur performa per kelas dan lini, menguji sensor kosong, dan menjalankan sanity check pada satu batch. Model akhir menggabungkan tiga seed.

## Menjalankan

Buka [Klasifikasi_Cacat.ipynb](Klasifikasi_Cacat.ipynb). Sel 1 membaca `klasifikasi_train.npz`, `klasifikasi_train_meta.csv`, `klasifikasi_test.npz`, dan `klasifikasi_test_meta.csv` dari folder data. Jalankan Sel 1–4, lalu panggil `jalankan_tugas1()` di Sel 5. Sel 6 mengekspor hasil.

Pada checkout repositori, notebook memuat checkpoint CPU dari `model/` jika cocok dengan dataset dan konfigurasi. Jika checkpoint tidak tersedia, notebook melatih model dari data UTS. Konfigurasi referensi memakai CPU empat thread. Gunakan hasil satu eksekusi yang sama untuk prediksi dan laporan.

## Keluaran

- [prediksi_klasifikasi.csv](../../prediksi_klasifikasi.csv) memuat kolom `id_unit,kelas` dengan satu baris per unit uji.
- [Bagian laporan 01](../../laporan_bagian/01_klasifikasi_cacat.pdf) memuat eksplorasi, ablasi, kurva, dan hasil evaluasi.
- Bukti tambahan, termasuk daftar kandidat audit label dan pembagian data, dibuat di `artifacts/tugas1/` pada direktori hasil notebook.
