# UTS Deep Learning

Tiga eksperimen pembelajaran mendalam pada dataset UTS untuk klasifikasi cacat, prediksi sisa umur mesin, dan klaster material. Setiap tugas memiliki notebook mandiri serta README yang menjelaskan masukan, eksperimen, dan keluaran.

| Tugas | Dokumentasi | Hasil referensi |
| --- | --- | --- |
| 1. Klasifikasi cacat | [README Tugas 1](kode/tugas1/README.md) | Macro-F1 evaluasi 0,818 |
| 2. Prediksi RUL | [README Tugas 2](kode/tugas2/README.md) | RMSE evaluasi 18,16 siklus |
| 3. Klaster material | [README Tugas 3](kode/tugas3/README.md) | 4 klaster dari 36.000 sampel |

## Susunan repositori

```text
.
├── README.md
├── laporan.pdf
├── prediksi_klasifikasi.csv
├── prediksi_rul.csv
├── cluster.csv
├── kontribusi.md
├── kode/
│   ├── README.md
│   ├── requirements.txt
│   ├── UTS_Deep_Learning.ipynb
│   ├── tugas1/        # Notebook, README, dan checkpoint klasifikasi
│   ├── tugas2/        # Notebook, README, dan checkpoint RUL
│   └── tugas3/        # Notebook dan README klaster
└── laporan_bagian/
    ├── 01_klasifikasi_cacat.pdf
    ├── 02_prediksi_rul.pdf
    └── 03_klaster_material.pdf
```

[Laporan gabungan](laporan.pdf) memiliki satu halaman ringkasan, diikuti bagian Tugas 1, 2, dan 3 dengan nomor halaman berkelanjutan. PDF di `laporan_bagian/` menyimpan masing-masing bagian untuk revisi kelompok.

## Reproduksi dan pengumpulan

Buka [panduan reproduksi](kode/README.md). Notebook gabungan menjalankan ketiga notebook mandiri dan membangun ulang PDF serta tiga CSV melalui `jalankan_semua()`. Data UTS dibaca dari Drive atau direktori lokal yang dikonfigurasi.

Kontribusi aktual, penggunaan AI, tautan pull request, dan persetujuan anggota dicatat di [kontribusi.md](kontribusi.md). Setiap anggota bekerja dari fork dan mengirim pull request ke repositori kelompok. Periksa enam komponen wajib di branch `main` setelah PR digabung.
