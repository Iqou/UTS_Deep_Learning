# Tugas 3 - Klaster material multimodal

Tugas ini mengelompokkan sampel tanpa label menggunakan citra mikrostruktur dan spektrum. Label material referensi disediakan pada tahap penilaian ARI.

## Yang dikerjakan notebook

- Memeriksa perbedaan mikroskop, termasuk kecerahan, orientasi, skala tekstur, baseline spektrum, spike, dan posisi puncak.
- Melatih dua denoising autoencoder untuk memperoleh embedding citra dan spektrum. Membandingkan embedding satu modalitas dengan gabungannya.
- Memilih jumlah klaster dari silhouette dan stabilitas dua resampling. Membandingkan K-means dengan DEC pada embedding gabungan.
- Menampilkan partisi lewat t-SNE dan meninjau sampel bermargin rendah. ARI antarpartisi mengukur kesepakatan internal.

## Menjalankan

Buka [Klaster_Material.ipynb](Klaster_Material.ipynb). Sel 1 membaca `clustering.npz`, `clustering_meta.csv`, serta `kamus_data.md` dari folder data. Jalankan Sel 1–4, lalu panggil `jalankan_tugas3()` di Sel 5. Sel 6 mengunduh CSV, PDF bagian, dan bukti eksperimen. CPU empat thread dipakai sebagai konfigurasi referensi.

## Keluaran

- [cluster.csv](../../cluster.csv) memuat kolom `id_sampel,cluster` dengan satu baris per sampel.
- [Bagian laporan 03](../../laporan_bagian/03_klaster_material.pdf) merangkum kandidat jumlah klaster, ablasi modalitas, t-SNE, dan prototipe.
- Tabel pencarian jumlah klaster serta grafik dibuat ulang oleh notebook pada setiap eksekusi.

[DEC](https://proceedings.mlr.press/v48/xieb16.html) · [t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html)
