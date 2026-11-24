# Kode dan reproduksi

| Notebook | Fungsi |
| --- | --- |
| [UTS_Deep_Learning.ipynb](UTS_Deep_Learning.ipynb) | Menjalankan ketiga notebook mandiri, memeriksa keluaran, dan menggabungkan PDF secara urut. |
| [Tugas 1](tugas1/README.md) | Eksperimen CNN citra dan sensor, fusi, serta prediksi kelas cacat. |
| [Tugas 2](tugas2/README.md) | Baseline, GRU, CNN, interval conformal, dan prediksi sisa umur. |
| [Tugas 3](tugas3/README.md) | Embedding multimodal, pemilihan jumlah klaster, K-means, dan DEC. |

## Seluruh eksperimen di Google Colab

1. Siapkan checkout direktori repositori di runtime Colab, misalnya melalui `git clone` terhadap URL repositori kelompok. Gunakan variabel lingkungan `UTS_REPO_DIR` pada Sel 1 jika lokasi belum terdeteksi.
2. Simpan sembilan berkas dataset asli di `Drive Saya/UTS_DL/data_uts_dl/`, atau sesuaikan `UTS_DATA_DIR`. Notebook hanya membaca folder data.
3. Buka notebook gabungan, jalankan Sel 1–2, lalu jalankan `jalankan_semua()` di Sel 3. Sel ini menghasilkan `laporan.pdf`, PDF bagian, dan ketiga CSV dalam satu eksekusi.
4. Sel 4 mengunduh arsip hasil. Kode dan README tetap tersedia di checkout repositori.

## Seluruh eksperimen dari Jupyter lokal

Konfigurasi referensi memakai Python 3.12, PyTorch 2.8.0 CPU, dan versi pada [requirements.txt](requirements.txt). Jalankan pemasangan dari root checkout.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r kode/requirements.txt
python -m pip install nbconvert ipykernel
```

Letakkan data di `data_uts_dl/` pada root checkout atau set `UTS_DATA_DIR` ke lokasi dataset. Jalankan satu perintah berikut dari root checkout.

```bash
jupyter nbconvert --to notebook --execute kode/UTS_Deep_Learning.ipynb --ExecutePreprocessor.timeout=-1 --output-dir hasil_reproduksi
```

Perintah memakai kernel Python dengan dependensi di atas. Prediksi, bagian laporan, laporan gabungan, dan bukti evaluasi muncul di `hasil_reproduksi/`. Gunakan `UTS_HASIL_DIR` untuk mengganti lokasi. Versi pustaka dan perangkat aktual ditulis di hasil eksperimen.

## Checkpoint dan konsistensi laporan

- Tugas 1 dan 2 memuat checkpoint CPU yang tersedia di folder `model/` masing-masing jika konfigurasi dan sidik data cocok. Tanpa checkpoint, notebook melatih model dari dataset.
- Tugas 3 melatih ulang autoencoder dan DEC dari dataset tanpa label.
- Gunakan prediksi dan PDF dari eksekusi yang sama. Perubahan pustaka atau perangkat komputasi dapat memengaruhi hasil numerik.
- Berkas dan urutan laporan pengumpulan tercantum di [README repositori](../README.md#susunan-repositori).
