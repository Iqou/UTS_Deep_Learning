# UTS Deep Learning --- Task 1, Task 2, dan Task 3

Repositori ini berisi implementasi, eksperimen, evaluasi, dan hasil
akhir tiga tugas Ujian Tengah Semester (UTS) mata kuliah Deep Learning.
Seluruh eksperimen menggunakan dataset yang disediakan untuk tugas dan
dijalankan menggunakan Python serta PyTorch. Notebook demonstrasi
disiapkan agar proses inferensi dan pemeriksaan hasil dapat dijalankan
kembali melalui **Google Colab**.

## Daftar Isi

-   [Ringkasan Proyek](#ringkasan-proyek)
-   [Struktur Direktori](#struktur-direktori)
-   [Persiapan Google Drive](#persiapan-google-drive)
-   [Task 1 --- Multimodal
    Classification](#task-1--multimodal-classification)
-   [Task 2 --- Remaining Useful Life
    Prediction](#task-2--remaining-useful-life-prediction)
-   [Task 3 --- Multimodal Clustering](#task-3--multimodal-clustering)
-   [Cara Menjalankan Notebook di Google
    Colab](#cara-menjalankan-notebook-di-google-colab)
-   [File Hasil dan Submission](#file-hasil-dan-submission)
-   [Reproduksibilitas dan Catatan
    Evaluasi](#reproduksibilitas-dan-catatan-evaluasi)
-   [Catatan Penggunaan](#catatan-penggunaan)

------------------------------------------------------------------------

## Ringkasan Proyek

  -----------------------------------------------------------------------
  Task              Permasalahan      Pendekatan utama  Hasil utama
  ----------------- ----------------- ----------------- -----------------
  Task 1            Klasifikasi cacat Multimodal CNN,   Macro-F1 validasi
                    produk            citra dan sinyal  **0,750697**
                                      sensor, ensemble  
                                      tiga seed         

  Task 2            Prediksi          GRU ensemble dan  Endpoint
                    Remaining Useful  Temporal          validation RMSE
                    Life (RUL)        Convolutional     **20,638390**
                                      Network (TCN)     

  Task 3            Pengelompokan     Denoising         Silhouette DEC
                    jenis material    Autoencoder       fusion
                                      multimodal dan    **0,432194**
                                      Deep Embedded     
                                      Clustering (DEC)  
  -----------------------------------------------------------------------

Ketiga task menggunakan pendekatan berbeda:

1.  **Task 1 --- Supervised classification:** model mempelajari hubungan
    antara fitur citra, sinyal sensor, dan label kelas cacat.
2.  **Task 2 --- Supervised regression:** model mempelajari pola
    temporal data sensor untuk memperkirakan sisa umur pakai mesin.
3.  **Task 3 --- Unsupervised clustering:** model mencari struktur
    kelompok berdasarkan kemiripan citra mikrostruktur dan spektrum
    tanpa menggunakan label jenis material.

## Struktur Direktori

Struktur berikut direkomendasikan untuk menyatukan semua berkas proyek
setelah masing-masing paket Task 1, Task 2, dan Task 3 diekstrak.

``` text
UTS_Deep_Learning_Final/
│
├── README.md
│
├── notebooks/
│   ├── Task1_Presentasi_Google_Colab.ipynb
│   ├── Task2_Presentasi_Google_Colab.ipynb
│   └── Task3_Presentasi_Google_Colab.ipynb
│
├── data/
│   ├── task1/
│   │   ├── klasifikasi_train.npz
│   │   ├── klasifikasi_train_meta.csv
│   │   ├── klasifikasi_test.npz
│   │   └── klasifikasi_test_meta.csv
│   │
│   ├── task2/
│   │   ├── rul_train.csv
│   │   └── rul_test.csv
│   │
│   └── task3/
│       ├── clustering.npz
│       └── clustering_meta.csv
│
├── task1/
│   ├── kode/
│   │   ├── task1.py
│   │   ├── data_task1.py
│   │   ├── reproduksi.py
│   │   ├── laporan.py
│   │   └── requirements.txt
│   ├── hasil/
│   │   ├── bobot_42.pt
│   │   ├── bobot_123.pt
│   │   ├── bobot_2026.pt
│   │   ├── metrik.csv
│   │   ├── confusion.csv
│   │   └── kurva.png
│   ├── Laporan_Task_1.pdf
│   └── prediksi_klasifikasi.csv
│
├── task2/
│   ├── weights/
│   │   ├── gru_seed42.pt
│   │   ├── gru_seed123.pt
│   │   ├── gru_seed2026.pt
│   │   └── tcn_seed42.pt
│   ├── hasil/
│   ├── task2_solution.py
│   ├── reproduce_inference.py
│   ├── preprocessor_parameters.npz
│   ├── metrics_validation.json
│   ├── Laporan_Task_2_RUL.pdf
│   └── prediksi_rul.csv
│
├── task3/
│   ├── weights/
│   │   ├── image_autoencoder.pt
│   │   ├── spectrum_autoencoder.pt
│   │   └── dec_fusion.pt
│   ├── embeddings/
│   ├── figures/
│   ├── task3_solution.py
│   ├── reproduce_inference.py
│   ├── finalize_task3.py
│   ├── run_task3.py
│   ├── metrics_task3.json
│   ├── Laporan_Task_3_Multimodal_Clustering.pdf
│   └── cluster.csv
│
└── laporan/
    └── Laporan_Akhir.pdf
```

> Struktur di atas adalah struktur penggabungan yang direkomendasikan.
> Nama folder di dalam paket ZIP asli dapat berbeda. Pertahankan isi dan
> nama file sumber saat mengekstrak paket, atau sesuaikan path pada
> notebook jika memilih nama folder lain. Folder `laporan/` digunakan
> untuk laporan akhir gabungan apabila sudah disusun.

## Persiapan Google Drive

Notebook yang disediakan menggunakan folder Google Drive terpisah untuk
setiap task. Dengan pemisahan ini, file data, bobot model, dan hasil
tiap task tidak tercampur.

### Task 1

``` text
MyDrive/
└── UTS_Deep_Learning_Task1/
    ├── Task_1_Klasifikasi.zip
    └── data/
        ├── klasifikasi_train(1).npz
        ├── klasifikasi_train_meta(1).csv
        ├── klasifikasi_test(1).npz
        └── klasifikasi_test_meta(1).csv
```

### Task 2

``` text
MyDrive/
└── UTS_Deep_Learning_Task2/
    ├── Task_2_RUL_Colab.zip
    └── data/
        ├── rul_train(1).csv
        └── rul_test(1).csv
```

### Task 3

``` text
MyDrive/
└── UTS_Deep_Learning_Task3/
    ├── Task_3_Multimodal_Clustering_Colab.zip
    └── data/
        ├── clustering(1).npz
        └── clustering_meta(1).csv
```

**Dataset tidak disertakan di dalam paket ZIP kode dan model.** Upload
dataset asli yang diberikan untuk tugas ke folder `data/` masing-masing.
Pastikan file dataset dan metadata berasal dari rilis yang sama. Untuk
Task 3, urutan `indeks` pada metadata harus sesuai dengan urutan sampel
pada file NPZ.

## Task 1 --- Multimodal Classification

### Tujuan

Mengklasifikasikan kondisi atau jenis cacat produk berdasarkan dua
sumber informasi:

-   Citra permukaan produk.
-   Sinyal sensor mesin.

Model menghasilkan prediksi untuk tujuh kelas:

-   `normal`
-   `gores`
-   `noda`
-   `retak`
-   `gelembung`
-   `las_dingin`
-   `kontaminasi`

### Dataset dan pembagian

Eksperimen menggunakan 48.000 unit untuk pengembangan model:

-   38.400 unit untuk pelatihan.
-   9.600 unit untuk validasi.

Data validasi dibagi dengan mempertimbangkan kelas dan lini produksi
agar distribusinya terwakili. Data uji terdiri dari 16.000 unit dan
label aktualnya tidak tersedia untuk eksperimen lokal.

### Metode

1.  Membaca citra, sinyal sensor, dan metadata.
2.  Memeriksa kesesuaian ID dan indeks data.
3.  Membentuk fitur atau feature maps citra dan sinyal.
4.  Melakukan standardisasi berdasarkan data pelatihan.
5.  Menggunakan encoder CNN 2D untuk citra dan CNN 1D untuk sinyal
    sensor.
6.  Menggabungkan representasi kedua modalitas.
7.  Melatih model dengan tiga seed: 42, 123, dan 2026.
8.  Merata-ratakan probabilitas ketiga model untuk menghasilkan prediksi
    ensemble.

Citra dan sensor digunakan bersama karena keduanya menangkap
karakteristik cacat yang berbeda. Sebagai contoh, kelas `las_dingin`
lebih mudah dikenali dari sinyal sensor, sedangkan kelas `noda` lebih
informatif dari citra.

### Hasil utama

  Model                             Macro-F1 validasi
  ------------------------------- -------------------
  Citra saja                                 0,433813
  Sensor saja                                0,414373
  Ensemble multimodal tiga seed          **0,750697**

Performa ensemble berdasarkan lini produksi:

  Lini            Jumlah sampel validasi   Macro-F1
  ------------- ------------------------ ----------
  A                                7.193   0,790033
  B                                2.407   0,635516
  Keseluruhan                      9.600   0,750697

Hasil per lini perlu diperhatikan karena komposisi lini pada data uji
berbeda dari data latih. Karena itu, skor validasi keseluruhan tidak
boleh ditafsirkan tanpa mempertimbangkan perbedaan distribusi lini.

### Berkas penting

-   `task1.py`: model klasifikasi, training, validasi, dan inferensi.
-   `data_task1.py`: pemuatan data dan preprocessing.
-   `reproduksi.py`: pipeline reproduksi eksperimen.
-   `hasil/bobot_42.pt`, `bobot_123.pt`, `bobot_2026.pt`: bobot final.
-   `hasil/metrik.csv`: metrik keseluruhan dan per kelompok.
-   `hasil/kurva.png`: kurva pelatihan.
-   `hasil/confusion.csv`: confusion matrix.
-   `prediksi_klasifikasi.csv`: prediksi data uji.

### Reproduksi di Google Colab

Buka `Task1_Presentasi_Google_Colab.ipynb`, jalankan cell dari atas ke
bawah, dan izinkan notebook mengakses Google Drive.

Notebook menggunakan bobot final untuk menghitung ulang Macro-F1 pada
validasi dan membandingkan hasil prediksi 16.000 unit uji dengan CSV
submission. Mode ini merupakan **inferensi dari bobot tersimpan**, bukan
training ulang dari awal.

Target reproduksi: Macro-F1 validasi sekitar **0,7507** dan prediksi
test cocok dengan `prediksi_klasifikasi.csv`.

## Task 2 --- Remaining Useful Life Prediction

### Tujuan

Memperkirakan **Remaining Useful Life (RUL)**, yaitu perkiraan sisa
siklus operasi mesin sebelum mencapai akhir masa pakainya.

Label RUL dibentuk dengan rumus:

\[ RUL_t = `\min`{=tex}(150, T\_{`\text{failure}`{=tex}} - t) \]

Nilai RUL dibatasi maksimum 150 siklus.

### Dataset dan pembagian

-   Data pelatihan: 900 mesin, 254.174 baris pengamatan.
-   Data uji: 300 mesin, 51.692 baris pengamatan.
-   Sensor: 21 kanal.
-   Pengaturan operasi: 3 variabel.

Pembagian validasi dilakukan berdasarkan `id_unit`, bukan secara acak
per baris, agar data dari mesin yang sama tidak bocor ke subset training
dan validation.

-   720 mesin untuk training.
-   180 mesin untuk validation.

### Preprocessing dan metode

1.  Membentuk label RUL dengan batas maksimum 150.
2.  Melakukan normalisasi yang mempertimbangkan kelompok pengaturan
    mesin.
3.  Menggunakan median dan robust MAD untuk mengurangi pengaruh nilai
    ekstrem.
4.  Menangani nilai sensor yang hilang dengan imputasi dan indikator
    missingness.
5.  Membentuk sequence sepanjang 30 siklus dengan stride 3.
6.  Menghasilkan 53 fitur per timestep.
7.  Membandingkan baseline LightGBM, GRU, dan TCN.
8.  Menggabungkan prediksi tiga model GRU dan satu model TCN.

Model final menggunakan:

\[ `\widehat{RUL}`{=tex} =0.60
`\times `{=tex}`\overline{\widehat{RUL}}`{=tex}*{`\text{GRU}`{=tex}}
+0.40 `\times `{=tex}`\widehat{RUL}`{=tex}*{`\text{TCN}`{=tex}} \]

Tiga model GRU menggunakan seed 42, 123, dan 2026; TCN menggunakan seed
42.

### Hasil validasi

Metrik berikut adalah hasil **machine endpoint validation** pada 180
mesin holdout.

  Model                               RMSE         MAE
  ------------------------ --------------- -----------
  LightGBM baseline              21,050860   15,566204
  GRU 3-seed ensemble            20,861393   15,453866
  TCN                            21,406004   16,817570
  Final GRU-TCN ensemble     **20,638390**   15,824810

### Ringkasan prediksi data uji

  Informasi                       Hasil
  ------------------------ ------------
  Jumlah mesin uji                  300
  Minimum prediksi RUL         3,997773
  Rata-rata prediksi RUL      91,827543
  Median prediksi RUL        105,424835
  Maksimum prediksi RUL      150,000000

**Perbedaan penting:** rata-rata prediksi 91,827543 adalah ringkasan
prediksi pada data uji, bukan RMSE. RMSE 20,638390 dihitung pada data
validasi yang memiliki label aktual.

### Berkas penting

-   `task2_solution.py`: pipeline training dan evaluasi.
-   `reproduce_inference.py`: inferensi menggunakan bobot final.
-   `weights/`: tiga bobot GRU dan satu bobot TCN.
-   `preprocessor_parameters.npz`: parameter preprocessing untuk
    inferensi.
-   `metrics_validation.json`: catatan metrik validasi.
-   `hasil/`: kurva, tabel, dan visualisasi pendukung.
-   `prediksi_rul.csv`: prediksi untuk 300 mesin uji.
-   `prediksi_rul_interval.csv`: keluaran interval prediksi.

### Reproduksi di Google Colab

Buka `Task2_Presentasi_Google_Colab.ipynb`, jalankan cell dari atas ke
bawah, dan izinkan akses Google Drive.

Notebook menjalankan inferensi cepat dengan empat bobot final dan
membandingkan 300 prediksi baru dengan `prediksi_rul.csv`.

Cell metrik endpoint validation membaca catatan hasil eksperimen pada
`metrics_validation.json`. Untuk **menghitung ulang RMSE dari proses
training dan evaluasi**, aktifkan bagian training penuh pada notebook.
Proses tersebut lebih lama karena melatih ulang baseline, GRU, dan TCN
pada split validasi.

Target inferensi cepat: rata-rata prediksi RUL sekitar **91,827543** dan
300 ID cocok dengan submission.

## Task 3 --- Multimodal Clustering

### Tujuan

Mengelompokkan sampel material berdasarkan kemiripan karakteristik citra
mikrostruktur dan spektrum komposisi. Task ini menggunakan
**unsupervised learning**, sehingga tidak menggunakan label jenis
material sebagai target pelatihan.

### Dataset

-   Jumlah sampel: 36.000.
-   Citra mikrostruktur: 64 × 64 piksel.
-   Spektrum: 128 titik.
-   Metadata mikroskop: M1 dan M2.

Setiap sampel memiliki pasangan citra, spektrum, dan metadata.
Keselarasan indeks harus diperiksa sebelum preprocessing.

### Preprocessing dan representasi

**Citra:** - Normalisasi per citra. - Clipping nilai ekstrem. -
Denoising augmentation saat training.

**Spektrum:** - Deteksi dan koreksi spike lokal. - Baseline
correction. - Standard Normal Variate (SNV). - Denoising augmentation
saat training.

Dua autoencoder digunakan karena struktur data berbeda:

-   CNN denoising autoencoder untuk citra, menghasilkan embedding 48
    dimensi.
-   MLP denoising autoencoder untuk spektrum, menghasilkan embedding 32
    dimensi.

Kedua embedding digabungkan dengan bobot 80% citra dan 20% spektrum:

\[ Z\_{`\text{fusion}`{=tex}} =
\[`\sqrt{0.8}`{=tex}Z\_{`\text{image}`{=tex}},`\sqrt{0.2}`{=tex}Z\_{`\text{spectrum}`{=tex}}\]
\]

### Pemilihan jumlah cluster dan DEC

Jumlah cluster diuji pada rentang (k=2) sampai (k=12), menggunakan:

-   Silhouette Score --- semakin tinggi semakin baik.
-   Calinski--Harabasz Index --- semakin tinggi semakin baik.
-   Davies--Bouldin Index --- semakin rendah semakin baik.

Pemilihan dilakukan berdasarkan rata-rata peringkat ketiga metrik. Hasil
konsensus memilih **k = 2**. Calinski--Harabasz sedikit lebih tinggi
pada k=3, sedangkan Silhouette dan Davies--Bouldin mendukung k=2.

KMeans digunakan sebagai baseline. Metode deep clustering utama adalah
**Deep Embedded Clustering (DEC)**, yang menyempurnakan embedding dan
pusat cluster menggunakan target distribution dan KL-divergence.

### Hasil utama DEC fusion

  Metrik                                              Hasil
  ------------------------------------------ --------------
  Jumlah cluster                                          2
  Silhouette                                   **0,432194**
  Calinski--Harabasz                           3.874,726912
  Davies--Bouldin                                  1,049043
  Adjusted Mutual Information vs mikroskop         0,004869

Distribusi cluster:

  Cluster     Jumlah sampel   Persentase
  --------- --------------- ------------
  0                  10.896       30,27%
  1                  25.104       69,73%
  Total              36.000         100%

KMeans pada embedding citra saja memperoleh Silhouette 0,618854, lebih
tinggi daripada DEC fusion. Hasil ini dilaporkan secara transparan. DEC
tetap menjadi metode utama karena tujuan task adalah menerapkan deep
clustering, bukan hanya memaksimalkan satu metrik internal.

### Berkas penting

-   `task3_solution.py`: preprocessing, autoencoder, pemilihan k, dan
    DEC.
-   `run_task3.py`: runner training dari awal.
-   `reproduce_inference.py`: inferensi dari bobot final dan hitung
    ulang metrik.
-   `finalize_task3.py`: analisis dan visualisasi hasil.
-   `weights/`: bobot image autoencoder, spectrum autoencoder, dan DEC.
-   `embeddings/`: representasi laten hasil encoder.
-   `figures/`: contoh data, prototipe cluster, dan visualisasi t-SNE.
-   `metrics_task3.json`: metrik dan ringkasan eksperimen.
-   `cluster_count_criteria.csv`: perbandingan jumlah cluster.
-   `modality_ablation.csv`: perbandingan modalitas.
-   `cluster_summary.csv`: ringkasan karakteristik cluster.
-   `cluster.csv`: hasil clustering 36.000 sampel.

### Reproduksi di Google Colab

Buka `Task3_Presentasi_Google_Colab.ipynb`, jalankan cell dari atas ke
bawah, dan izinkan akses Google Drive.

Notebook memuat bobot final, membentuk ulang embedding, menjalankan DEC,
menghasilkan cluster baru, menghitung metrik internal, dan
membandingkannya dengan `cluster.csv`.

Target reproduksi: - Silhouette sekitar **0,432194**. - Cluster 0
sebanyak **10.896**. - Cluster 1 sebanyak **25.104**. - Exact label
match terhadap `cluster.csv` sebesar **1,0**.

**Catatan evaluasi:** label material sebenarnya tidak tersedia. Oleh
karena itu, ARI terhadap hidden reference tidak dapat dihitung secara
lokal. Exact label match atau ARI 1,0 terhadap `cluster.csv` hanya
menunjukkan bahwa inferensi ulang mereproduksi submission tersimpan; itu
bukan evaluasi terhadap label material sebenarnya.

## Cara Menjalankan Notebook di Google Colab

Tiga notebook presentasi disediakan:

  ---------------------------------------------------------------------------
  Notebook                                Kegunaan
  --------------------------------------- -----------------------------------
  `Task1_Presentasi_Google_Colab.ipynb`   Reproduksi Macro-F1 dan prediksi
                                          klasifikasi

  `Task2_Presentasi_Google_Colab.ipynb`   Inferensi RUL dan pemeriksaan hasil
                                          prediksi

  `Task3_Presentasi_Google_Colab.ipynb`   Inferensi DEC dan evaluasi
                                          clustering
  ---------------------------------------------------------------------------

Langkah umum:

1.  Download notebook yang sesuai.
2.  Buka [Google Colab](https://colab.research.google.com/).
3.  Pilih **File → Upload notebook**.
4.  Upload file `.ipynb`.
5.  Siapkan folder dan file di Google Drive sesuai bagian [Persiapan
    Google Drive](#persiapan-google-drive).
6.  Jalankan cell dari atas ke bawah.
7.  Berikan izin akses Google Drive ketika diminta.
8.  Periksa hasil metrik dan file output yang dihasilkan.

Notebook demonstrasi cepat menggunakan bobot model yang sudah disimpan.
Training ulang penuh tersedia sebagai langkah opsional dan membutuhkan
waktu serta komputasi lebih besar.

## File Hasil dan Submission

Tiga file submission utama:

  Task     File                           Jumlah baris Kolom
  -------- ---------------------------- -------------- ------------------------
  Task 1   `prediksi_klasifikasi.csv`           16.000 `id_unit`, `kelas`
  Task 2   `prediksi_rul.csv`                      300 `id_unit`, `rul`
  Task 3   `cluster.csv`                        36.000 `id_sampel`, `cluster`

Ketentuan pemeriksaan:

-   Jangan mengubah nama kolom submission.
-   Pastikan setiap ID muncul sesuai ketentuan tugas.
-   Jangan mengurutkan ulang atau mengubah ID tanpa alasan.
-   Untuk Task 1 dan Task 2, nilai prediksi harus sesuai tipe target
    masing-masing.
-   Untuk Task 3, nomor cluster merupakan label kelompok integer dan
    bersifat arbitrer.

## Reproduksibilitas dan Catatan Evaluasi

-   Gunakan dataset asli yang sesuai dengan masing-masing eksperimen.
-   Jangan mencampur file dataset atau metadata dari rilis yang berbeda.
-   Gunakan bobot dan parameter preprocessing yang disertakan untuk
    inferensi cepat.
-   Notebook menampilkan apakah suatu metrik dihitung ulang atau hanya
    dibaca dari catatan eksperimen.
-   Hasil dapat sedikit berbeda jika versi Python, PyTorch, NumPy,
    scikit-learn, perangkat, atau operasi komputasi berubah.
-   Simpan output reproduksi pada folder baru agar file submission dan
    bobot final tidak tertimpa.
-   Untuk Task 2, RMSE harus dihitung pada data validasi berlabel, bukan
    dari data test yang label aktualnya tersembunyi.
-   Untuk Task 3, jangan menafsirkan ID cluster sebagai nama material
    tanpa validasi referensi atau pakar.

## Catatan Penggunaan

README ini menjelaskan susunan dan alur proyek. Laporan masing-masing
task berisi uraian eksperimen yang lebih lengkap. Untuk kebutuhan
pengumpulan, gunakan laporan akhir gabungan sesuai instruksi dosen dan
sertakan tiga file submission yang diwajibkan.

Pastikan seluruh anggota kelompok memeriksa kode, hasil, dan pembagian
kontribusi masing-masing serta dapat menjelaskan proses eksperimen pada
saat ujian lisan.
