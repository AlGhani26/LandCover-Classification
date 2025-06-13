# LandCover-Classification

Project ini diajukan untuk menyelesaikan pelatihan LASKAR AI Dicoding Lintasarta NVIDIA.

## Deskripsi Proyek

Proyek ini bertujuan untuk melakukan segmentasi citra satelit menggunakan model deep learning UNet untuk klasifikasi tutupan lahan (land cover classification). Model UNet dilatih untuk mengklasifikasikan citra satelit ke dalam 7 kelas tutupan lahan yang berbeda, yaitu:

- Urban Land
- Agriculture Land
- Rangeland
- Forest Land
- Water
- Barren Land
- Unknown

Model ini memproses citra satelit dalam bentuk tiles (potongan gambar) berukuran 256x256 piksel untuk menghasilkan peta segmentasi tutupan lahan.

## Dataset

Dataset terdiri dari citra satelit dan mask multi-kelas yang sudah dipersiapkan dalam folder `dataset/` dengan pembagian:

- `train/` : Data latih (gambar dan mask)
- `valid/` : Data validasi (gambar dan mask)
- `test/` : Data uji (gambar tanpa mask)

Setiap citra satelit memiliki ekstensi `.jpg` dengan nama berakhiran `_sat.jpg`, sedangkan mask memiliki ekstensi `.png` dengan nama berakhiran `_mask.png`.

## Arsitektur Model

Model yang digunakan adalah arsitektur UNet standar dengan 7 kelas output. Model dilatih menggunakan optimizer Adam dengan fungsi loss sparse categorical crossentropy.

## Pelatihan Model

- Epochs: 100
- Batch size: 16
- Callback: ModelCheckpoint untuk menyimpan model terbaik berdasarkan nilai validasi loss

Model terbaik disimpan di `model/best_model_unet_optimasi.h5`.

## Evaluasi Model

Model dievaluasi menggunakan metrik:

- Intersection over Union (IoU)
- Dice Coefficient
- Pixel Accuracy

Evaluasi dilakukan per kelas dan secara keseluruhan dengan visualisasi hasil prediksi dibandingkan ground truth.

## Penggunaan Aplikasi

Aplikasi inferensi disediakan menggunakan Streamlit (`app.py`). Pengguna dapat mengunggah file citra satelit berformat `.tif`, kemudian aplikasi akan memproses dan menampilkan hasil segmentasi tutupan lahan secara visual.

### Cara Menjalankan

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi Streamlit:

```bash
streamlit run LandCover-Classification/app.py
```

3. Buka browser dan akses aplikasi Streamlit untuk mengunggah citra satelit dan melihat hasil segmentasi.

## Struktur Folder

```
LandCover-Classification/
├── app.py                  # Aplikasi Streamlit untuk inferensi
├── UNET.ipynb              # Notebook pelatihan dan evaluasi model UNet
├── requirements.txt        # Daftar dependencies Python
├── dataset/                # Folder dataset citra satelit dan mask
│   ├── train/
│   ├── valid/
│   └── test/
├── model/                  # Model hasil pelatihan dan history
│   ├── best_model_unet_optimasi.h5
│   └── history_unet.pkl
├── shapefile/              # Output shapefile hasil segmentasi
└── README.md               # Dokumentasi proyek ini
```

## Dependencies

- Python packages utama:
  - tensorflow
  - streamlit
  - rasterio
  - numpy
  - matplotlib
  - geopandas
  - Pillow
  - Shapely


