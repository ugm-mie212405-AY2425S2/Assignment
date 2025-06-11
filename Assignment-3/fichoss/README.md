# Parking Slot Motor Detection using Deep Learning

Proyek ini bertujuan untuk mendeteksi jumlah motor (0–5) pada gambar tempat parkir menggunakan pendekatan klasifikasi gambar berbasis transfer learning dengan ResNet18. Dataset terdiri dari gambar tempat parkir dengan jumlah motor yang bervariasi.


## Fitur Utama

- Klasifikasi jumlah motor: 6 kelas (0–5).
- Menggunakan model pretrained ResNet18.
- Augmentasi data dan preprocessing standar.
- Visualisasi hasil prediksi dan performa.
- Evaluasi lengkap: Akurasi, Confusion Matrix, Precision, Recall, F1-score.

## Struktur Folder
parking_project/
├── sorted_images/
│ ├── train/
│ │ ├── isi0/
│ │ ├── isi1/
│ │ └── ...
│ └── val/
│ ├── isi0/
│ ├── isi1/
│ └── ...
├── labels.csv
├── models/
│ └── model_classification.pth
├── notebook/
│ └── project.ipynb
└── README.md

## Dataset

- Jumlah gambar: 150
- Format: `.jpg`
- Penamaan file: `isiX.jpg` → menunjukkan jumlah motor.
- Gambar dibagi menjadi `train/` dan `val/` otomatis.


## Preprocessing & Augmentasi

- **Resize**: 224x224
- **Normalisasi**: ImageNet
- **Augmentasi (training)**:
  - `RandomHorizontalFlip`
  - `RandomRotation(10°)`
  - `ColorJitter (brightness)`


## Model

- Arsitektur: ResNet18 (pretrained)
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (`lr=0.001`)
- Batch Size: 16
- Epochs: 10
- Device: GPU (`cuda`) jika tersedia


## Evaluasi

- Akurasi Validasi: ±90 %
- Confusion Matrix: `results/confusion_matrix.png`
- Learning Curve: `results/learning_curve.png`
- Metrik: Precision, Recall, F1-score (per kelas)

---

## Visualisasi

- Prediksi batch gambar dalam grid 2x5
- Confusion Matrix berwarna
- Kurva loss dan akurasi

---

## Pengembangan Lanjut

- Perluas dataset dengan lebih banyak gambar dan kondisi.
- Uji model lebih kompleks: ResNet50, EfficientNet.
- Gunakan YOLOv5 untuk deteksi per slot.
- Buat aplikasi sederhana berbasis Streamlit atau Flask.

---

## Kontributor

- Arfizan Rabbani - 24/554316/NPA/19977
- Najma Clara Bella - 24/554317/NPA/19978
- Pramitha Witawacita Astadewi - 24/554318/NPA/19979
- Syahnahl Dilarexa - 24/554319/NPA/19980
- Isa Fadhilah Sumitro - 24/554324/NPA/19981

- Deskripsi: Proyek tugas akhir Pengenalan Pola





