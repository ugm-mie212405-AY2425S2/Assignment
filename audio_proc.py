import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from glob import glob
import librosa
import librosa.display
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os


# Fungsi untuk mengekstrak fitur MFCC dari file audio
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    '''
    Mengambil rata-rata setiap koefisien MFCC
    agar memperoleh vektor fitur berdimensi tetap
    '''
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean


# Fungsi untuk memprediksi digit dari file audio baru
def predict_digit(file_path, model):
    features = extract_features(file_path)
    features = features.reshape(1, -1)  # Menyesuaikan bentuk input untuk model
    return model.predict(features)[0]


if __name__ == "__main__":
    # Konfigurasi visualisasi
    sns.set_theme(style="white", palette=None)
    color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # Mendapatkan daftar file audio dari dataset Audio MNIST
    '''
    Bagian ini menjelaskan bagaimana data diakuisisi:
    Mahasiswa harus mendapatkan dataset audio sendiri, misalnya dengan
    merekam suara menggunakan mikrofon atau menggunakan dataset publik.
    Contoh:
    audio_files = ['path/to/audio1.wav', 'path/to/audio2.wav']
    Harap sesuaikan dengan sumber data yang digunakan.

    Contoh penggunaan:
    audio_files = glob(
        'D:/codes/audio/resources/mnist_dataset/recordings/*.wav'
    )
    '''
    audio_files = []  # mahasiswa harus mengisi dengan data mereka.
    print("Jumlah file audio:", len(audio_files))
    print("Contoh file audio:", audio_files[10])

    # Visualisasi salah satu file audio
    audio_to_analyze = audio_files[189]

    # Memuat file audio
    y, sr = librosa.load(audio_to_analyze)
    print(f'y (10 sample pertama): {y[:10]}')
    print(f'shape y: {y.shape}')
    print(f'Sampling rate: {sr}')

    # Visualisasi sinyal audio
    plt.figure(figsize=(18, 8))
    plt.suptitle(audio_to_analyze)

    # Plot sinyal audio mentah
    plt.subplot(231)
    plt.plot(pd.Series(y), lw=1, color=color_pal[0])
    plt.title('Raw Audio Example')

    # Trimming: menghapus silence di awal/akhir
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    plt.subplot(232)
    plt.plot(pd.Series(y_trimmed), lw=1, color=color_pal[1])
    plt.title('Raw Audio Trimmed Example')

    # Plot zoom in pada sebagian sinyal
    plt.subplot(233)
    plt.plot(pd.Series(y[2000:2500]), lw=1, color=color_pal[2])
    plt.title('Raw Audio Zoomed In Example')

    # Hitung Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot Spectrogram
    plt.subplot(234)
    librosa.display.specshow(S_db, x_axis='time', y_axis='log')
    plt.title('Spectrogram Example')
    plt.colorbar(format='%0.2f')

    # Hitung Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

    # Plot Mel Spectrogram
    plt.subplot(235)
    librosa.display.specshow(S_db_mel, x_axis='time', y_axis='log')
    plt.title('Mel Spectrogram Example')
    plt.colorbar(format='%0.2f')

    plt.show()

    # Menyiapkan dataset fitur dan label
    X = []
    y_labels = []

    for file in audio_files:
        # Ekstraksi fitur dari masing-masing file audio
        features = extract_features(file)
        X.append(features)

        '''
        Ubah logika ekstraksi label sesuai dengan pola nama file.
        Misal: jika file bernama "0_jackson_0.wav", maka label
        diambil dari elemen pertama setelah split '_'
        '''

        filename = os.path.basename(file)
        parts = filename.split('_')
        if parts[0].isdigit():
            label = parts[0]
        else:
            label = parts[0]
            print(f"Periksa pola nama file: {filename}")

        y_labels.append(label)

    X = np.array(X)
    y_labels = np.array(y_labels)
    print("Shape fitur:", X.shape)
    print("Contoh label:", y_labels[:10])

    # Membagi dataset menjadi data latih dan data uji (80% latih, 20% uji)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.2, random_state=42
    )

    # Melatih model classifier menggunakan SVM dengan kernel linear
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Evaluasi model pada data uji
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Akurasi model:", accuracy)
    print(classification_report(y_test, y_pred))

    # Contoh penggunaan: memprediksi digit dari file audio contoh
    test_file = audio_files[2980]
    predicted_digit = predict_digit(test_file, classifier)
    print("Digit yang terprediksi untuk file", test_file, ":", predicted_digit)

    test_file = audio_files[100]
    predicted_digit = predict_digit(test_file, classifier)
    print("Digit yang terprediksi untuk file", test_file, ":", predicted_digit)

    test_file = audio_files[1450]
    predicted_digit = predict_digit(test_file, classifier)
    print("Digit yang terprediksi untuk file", test_file, ":", predicted_digit)
