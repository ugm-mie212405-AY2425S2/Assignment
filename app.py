import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# Load model LSTM yang sudah dilatih
MODEL_PATH = "SER_LSTM_PP.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Label emosi dalam urutan yang diminta
EMOTION_LABELS = {
    0: "Disgust",
    1: "Angry",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Pleasant Surprise",
    6: "Sad"
}

# Fungsi untuk ekstraksi fitur MFCC
def extract_mfcc(filename, max_pad_len=40):
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)  # Rata-rata fitur MFCC

        # Pastikan panjang MFCC sesuai dengan input model (40)
        if len(mfcc) < max_pad_len:
            mfcc = np.pad(mfcc, (0, max_pad_len - len(mfcc)))
        elif len(mfcc) > max_pad_len:
            mfcc = mfcc[:max_pad_len]

        return mfcc
    except Exception as e:
        st.error(f"Error dalam ekstraksi fitur: {str(e)}")
        return None

# Konfigurasi halaman Streamlit
st.title("\U0001F399️ Speech Emotion Recognition")
st.write("Unggah file audio untuk mendeteksi emosi.")

# Upload file audio
uploaded_file = st.file_uploader("Pilih file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Simpan file yang diunggah sementara
    temp_audio_path = "uploaded_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # **Tambahkan pemutar audio di Streamlit**
    st.audio(temp_audio_path, format='audio/wav')

    # Ekstraksi fitur dari file audio
    features = extract_mfcc(temp_audio_path)
    
    if features is not None:
        # 🔥 **Fix Dimensi Input untuk Model LSTM**
        features = features.reshape(1, 40, 1)  # Ubah ke format (batch_size, timesteps, features)

        # Prediksi emosi dengan model LSTM
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)  # Ambil label dengan probabilitas tertinggi

        # Tampilkan hasil prediksi
        st.subheader("\U0001F50D Hasil Prediksi Emosi:")
        st.write(f"**{EMOTION_LABELS[predicted_label]}**")

        # Debug informasi tambahan
        st.write("\U0001F4CA Probabilitas Emosi:")
        for i, emotion in EMOTION_LABELS.items():
            st.write(f"{emotion}: {prediction[0][i]:.4f}")

    # Hapus file sementara setelah digunakan
    os.remove(temp_audio_path)
