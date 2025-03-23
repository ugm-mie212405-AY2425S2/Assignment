import os
import joblib
from utils import extract_features

# Load model dan encoder
model = joblib.load("genre_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def predict_genre(file_path):
    """Prediksi genre musik dari file audio"""
    features = extract_features(file_path)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    genre = label_encoder.inverse_transform(prediction)[0]
    return genre

print("\n🎵 Selamat datang di Sistem Prediksi Genre Musik 🎵")
print("---------------------------------------------------")

while True:
    # Input path file (support drag & drop)
    audio_path = input("\n🎼 Masukkan path file musik (.wav) atau ketik 'keluar' untuk berhenti:\n> ").strip()

    # Jika user ingin keluar
    if audio_path.lower() == 'keluar':
        print("👋 Terima kasih! Sampai jumpa lagi.")
        break

    # Hapus karakter tambahan dari drag & drop
    if audio_path.startswith("& '") and audio_path.endswith("'"):
        audio_path = audio_path[3:-1]

    # Jika user drag & drop file, hapus tanda kutip tambahan yang mungkin muncul
    if audio_path.startswith('"') and audio_path.endswith('"'):
        audio_path = audio_path[1:-1]

    # Konversi ke path absolut jika hanya nama file diberikan
    if not os.path.isabs(audio_path):
        audio_path = os.path.abspath(audio_path)

    # Validasi apakah file ada
    if not os.path.exists(audio_path):
        print("❌ File tidak ditemukan! Coba lagi.")
        continue

    # Validasi format file
    if not audio_path.lower().endswith('.wav'):
        print("⚠️ Error: Hanya menerima file .wav!")
        continue

    try:
        # Prediksi genre
        genre = predict_genre(audio_path)
        print(f"\n✅ Prediksi Genre: 🎵 {genre} 🎵\n")
    except Exception as e:
        print(f"❌ Terjadi kesalahan saat memproses file: {e}")
