import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa.display

guitar_path = 'dataset_gitar'
piano_path = 'dataset_piano'

def load_audio_files(path):
    audio_files = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        audio, sr = librosa.load(file_path, sr=None)
        audio_files.append(audio)
    return audio_files, sr

guitar_audios, sr = load_audio_files(guitar_path)
piano_audios, _ = load_audio_files(piano_path)


def extract_spectrogram(audio, sr):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return np.mean(mel_spec_db, axis=1) 

guitar_features = [extract_spectrogram(audio, sr) for audio in guitar_audios]
piano_features = [extract_spectrogram(audio, sr) for audio in piano_audios]

guitar_labels = [0] * len(guitar_features)
piano_labels = [1] * len(piano_features)

X = np.array(guitar_features + piano_features)
y = np.array(guitar_labels + piano_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Akurasi Klasifikasi KNN:", accuracy)

new_audio_path = 'Coba/coba3.wav'  
new_audio, _ = librosa.load(new_audio_path, sr=None)
new_feature = extract_spectrogram(new_audio, sr).reshape(1, -1)
prediction = knn.predict(new_feature)

if prediction == 0:
    print("Prediksi: Gitar")
else:
    print("Prediksi: Piano")

plt.figure(figsize=(12, 12))

plt.subplot(4, 1, 1)
guitar_spec = librosa.feature.melspectrogram(
    y=guitar_audios[0], sr=sr, n_mels=64)
guitar_spec_db = librosa.power_to_db(guitar_spec, ref=np.max)
librosa.display.specshow(guitar_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram Gitar (Dataset)')
plt.colorbar(format='%+2.0f dB')

plt.subplot(4, 1, 2)
piano_spec = librosa.feature.melspectrogram(
    y=piano_audios[0], sr=sr, n_mels=64)
piano_spec_db = librosa.power_to_db(piano_spec, ref=np.max)
librosa.display.specshow(piano_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram Piano (Dataset)')
plt.colorbar(format='%+2.0f dB')

plt.subplot(4, 1, 3)
new_spec = librosa.feature.melspectrogram(y=new_audio, sr=sr, n_mels=64)
new_spec_db = librosa.power_to_db(new_spec, ref=np.max)
librosa.display.specshow(new_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-Spectrogram Suara Baru')
plt.colorbar(format='%+2.0f dB')

plt.subplot(4, 1, 4)
if prediction == 0:
    librosa.display.specshow(guitar_spec_db, sr=sr,
                             x_axis='time', y_axis='mel')
    plt.title('Mel-Spectrogram Gitar (Prediksi)')
else:
    librosa.display.specshow(piano_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.title('Mel-Spectrogram Piano (Prediksi)')

plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
