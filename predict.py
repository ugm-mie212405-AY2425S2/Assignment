import librosa
import pickle
import numpy as np

with open("model_knn.pkl", "rb") as f:
    knn, encoder = pickle.load(f)

def predict_audio(file_path):
    """Memprediksi perintah dari file audio."""
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    max_length = 30
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    mfcc = mfcc.flatten().reshape(1, -1)
    prediction = knn.predict(mfcc)
    label = encoder.inverse_transform(prediction)[0]
    
    return label

file_path = "4.mp3"
predicted_command = predict_audio(file_path)
print(f"Prediksi perintah: {predicted_command}")
