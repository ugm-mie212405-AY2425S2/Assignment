import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


dataset_path = "dataset/" 

classes = os.listdir(dataset_path)

x, y = [], []

for label, class_name in enumerate(classes):
    class_path = os.path.join(dataset_path, class_name)
    
    for file in os.listdir(class_path):
        file_path = os.path.join(class_path, file)

        signal, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  

        x.append(mfcc_mean)
        y.append(label)  

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Akurasi Model: {accuracy * 100:.2f}%")
print("\n🔍 Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=classes))


def prediksi_suara(file_audio):
    signal, sr = librosa.load(file_audio, sr=22050) 
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13) 
    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)
    
    prediksi = knn.predict(mfcc_mean)
    return classes[prediksi[0]]


# uji coba
file_uji = "dataset/mobil/38689__shimsewn__car-traffic-blend-01.wav"  
hasil_prediksi = prediksi_suara(file_uji)
print(f"🔊 Suara dari '{file_uji}' terdeteksi sebagai: {hasil_prediksi}")
