import librosa
import librosa.display
import numpy as np
import os
import pickle

DATASET_PATH = "dataset/mentah/"
LABELS = ["maju", "mundur", "kanan", "kiri"]
OUTPUT_FILE = "dataset/features.pkl"

def extract_features(file_path, max_length=30):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]

    return mfcc.flatten()

X, y = [], []
for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump((X, y), f)

print(f"Dataset disimpan di {OUTPUT_FILE} dengan {len(X)} sampel.")
