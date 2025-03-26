import librosa
import numpy as np


def extract_features(file_path):
    """Ekstrak fitur dari audio file (MFCC, Chroma, Spectral Contrast)"""
    y, sr = librosa.load(file_path, sr=22050)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # Gabungkan fitur
    features = np.hstack([mfcc_mean, chroma_mean, contrast_mean])
    return features
