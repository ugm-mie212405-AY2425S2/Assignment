import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Dataset and model paths
DATA_PATH = "archive/cats_dogs"
MODEL_PATH = "cat_dog_audio_model.h5"

# Audio parameters
N_MFCC = 40  # Number of MFCC coefficients
DURATION = 2  # Audio duration in seconds
SAMPLE_RATE = 22050  # Sample rate
MAX_LENGTH = 87  # Maximum length of MFCC after padding/truncation

def extract_features(file_path, sr=SAMPLE_RATE):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=sr, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Pad or truncate to ensure uniform dimensions
    if mfcc.shape[1] < MAX_LENGTH:
        pad_width = MAX_LENGTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LENGTH]
    
    return mfcc

# Load dataset
X, y = [], []
for file in os.listdir(DATA_PATH):
    file_path = os.path.join(DATA_PATH, file)
    if file.endswith(".wav"):
        mfcc = extract_features(file_path)
        X.append(mfcc)
        y.append(0 if "cat" in file else 1)

X = np.array(X)
y = np.array(y)

# Expand dimensions for CNN input
X = X[..., np.newaxis]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Model accuracy: {acc:.2f}")

# Save model
model.save(MODEL_PATH)
print(f"Model saved as {MODEL_PATH}")
