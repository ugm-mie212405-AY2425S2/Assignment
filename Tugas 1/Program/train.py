import librosa
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

def extract_waveform(file_path, max_length=22050):
    y, sr = librosa.load(file_path, sr=22050)
    if len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)), mode='constant')
    else:
        y = y[:max_length]
    return y

def load_dataset(data_dir):
    X, y = [], []
    for category in ["cat", "dog"]:
        folder_path = os.path.join(data_dir, category)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            feature = extract_waveform(file_path)
            X.append(feature)
            y.append(category)
    return np.array(X), np.array(y)

# Load training and testing datasets
train_data_dir = "../cats_dogs/train"
test_data_dir = "../cats_dogs/test"

X_train, y_train = load_dataset(train_data_dir)
X_test, y_test = load_dataset(test_data_dir)

X_train = X_train.reshape(-1, 22050, 1)
X_test = X_test.reshape(-1, 22050, 1)

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)
y_train_encoded = tf.keras.utils.to_categorical(y_train_encoded, num_classes=2)
y_test_encoded = tf.keras.utils.to_categorical(y_test_encoded, num_classes=2)

model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(22050, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_encoded, epochs=10, batch_size=8, validation_data=(X_test, y_test_encoded))

# Save the trained model
model.save("audio_classification_model.h5")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
