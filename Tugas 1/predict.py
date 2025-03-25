import librosa
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model("audio_classification_model.h5")

# Initialize LabelEncoder with the same labels
le = LabelEncoder()
le.fit(["cat", "dog"])

def extract_waveform(file_path, max_length=22050):
    y, sr = librosa.load(file_path, sr=22050)
    if len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)), mode='constant')
    else:
        y = y[:max_length]
    return y.reshape(1, 22050, 1), y

def predict_sound(file_path):
    feature, raw_waveform = extract_waveform(file_path)
    prediction = model.predict(feature)
    predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(raw_waveform, label='Waveform')
    plt.title(f'Predicted Label: {predicted_label}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    
    return predicted_label

file_path = "../cats_dogs/cat_90.wav"  
print(f"Predicted label: {predict_sound(file_path)}")
