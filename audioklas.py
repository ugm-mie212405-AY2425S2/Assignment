import numpy as np
import librosa
import tensorflow as tf

# Load model
MODEL_PATH = "cat_dog_audio_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Debugging: Check model input and output shape
print(f"Model Input Shape: {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")

def process_audio(file_path, sr=22050, duration=2.0):
    """
    Process the audio file and extract MFCC features.
    
    Parameters:
    file_path (str): Path to the audio file.
    sr (int): Sampling rate.
    duration (float): Duration of the audio to load.
    
    Returns:
    np.ndarray: Processed MFCC features ready for model prediction.
    """
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Ensure MFCC shape (40, 87)
    if mfcc.shape[1] < 87:
        pad_width = 87 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1] > 87:
        mfcc = mfcc[:, :87]

    # Adjust dimensions (batch, height, width, channel)
    mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension (1)
    mfcc = np.expand_dims(mfcc, axis=0)   # Add batch dimension

    print(f"Processed Audio Shape: {mfcc.shape}")  # Debugging
    return mfcc

# Path to the audio file for testing
AUDIO_FILE = "dog_barking_modified.wav"

# Prediction
input_data = process_audio(AUDIO_FILE)
prediction = model.predict(input_data)[0]  # Get first result

# Check if model uses sigmoid or softmax
if model.output_shape[-1] == 1:  # Binary classification (Sigmoid)
    confidence = float(prediction[0]) * 100  # Convert to percentage
    predicted_label = "Dog" if prediction[0] > 0.5 else "Cat"
    confidence = max(confidence, 100 - confidence)  # Get highest confidence
    print(f"Predicted Label: {predicted_label} ({confidence:.2f}%)")
else:  # Multi-class classification (Softmax)
    label_map = {0: "Cat", 1: "Dog"}
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index]) * 100  # Convert to percentage
    print(f"Confidence Scores: Cat {prediction[0] * 100:.2f}%, Dog {prediction[1] * 100:.2f}%")
    print(f"Predicted Label: {label_map[predicted_index]} ({confidence:.2f}%)")
