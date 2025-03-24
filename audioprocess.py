import librosa
import soundfile as sf
import numpy as np

def adjust_volume(y, db_change):
    """Adjust the volume of an audio signal."""
    factor = 10 ** (db_change / 20)
    return y * factor

def adjust_pitch(y, sr, n_steps):
    """Adjust the pitch of an audio signal."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    """Stretch the time of an audio signal."""
    return librosa.effects.time_stretch(y, rate)

def separate_harmonic_percussive(y):
    """Separate harmonic and percussive components from an audio signal."""
    return librosa.effects.hpss(y)

def enhance_percussive(y):
    """Enhance the percussive components of an audio signal."""
    return librosa.effects.percussive(y)

def enhance_harmonic(y):
    """Enhance the harmonic components of an audio signal."""
    return librosa.effects.harmonic(y)

def trim_silence(y, top_db=40):
    """Trim leading and trailing silence from an audio signal."""
    return librosa.effects.trim(y, top_db=top_db)[0]

def process_audio(
    input_path, output_path, db_change=0, pitch_change=0, stretch_rate=1.0,
    apply_hpss=False, enhance_perc=False, enhance_harm=False, trim_audio=False
):
    """
    Process an audio file with various effects.

    Parameters:
    input_path (str): Path to the input audio file.
    output_path (str): Path to save the processed audio file.
    db_change (float): Change in volume (in dB).
    pitch_change (float): Change in pitch (in semitones).
    stretch_rate (float): Time stretching factor.
    apply_hpss (bool): Whether to apply harmonic-percussive separation.
    enhance_perc (bool): Whether to enhance percussive components.
    enhance_harm (bool): Whether to enhance harmonic components.
    trim_audio (bool): Whether to trim silence.
    """
    y, sr = librosa.load(input_path, sr=None)
    
    y = adjust_volume(y, db_change)
    y = adjust_pitch(y, sr, pitch_change)
    
    if stretch_rate != 1.0:
        y = time_stretch(y, stretch_rate)
    
    if apply_hpss:
        y, _ = separate_harmonic_percussive(y)
    
    if enhance_perc:
        y = enhance_percussive(y)
    
    if enhance_harm:
        y = enhance_harmonic(y)
    
    if trim_audio:
        y = trim_silence(y)
    
    sf.write(output_path, y, sr)
    print(f"Processed audio saved: {output_path}")

# Example usage
INPUT_AUDIO = "dog.mp3"  # Replace with actual file
OUTPUT_AUDIO = "dog_barking_modified.wav"
DB_CHANGE = -5  # Volume adjustment
PITCH_CHANGE = 3  # Pitch adjustment
STRETCH_RATE = 1.0  # Time stretching factor
APPLY_HPSS = True  # Harmonic-percussive separation
ENHANCE_PERC = False  # Percussion enhancement
ENHANCE_HARM = False  # Harmonic enhancement
TRIM_AUDIO = True  # Silence trimming

process_audio(
    INPUT_AUDIO, OUTPUT_AUDIO, DB_CHANGE, PITCH_CHANGE, STRETCH_RATE,
    APPLY_HPSS, ENHANCE_PERC, ENHANCE_HARM, TRIM_AUDIO
)