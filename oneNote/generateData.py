import numpy as np
import soundfile as sf
import os

def generate_note(freq, duration=1.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Add slight randomness
    detune = np.random.uniform(-0.5, 0.5)  # up to ±0.5 Hz detune
    amp_variation = np.random.uniform(0.45, 0.55)  # amplitude slightly varies
    noise = np.random.normal(0, 0.005, t.shape)  # low amplitude Gaussian noise

    wave = amp_variation * np.sin(2 * np.pi * (freq + detune) * t) + noise
    return wave

# Create folders and save WAV files
notes = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00,
    "A#4": 466.16, "B4": 493.88, "C5": 523.25
}


os.makedirs("oneNote/data", exist_ok=True)

for note, freq in notes.items():
    os.makedirs(f"oneNote/data/{note}", exist_ok=True)
    for i in range(10):  # generate 10 samples of each note
        wave = generate_note(freq)
        sf.write(f"oneNote/data/{note}/{note}_{i}.wav", wave, 22050)
