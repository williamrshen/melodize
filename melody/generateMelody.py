import numpy as np
import soundfile as sf
import os
import random

# --- Major Scales ---
major_keys = {
    "C": ["C", "D", "E", "F", "G", "A", "B"],
    "G": ["G", "A", "B", "C", "D", "E", "F#"],
    "D": ["D", "E", "F#", "G", "A", "B", "C#"],
    "A": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "F#": ["F#", "G#", "A#", "B", "C#", "D#", "F"],
    "C#": ["C#", "D#", "F", "F#", "G#", "A#", "C"],
    "F": ["F", "G", "A", "A#", "C", "D", "E"],
    "A#": ["A#", "C", "D", "D#", "F", "G", "A"],
    "D#": ["D#", "F", "G", "G#", "A#", "C", "D"],
    "G#": ["G#", "A#", "C", "C#", "D#", "F", "G"],
}

# --- Frequencies ---
note_freqs = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00,
    "A#4": 466.16, "B4": 493.88, "C5": 523.25
}

# --- Note Generator ---
def generate_note(freq, duration=1.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    detune = np.random.uniform(-0.5, 0.5)
    amp_variation = np.random.uniform(0.45, 0.55)
    noise = np.random.normal(0, 0.005, t.shape)
    wave = amp_variation * np.sin(2 * np.pi * (freq + detune) * t) + noise
    return wave

# --- Melody Generator ---
def generate_pleasant_melody(notes_in_key, tonic, length=12):
    melody = [tonic]  # start with tonic
    idx = notes_in_key.index(tonic)

    for _ in range(length - 2):  # build inner part of melody
        step = random.choice([-2, -1, 0, 1, 2])  # prefer small steps
        idx = max(0, min(len(notes_in_key) - 1, idx + step))
        melody.append(notes_in_key[idx])

    melody.append(tonic)  # end on tonic
    return melody

# --- Melody Settings ---
songs = 50
shortest = 20
longest = 40
sr = 22050
os.makedirs("melody/songs", exist_ok=True)

# --- Generate Songs ---
for i in range(songs):
    key = random.choice(list(major_keys.keys()))
    scale = major_keys[key]
    tonic = scale[0]
    bpm = random.randint(80, 140)
    beat_duration = 60 / bpm

    song_len = random.randint(shortest, longest)
    melody_notes = generate_pleasant_melody(scale, tonic, song_len)

    song_dir = f"melody/songs/song_{i}"
    os.makedirs(song_dir, exist_ok=True)

    full_wave = []

    for j, note in enumerate(melody_notes):
        note_name = note + "4"
        freq = note_freqs.get(note_name)
        if freq is None:
            print(f"Skipping {note_name} (not found in freq table)")
            continue

        wave = generate_note(freq, duration=beat_duration, sr=sr)
        sf.write(f"{song_dir}/{j}_{note}.wav", wave, sr)
        full_wave.append(wave)

    # Save full melody
    song_audio = np.concatenate(full_wave)
    sf.write(f"{song_dir}/full_song.wav", song_audio, sr)

    # Save metadata
    with open(f"{song_dir}/metadata.txt", "w") as f:
        f.write(f"{key}\n{bpm}\n{' '.join(melody_notes)}\n")

