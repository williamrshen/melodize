import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# Input and output folders
input_base = "oneNote/data"
output_base = "oneNote/mel_images"

# Create output directory
os.makedirs(output_base, exist_ok=True)

# Loop through notes (folders)
for note_folder in os.listdir(input_base):
    note_path = os.path.join(input_base, note_folder)
    if not os.path.isdir(note_path):
        continue

    output_note_path = os.path.join(output_base, note_folder)
    os.makedirs(output_note_path, exist_ok=True)

    # Loop through .wav files
    for file in os.listdir(note_path):
        if not file.endswith(".wav"):
            continue

        input_file = os.path.join(note_path, file)
        output_file = os.path.join(output_note_path, file.replace(".wav", ".png"))

        # Load audio
        y, sr = librosa.load(input_file)

        # Create mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Plot and save
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, fmax=8000)
        plt.axis('off')  # no axis for clean image
        plt.tight_layout(pad=0)
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()
