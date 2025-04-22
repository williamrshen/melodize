import os
import numpy as np
from collections import Counter
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time


# --- Model Class Index Mapping (same order as model was trained on)
index_to_label = {
    0: 'A#4', 1: 'A4', 2: 'B4', 3: 'C#4', 4: 'C4', 5: 'C5',
    6: 'D#4', 7: 'D4', 8: 'E4', 9: 'F#4', 10: 'F4', 11: 'G#4', 12: 'G4'
}
note_labels = [index_to_label[i] for i in range(len(index_to_label))]


overall_avgs = []


def predict_chunks(folder_path, model_path="oneNote/note_model.h5", sr=22050):
    model = load_model(model_path)
    img_size = (128, 128)

    chunk_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".wav")
    ])

    predictions = []
    total_time = 0

    for chunk_file in chunk_files:
        start_time = time.time()

        full_path = os.path.join(folder_path, chunk_file)
        audio, _ = sf.read(full_path)

        # Stretch/compress to 1 second (22050 samples)
        audio_fixed = librosa.util.fix_length(audio, size=sr)

        # Create mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio_fixed, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Plot spectrogram to image
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis=None, y_axis=None, fmax=8000)
        plt.axis('off')
        plt.tight_layout(pad=0)
        img_path = "temp_img.png"
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load image and preprocess
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array, verbose=0)
        note_index = note_labels[np.argmax(pred)]
        predictions.append(note_index)

        total_time += time.time() - start_time

    avg_time = total_time / len(chunk_files) if chunk_files else 0
    print(f"Average time per note: {avg_time:.4f} seconds")
    overall_avgs.append(avg_time)

    return predictions



def get_note_name(note):
    return note[:-1] 

def get_note_frequencies(notes_seq):
    return Counter(get_note_name(note) for note in notes_seq)

def detect_key(notes_seq):
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

    note_counts = get_note_frequencies(notes_seq)

    def score(key_name, key_notes):
        s = 0
        for note, count in note_counts.items():
            if note in key_notes:
                s += count
            else:
                s -= count * 0.8  # penalize out-of-key notes
        s += note_counts.get(key_name, 0) * 3  # bonus for tonic note
        return s

    best_key = max(major_keys.items(), key=lambda item: score(item[0], item[1]))
    return best_key[0]



index_to_label = {
    0: 'A#4', 1: 'A4', 2: 'B4', 3: 'C#4', 4: 'C4', 5: 'C5',
    6: 'D#4', 7: 'D4', 8: 'E4', 9: 'F#4', 10: 'F4', 11: 'G#4', 12: 'G4'
}
note_labels = [index_to_label[i] for i in range(len(index_to_label))]

for song_folder in os.listdir("melody/songs"):
    prediction_folder = f"melody/predictions/{song_folder}"
    print(f"Predicting notes in: {song_folder}")
    note_predictions = predict_chunks(prediction_folder)
    predicted_key = detect_key(note_predictions)
    with open(f"melody/predictions/{song_folder}/predictions.txt", "w") as f:
        f.write(f"{predicted_key}\n{' '.join(note[:-1] for note in note_predictions)}\n")

print(sum(overall_avgs) / len(overall_avgs))
