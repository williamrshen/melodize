import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- Settings ---
num_tests = 100
output_dir = "oneNote/test_batch"
img_size = (128, 128)
sr = 22050

# --- Notes and Frequencies ---
notes = {
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63,
    "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00,
    "A#4": 466.16, "B4": 493.88, "C5": 523.25
}

# --- Model Class Index Mapping (same order as model was trained on)
index_to_label = {
    0: 'A#4', 1: 'A4', 2: 'B4', 3: 'C#4', 4: 'C4', 5: 'C5',
    6: 'D#4', 7: 'D4', 8: 'E4', 9: 'F#4', 10: 'F4', 11: 'G#4', 12: 'G4'
}
note_labels = [index_to_label[i] for i in range(len(index_to_label))]

# --- Utilities ---
def generate_note(freq, duration=1.0, sr=22050, detune_cents=50):
    """
    Generate a sine wave with slight frequency variation (detuning).
    `detune_cents` is the maximum shift in cents (100 cents = 1 semitone).
    """
    # Convert detune in cents to frequency ratio
    cents_shift = np.random.uniform(-detune_cents, detune_cents)
    detune_ratio = 2 ** (cents_shift / 1200.0)
    varied_freq = freq * detune_ratio

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * varied_freq * t)
    return wave


# --- Init ---
os.makedirs(output_dir, exist_ok=True)
model = load_model("oneNote/note_model.h5")

# --- Accuracy Tracking ---
correct = 0
y_true = []
y_pred = []

# --- Run Tests ---
for i in range(num_tests):
    true_note = random.choice(note_labels)
    freq = notes[true_note]

    # 1. Generate audio
    wav_path = os.path.join(output_dir, f"{true_note}_test_{i}.wav")
    wave = generate_note(freq)
    sf.write(wav_path, wave, sr)

    # 2. Convert to mel spectrogram
    y, sr = librosa.load(wav_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    img_path = wav_path.replace(".wav", ".png")
    plt.figure(figsize=(2, 2))
    librosa.display.specshow(mel_db, sr=sr, fmax=8000)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # 3. Predict
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    predicted_note = note_labels[np.argmax(pred)]

    # 4. Evaluate
    is_correct = predicted_note == true_note
    if is_correct:
        correct += 1

    y_true.append(true_note)
    y_pred.append(predicted_note)

    print(f"[{i+1}/{num_tests}] True: {true_note}, Predicted: {predicted_note} — {'✅' if is_correct else '❌'}")

# --- Final Accuracy ---
accuracy = correct / num_tests * 100
print(f"\n🎯 Final Accuracy: {accuracy:.2f}% ({correct}/{num_tests})")

# --- (Optional) Confusion Matrix ---
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred, labels=note_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=note_labels)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
