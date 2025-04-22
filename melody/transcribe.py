import os
import soundfile as sf
import librosa

def split_audio_by_bpm(file_path, bpm, output_folder, sr=22050):
    os.makedirs(output_folder, exist_ok=True)

    # Load the full audio
    audio, samplerate = sf.read(file_path)
    
    # Calculate beat duration in samples
    beat_duration_sec = 60 / bpm
    beat_samples = int(beat_duration_sec * samplerate)

    total_samples = len(audio)
    num_chunks = total_samples // beat_samples

    for i in range(num_chunks):
        start = i * beat_samples
        end = start + beat_samples
        chunk = audio[start:end]

        chunk_filename = os.path.join(output_folder, f"note_{i:03d}.wav")
        sf.write(chunk_filename, chunk, samplerate)

    print(f"Split into {num_chunks} chunks of {beat_duration_sec:.2f} seconds each.")

def stretch_audio(input_path, output_path, target_duration=1.0):
    y, sr = librosa.load(input_path, sr=None)
    original_duration = librosa.get_duration(y=y, sr=sr)
    
    if original_duration == 0:
        print(f"Skipping {input_path} (empty file).")
        return

    stretch_rate = original_duration / target_duration
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, y_stretched, sr)


for song_folder in os.listdir("melody/songs"):
    with open(f"melody/songs/{song_folder}/metadata.txt", "r") as f:
        key = f.readline().strip()
        bpm = int(f.readline().strip())
        melody = f.readline().strip().split(" ")
    split_audio_by_bpm(f"melody/songs/{song_folder}/full_song.wav", bpm, output_folder=f"melody/predictions/{song_folder}")
