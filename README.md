# 🎵 Melodize

**Melodize** is a machine learning project that detects musical notes and keys from audio files using mel spectrograms. It leverages synthetic data, deep learning, and audio signal processing to train a model capable of identifying individual notes and key signatures.

## 🚀 Features

- 🎶 Generates synthetic audio samples for all notes in a defined pitch range
- 📊 Converts audio waveforms into 128×128 mel spectrogram images
- 🧠 Trains a Convolutional Neural Network (CNN) using PyTorch
- 🧪 Predicts notes and keys from unseen audio segments
- 📈 Evaluates model accuracy, inference time, and key prediction performance

## 🧠 Model Performance

| Metric                   | Result                    |
|--------------------------|---------------------------|
| Note Prediction Accuracy | **100% (1464/1464)**      |
| Key Prediction Accuracy  | **84% (42/50)**           |
| Avg Inference Time       | **0.445 seconds/note**    |

## 🛠️ Tech Stack

- Python (NumPy, SoundFile, Librosa)
- Tensorflow (CNNs for image classification)
- Matplotlib (for visualization)
- Time (evaluation metrics)



