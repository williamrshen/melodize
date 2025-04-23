import os

metadata_dir = "melody/songs"
predictions_dir = "melody/predictions"

correct_keys = 0
total_keys = 0

total_notes = 0
correct_notes = 0

for i in range(100):  # from 0 to 10
    metadata_path = os.path.join(metadata_dir, f"song_{i}", "metadata.txt")
    prediction_path = os.path.join(predictions_dir, f"song_{i}", "predictions.txt")

    if not os.path.exists(metadata_path) or not os.path.exists(prediction_path):
        print(f"Missing file(s) for song_{i}")
        continue

    with open(metadata_path, "r") as f:
        true_key = f.readline().strip()
        true_bpm = f.readline().strip()  # unused, but can be read
        true_notes = f.readline().strip().split()

    with open(prediction_path, "r") as f:
        predicted_key = f.readline().strip()
        predicted_notes = f.readline().strip().split()

    # Compare keys
    total_keys += 1
    if true_key == predicted_key:
        correct_keys += 1
    else:
        print(f"Key mismatch for song_{i}: {true_key} vs {predicted_key}")

    # Compare notes
    length = min(len(true_notes), len(predicted_notes))
    total_notes += length
    for t, p in zip(true_notes[:length], predicted_notes[:length]):
        if t == p:
            correct_notes += 1

# Summary
print(f"Correct keys: {correct_keys}/{total_keys}")
print(f"Correct notes: {correct_notes}/{total_notes} ({(correct_notes / total_notes * 100):.2f}%)")
