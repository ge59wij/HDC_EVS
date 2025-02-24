import os
import glob
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATASET_TYPE = "h5"  # "h5" or "pkl"
SPLIT = "train"
DATASET_FOLDER = f"/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/{SPLIT}" \
    if DATASET_TYPE == "h5" else f"/space/chair-nas/tosy/preprocessed_dat_chifoumi/{SPLIT}"
GESTURE_CLASS = None  #None for all 0 1 2
NUM_SAMPLES = 1000
SORT_BY_DURATION = True  # Shortest to longest
SKIP_FIRST_N = 0  # Skip first N samples
BIN_SIZE = 10000 if DATASET_TYPE == "pkl" else None  # Bin size for pkl only

sample_list = []
if DATASET_TYPE == "h5":
    all_files = sorted(glob.glob(f"{DATASET_FOLDER}/*.h5"))
    for h5_path in all_files:
        with h5py.File(h5_path, "r") as f:
            event_data = f["data"][:]
            class_id = f["class_id"][()]
        if GESTURE_CLASS is not None and class_id != GESTURE_CLASS:
            continue
        T = event_data.shape[0]  # Number of time bins
        sample_list.append((T, h5_path, event_data))

elif DATASET_TYPE == "pkl":
    all_files = sorted(glob.glob(f"{DATASET_FOLDER}/*.pkl"))
    for file_path in all_files:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        events, label = data[0], data[1]
        if GESTURE_CLASS is not None and label != GESTURE_CLASS:
            continue
        timestamps = events["t"]
        duration = np.max(timestamps) - np.min(timestamps)
        sample_list.append((duration, file_path))

if SORT_BY_DURATION:
    sample_list.sort(key=lambda x: x[0])

selected_samples = sample_list[SKIP_FIRST_N:SKIP_FIRST_N + NUM_SAMPLES]

dataset_label = f"{DATASET_TYPE.upper()} ({SPLIT})"
print(f"Using {len(selected_samples)} samples for class {GESTURE_CLASS if GESTURE_CLASS is not None else 'ALL'} from {dataset_label}:")
for sample in selected_samples:
    print(f"   {sample[1]} (Duration: {sample[0] / 1e6:.2f} s)" if DATASET_TYPE == "pkl" else f"   {sample[1]} (Time bins: {sample[0]})")

plt.figure(figsize=(12, 6))

for i, sample in enumerate(selected_samples):
    if DATASET_TYPE == "h5":
        T, file_path, event_data = sample
        event_counts = event_data.sum(axis=(1, 2, 3))
        bins = np.arange(T)  # Start at 0
        label_text = f"Sample {i + 1} ({T} bins)"
    else:
        duration, file_path = sample
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        events = data[0]
        timestamps = events["t"] - np.min(events["t"])
        bins = np.arange(0, np.max(timestamps), BIN_SIZE)
        event_counts, _ = np.histogram(timestamps, bins=bins)
        label_text = f"Sample {i + 1} ({duration / 1e6:.2f} s)"

    plt.plot(bins[:-1] if DATASET_TYPE == "pkl" else bins, event_counts, marker="o", linestyle="-", label=label_text)


plt.xlabel("Time Bins" if DATASET_TYPE == "h5" else "Time (µs)")
plt.ylabel("Total Event Count")
plt.title(f"Event Count Over Time for {'Gesture Class ' + str(GESTURE_CLASS) if GESTURE_CLASS is not None else 'All Gestures'} | {dataset_label}")
plt.legend(title="Sample Duration")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
