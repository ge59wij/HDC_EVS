import pickle
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random

dataset_folder = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/val/"
gesture_class = 1
num_samples = 30
skip_first_n = 8
sort_by_duration = True
bin_size = 10000  # (10000=10ms)

all_files = sorted(glob.glob(f"{dataset_folder}/*.pkl"))
sample_list = []

for file_path in all_files:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    events, label = data[0], data[1]
    if gesture_class is not None and label != gesture_class:
        continue
    try:
        timestamps = events["t"]
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")
        continue
    timestamps = events["t"]
    duration = np.max(timestamps) - np.min(timestamps)

    sample_list.append((duration, file_path))

random.shuffle(sample_list)

if sort_by_duration:
    sample_list.sort(key=lambda x: x[0])  # reverse=True

selected_samples = sample_list[skip_first_n:skip_first_n + num_samples]

print(f"Using {len(selected_samples)} samples for class {gesture_class}:")
for dur, fname in selected_samples:
    print(f"   {fname} (Duration: {dur / 1e6:.2f} s)")

cmap = plt.colormaps["gist_ncar"]
plt.style.use("dark_background")
plt.figure(figsize=(12, 6))

for i, (duration, file_path) in enumerate(selected_samples):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    events = data[0]
    timestamps = events["t"]
    timestamps -= np.min(timestamps)

    bins = np.arange(0, np.max(timestamps), bin_size)

    event_counts, _ = np.histogram(timestamps, bins=bins)
    color = cmap(i / (num_samples - 1))
    plt.plot(bins[:-1], event_counts, marker="o", linestyle="-", color=color,
            label=f"Sample {i + 1} ({duration / 1e6:.2f}s)")

plt.xlabel("Time (µs)")
plt.ylabel("Event Count (per bin)")
plt.title(f"Event Count Over Time for Gesture Class {gesture_class} (Bin Size: {bin_size} µs)")
plt.legend(title="Sample Duration")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
