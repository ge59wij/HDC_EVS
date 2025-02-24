import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
'''H5 form prophesee!!'''
dataset_folder = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/"
gesture_class = 2
num_samples = 60
sort_by_duration = True  #shortest to longest

all_h5_files = sorted(glob.glob(f"{dataset_folder}/*.h5"))
sample_list = []

file_pairs = []
for h5_file in all_h5_files:
    npy_file = h5_file.replace('.h5', '_bbox.npy')
    if os.path.exists(npy_file):
        file_pairs.append((h5_file, npy_file))

for h5_path, npy_path in file_pairs:
    bbox_data = np.load(npy_path, allow_pickle=True)
    class_id = int(bbox_data[0][5])
    if class_id != gesture_class:
        continue

    with h5py.File(h5_path, 'r') as f:
        tensor_data = f['data'][:]  # Shape: [T, 2, 120, 160]

    T = tensor_data.shape[0]  # Number of time bins (duration estimate)

    sample_list.append((T, h5_path, tensor_data))

if sort_by_duration:
    sample_list.sort(key=lambda x: x[0])

selected_samples = sample_list[:num_samples]

print(f"Using {len(selected_samples)} samples for class {gesture_class}:")
for T, fname, _ in selected_samples:
    print(f"  - {fname} (Time bins: {T})")

plt.figure(figsize=(12, 6))

for i, (T, h5_path, tensor_data) in enumerate(selected_samples):
    # Sum over spatial dimensions and polarities: [T, 2, 120, 160] =>[T]
    event_counts = tensor_data.sum(axis=(1, 2, 3))
    # Time bins (X-axis)
    bins = np.arange(1, T + 1)
    plt.plot(bins, event_counts, marker="o", linestyle="-", label=f"Sample {i+1} ({T} bins)")
plt.xlabel("Time Bins (each = 1000 µs)")
plt.ylabel("Total Event Count")
plt.title(f"Event Count Over Time for Gesture Class {gesture_class}")
plt.legend(title="Sample Duration (bins)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
