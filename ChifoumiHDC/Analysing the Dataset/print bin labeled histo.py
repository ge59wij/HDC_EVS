import os
import h5py
import numpy as np

FILE_PATH = input("Enter the path of the HDF5 file to check: ")

if not os.path.exists(FILE_PATH):
    print("Error: File not found.")
    exit()
with h5py.File(FILE_PATH, "r") as f:
    print(f"Loaded File: {FILE_PATH}")
    for key in f.keys():
        print(f"- {key}: {f[key].shape}, dtype={f[key].dtype}")

    event_data = f["data"][:]
    class_id = f["class_id"][()].item()
    labels = f["labels"][:]

T = event_data.shape[0]

print(f"\nDataset Info:")
print(f"- Data Shape: {event_data.shape} (T={T}, H={event_data.shape[2]}, W={event_data.shape[3]})")
print(f"- class_id: {class_id}")
print(f"- Labels Shape: {labels.shape}")

if labels.shape != (T,):
    print(f"Warning: Labels shape mismatch! Expected ({T},), got {labels.shape}")

print("\nBin Labels:")
for t in range(T):
    label = labels[t]
    label_status = "BACKGROUND" if label == 404 else f"Gesture {label}"
    print(f"Bin {t}: {label_status}")

background_bins = np.where(labels == 404)[0]
if len(background_bins) > 0:
    print("\nBackground Bins (404):", background_bins.tolist())
else:
    print("\nNo background bins found in this sample.")
