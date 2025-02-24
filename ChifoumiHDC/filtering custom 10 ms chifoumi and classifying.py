import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
#'''
'''
"""
HDF5 Gesture Dataset Filtering & Processing
-------------------------------------------------
- Loading HDF5 files containing event histograms (histo_quantized).
- Using corresponding NPY bounding box files to determine relevant timestamps per sample, filtering out time bins to retain only gesture-relevant ones.
- Assigning a class label based on the NPY metadata. 5th element in any row. 

- Log metadata ONLY for 'val' and 'test' splits.
- Processed HDF5 files are stored in '/processed/{split}/'.
- For 'val' and 'test', a 'processing_log.txt' is generated, listing:
  - Original & Processed HDF5 paths
  - Class ID of each sample
  - Original and filtered tensor shapes
  - Processing details for debugging
NB:
- script assumes h5 files are quantized histograms with shape [T, 2, H, W] which matching _bbox.npy files.
- The time bin size is 10ms, and timestamps from NPY files are converted accordingly. class id in 5th el. 
'''

SPLITS = ["train", "val", "test"]

H5_BASE_FOLDER = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi"
OUTPUT_BASE_FOLDER = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed"

for split in SPLITS:
    os.makedirs(os.path.join(OUTPUT_BASE_FOLDER, split), exist_ok=True)

for split in SPLITS:
    print(f"\nProcessing split: {split}")

    H5_INPUT_FOLDER = os.path.join(H5_BASE_FOLDER, split)
    OUTPUT_FOLDER = os.path.join(OUTPUT_BASE_FOLDER, split)

    h5_files = sorted([f for f in os.listdir(H5_INPUT_FOLDER) if f.endswith(".h5")])

    # Only create a log file for val and test splits
    log_file_path = os.path.join(OUTPUT_FOLDER, "processing_log.txt") if split in ["val", "test"] else None
    if log_file_path:
        with open(log_file_path, "w") as log_file:
            log_file.write(f"Processed HDF5 Files for {split}\n")
            log_file.write(f"{'=' * 50}\n\n")

    for h5_file in tqdm(h5_files, desc=f"Processing {split} files"):
        h5_path = os.path.join(H5_INPUT_FOLDER, h5_file)
        npy_path = h5_path.replace(".h5", "_bbox.npy")

        if not os.path.exists(npy_path):
            print(f"Skipping {h5_file}: No corresponding bbox file.")
            continue

        with h5py.File(h5_path, "r") as f:
            event_data = torch.tensor(f["data"][:], dtype=torch.float32)

        bbox_data = np.load(npy_path, allow_pickle=True)
        if bbox_data.dtype.names is not None:
            bbox_data = np.column_stack([bbox_data[name] for name in bbox_data.dtype.names])

        start_ts, end_ts = int(bbox_data[0][0]), int(bbox_data[-1][0])
        class_id = int(bbox_data[0][5])

        start_bin = int(start_ts / 10_000)  # Convert from microseconds to 10ms bins
        end_bin = int(end_ts / 10_000)  # Convert from microsecond to 10ms bins

        start_bin = max(0, start_bin)
        end_bin = min(event_data.shape[0], end_bin)

        filtered_events = event_data[start_bin:end_bin]

        output_path = os.path.join(OUTPUT_FOLDER, h5_file)
        with h5py.File(output_path, "w") as f:
            f.create_dataset("data", data=filtered_events.numpy())
            f.create_dataset("class_id", data=np.array([class_id], dtype=np.int32))

        if log_file_path:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Original HDF5: {h5_path}\n")
                log_file.write(f"Processed HDF5: {output_path}\n")
                log_file.write(f"Class ID: {class_id}\n")
                log_file.write(f"Original Shape: {event_data.shape}\n")
                log_file.write(f"Filtered Shape: {filtered_events.shape}\n")
                log_file.write("-" * 50 + "\n")

        print(f"Saved {output_path} | Shape: {filtered_events.shape} | Class: {class_id}")

print("Processing complete!")



'''

import h5py
import os
import torch


ORIGINAL_H5_DIR = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/train/"
PROCESSED_H5_DIR = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/"


FILENAME = "paper_02P_0_0_1800000_3300000.h5"  # Change this to test different files


original_h5_path = os.path.join(ORIGINAL_H5_DIR, FILENAME)
processed_h5_path = os.path.join(PROCESSED_H5_DIR, FILENAME)


if not os.path.exists(original_h5_path):
    print(f"Original file not found: {original_h5_path}")
    exit()
if not os.path.exists(processed_h5_path):
    print(f"Processed file not found: {processed_h5_path}")
    exit()

with h5py.File(original_h5_path, "r") as f:
    original_data = torch.tensor(f["data"][:])

with h5py.File(processed_h5_path, "r") as f:
    processed_data = torch.tensor(f["data"][:])

print(f"Original: {original_data.shape} (T={original_data.shape[0]})")
print(f"Processed: {processed_data.shape} (T={processed_data.shape[0]})")

# If processed T is shorter, filtering worked!
if processed_data.shape[0] < original_data.shape[0]:
    print(" Filtering removed some time bins!")
else:
    print(" No reduction in T, check filtering logic!")

# Print first and last few time bins to confirm shortening
print("\nOriginal First 5 T bins:")
print(original_data[:5])
print("\nOriginal Last 5 T bins:")
print(original_data[-5:])

print("\nProcessed First 5 T bins:")
print(processed_data[:5])
print("\nProcessed Last 5 T bins:")
print(processed_data[-5:])
#'''