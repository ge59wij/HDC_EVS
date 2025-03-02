import h5py
import numpy as np
import os

# Path to test dataset
test_dataset_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/test/"

# Initialize statistics
min_event = float("inf")
max_event = float("-inf")
total_sum = 0
total_bins = 0
unique_values = set()

# Iterate over all HDF5 files in the test dataset
for filename in os.listdir(test_dataset_path):
    if filename.endswith(".h5"):
        file_path = os.path.join(test_dataset_path, filename)

        with h5py.File(file_path, "r") as f:
            # Extract event data (ON/OFF events per bin)
            event_data = f["data"][:]  # Shape: (T, 2, H, W)

            # Compute min, max, and sum of events
            min_event = min(min_event, event_data.min())
            max_event = max(max_event, event_data.max())
            total_sum += event_data.sum()
            total_bins += event_data.shape[0]  # T bins

            # Store unique values (rounded to 4 decimals for clarity)
            unique_values.update(np.unique(event_data.round(4)))

# Compute average event count per bin
average_event = total_sum / total_bins if total_bins > 0 else 0

# Print results
print(f"\n[SUMMARY OF TEST DATASET]")
print(f"  - Minimum Event Count: {min_event}")
print(f"  - Maximum Event Count: {max_event}")
print(f"  - Average Event Count per Bin: {average_event:.2f}")
print(f"  - Number of Unique Values: {len(unique_values)}")
print(f"  - Sample Unique Values (Rounded): {sorted(unique_values)[:20]}")
