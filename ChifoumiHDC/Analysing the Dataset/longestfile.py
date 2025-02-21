import os
import pickle
import numpy as np
from glob import glob

def find_max_timestamp(dataset_path):
    max_timestamp = 0
    max_timestamp_file = None

    pkl_files = glob(os.path.join(dataset_path, "**/*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} .pkl files. Processing...")
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                events, _ = pickle.load(f)
            if events.size == 0:
                continue
            file_max_time = events["t"].max()
            if file_max_time > max_timestamp:
                max_timestamp = file_max_time
                max_timestamp_file = pkl_file
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    if max_timestamp_file:
        print("\n=== HIGHEST TIMESTAMP FOUND ===")
        print(f"Max Timestamp: {max_timestamp} ms")
        print(f"Corresponding File: {max_timestamp_file}")
    else:
        print("\nNo valid event data found.")

DATASET_PATH = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/test"
find_max_timestamp(DATASET_PATH)