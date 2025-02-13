import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from metavision_core.event_io import EventsIterator

# --- Paths ---
DATASET_PATH = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT"
SAVE_DIR = "/space/chair-nas/tosy/pt_chifoumi/"
os.makedirs(SAVE_DIR, exist_ok=True)

SPLITS = ["train"]
DELTA_T = 10000
MIN_EVENTS_PER_CHUNK = 10  # Skip chunks with fewer than this many events

# --- Get File Pairs ---
def get_file_pairs(root_dir, split):
    split_dir = os.path.join(root_dir, split)
    pairs = []
    all_files = sorted(glob.glob(os.path.join(split_dir, '*_bbox.npy')))

    for bbox_file in all_files:
        base_name = bbox_file.replace('_bbox.npy', '')
        dat_file_cd = base_name + "_cd.dat"
        dat_file_td = base_name + "_td.dat"

        if os.path.exists(dat_file_cd):
            pairs.append((dat_file_cd, bbox_file))
        elif os.path.exists(dat_file_td):
            pairs.append((dat_file_td, bbox_file))

    return pairs

# --- Processing & Saving ---
def process_and_save(split):
    file_pairs = get_file_pairs(DATASET_PATH, split)
    print(f"Processing {len(file_pairs)} files for {split} split...")

    max_timestamp_after_normalization = 0

    for dat_file, bbox_file in tqdm(file_pairs):
        save_path = os.path.join(SAVE_DIR, split, os.path.basename(dat_file) + ".pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            continue  # Skip if already processed

        bbox_data = np.load(bbox_file, allow_pickle=True)
        start_time, end_time = int(bbox_data[0][0]), int(bbox_data[-1][0])
        class_id = int(bbox_data[0][5])

        print(f"Processing {dat_file}")
        print(f"BBox start_time: {start_time}, end_time: {end_time}")

        events = []
        first_valid_timestamp = None
        iterator = EventsIterator(input_path=dat_file, mode="delta_t", delta_t=DELTA_T)

        for ev in iterator:
            if len(ev) == 0:
                continue  # Skip empty chunks

            relevant = ev[(ev["t"] >= start_time) & (ev["t"] <= end_time)]
            if len(relevant) > 0:
                if first_valid_timestamp is None:
                    first_valid_timestamp = relevant["t"].min()

                relevant["t"] -= first_valid_timestamp  # Normalize timestamps to start from 0

                max_timestamp_after_normalization = max(max_timestamp_after_normalization, relevant["t"].max())

                if len(relevant) < MIN_EVENTS_PER_CHUNK:
                    continue

                # Convert directly to PyTorch tensor
                events_tensor = torch.tensor(
                    np.column_stack((relevant["t"], relevant["x"], relevant["y"], relevant["p"])),
                    dtype=torch.float32
                )
                events.append(events_tensor)

        if events:
            events_tensor = torch.cat(events, dim=0)  # Stack all events in one tensor
            print(f"Total events collected: {events_tensor.shape[0]}")
            print(f"First event timestamp after normalization: {events_tensor[0, 0]}")
        else:
            print(f"No valid events found for {dat_file}, skipping save")
            continue

        # Ensure label is correctly formatted as a tensor
        label_tensor = torch.tensor(class_id, dtype=torch.long)
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)  # Convert scalar to tensor

        # Save `.pt` file
        torch.save({"events": events_tensor, "label": label_tensor}, save_path)
        print(f"Successfully saved: {save_path}")

    print(f" Max timestamp after normalization across all files: {max_timestamp_after_normalization}")

# --- Run the script ---
if __name__ == "__main__":
    for split in SPLITS:
        process_and_save(split)
