import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from metavision_core.event_io import EventsIterator
import random

DATASET_PATH = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT"
SAVE_DIR = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/"
os.makedirs(SAVE_DIR, exist_ok=True)

SPLITS = ["train", "val", "test"]
DELTA_T = 10000
MIN_EVENTS_PER_CHUNK = 2000  # Skip chunks with fewer than this many events

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

def process_and_save(split):
    file_pairs = get_file_pairs(DATASET_PATH, split)
    print(f"Processing {len(file_pairs)} files for {split} split...")

    for dat_file, bbox_file in tqdm(file_pairs):
        save_path = os.path.join(SAVE_DIR, split, os.path.basename(dat_file) + ".pkl")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            continue

        bbox_data = np.load(bbox_file, allow_pickle=True)
        start_time, end_time = int(bbox_data[0][0]), int(bbox_data[-1][0])
        class_id = int(bbox_data[0][5])

        print(f"Processing {dat_file}")
        print(f"BBox start_time: {start_time}, end_time: {end_time}")

        events = []
        first_valid_timestamp, last_valid_timestamp = None, None
        iterator = EventsIterator(input_path=dat_file, mode="delta_t", delta_t=DELTA_T)
        previous_timestamp = None

        for ev in iterator:
            if len(ev) == 0:
                continue  # Skip empty chunks immediately

            relevant = ev[(ev["t"] >= start_time) & (ev["t"] <= end_time)]
            if len(relevant) > 0:
                current_timestamp = relevant["t"].min()
                if previous_timestamp == current_timestamp:
                    print(f"Warning: Duplicate timestamp {current_timestamp} detected in chunks!")
                previous_timestamp = current_timestamp

                if first_valid_timestamp is None:
                    first_valid_timestamp = relevant["t"].min()
                last_valid_timestamp = relevant["t"].max()

                if len(relevant) < MIN_EVENTS_PER_CHUNK:
                    print(f"Skipping chunk with {len(relevant)} events (too few) at timestamp {current_timestamp}")
                    continue

                print(f"{len(relevant)} events found in this chunk at timestamp {current_timestamp}")
                events.append(np.stack((relevant["t"].astype(np.float32), relevant["x"], relevant["y"], relevant["p"]), axis=-1))

        if events:
            events = np.concatenate(events, axis=0)
            print(f"Total events collected: {events.shape[0]}")
            print(f"First valid event timestamp: {first_valid_timestamp}, Last valid event timestamp: {last_valid_timestamp}")
        else:
            print(f"No valid events found for {dat_file}, skipping save")
            continue

        with open(save_path, "wb") as f:
            pickle.dump((events, class_id), f)

        print(f"Successfully saved: {save_path}")

def load_and_check_pkl():
    random_split = random.choice(SPLITS)
    pkl_files = glob.glob(os.path.join(SAVE_DIR, random_split, "*.pkl"))

    if not pkl_files:
        print("No pickle files found")
        return

    random_file = random.choice(pkl_files)
    print(f"Loading and checking: {random_file}")

    with open(random_file, "rb") as f:
        events, class_id = pickle.load(f)

    print(f"Class ID: {class_id}")
    print(f"First 5 timestamps after loading: {events[:5, 0]}")

    if np.any(np.isinf(events[:, 0])):
        print("Error: Found inf timestamps in pickle file")
    else:
        print("Timestamps are valid after loading")

if __name__ == "__main__":
    for split in SPLITS:
        process_and_save(split)
    load_and_check_pkl()
