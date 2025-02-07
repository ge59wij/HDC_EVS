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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure split folder exists

        if os.path.exists(save_path):
            continue  # Skip if already preprocessed

        # Load bbox info
        bbox_data = np.load(bbox_file, allow_pickle=True)
        start_time, end_time = int(bbox_data[0][0]), int(bbox_data[-1][0])
        class_id = int(bbox_data[0][5])

        # Load and filter event data
        events = []
        iterator = EventsIterator(input_path=dat_file, mode="delta_t", delta_t=DELTA_T)
        for ev in iterator:
            relevant = ev[(ev["t"] >= start_time) & (ev["t"] <= end_time)]
            events.append(np.stack((relevant["t"], relevant["x"], relevant["y"], relevant["p"]), axis=-1))

        if events:
            events = np.concatenate(events, axis=0)
        else:
            events = np.empty((0, 4))  # Empty if no events

        # Save processed events and label
        with open(save_path, "wb") as f:
            pickle.dump((events, class_id), f)

        print(f"Saved: {save_path}")

for split in SPLITS:
    process_and_save(split)
