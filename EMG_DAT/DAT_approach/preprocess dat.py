import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from metavision_core.event_io import EventsIterator

DATASET_PATH = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT"
SAVE_DIR = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/train2/"
os.makedirs(SAVE_DIR, exist_ok=True)

SPLITS = ["train"]
DELTA_T = 10000
MIN_EVENTS_PER_CHUNK = 10  # Skip chunks with fewer than this many events


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

    max_timestamp_after_normalization = 0

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
        first_valid_timestamp = None
        iterator = EventsIterator(input_path=dat_file, mode="delta_t", delta_t=DELTA_T)

        for ev in iterator:
            if len(ev) == 0:
                continue  # Skip empty chunks immediately

            relevant = ev[(ev["t"] >= start_time) & (ev["t"] <= end_time)]
            if len(relevant) > 0:
                if first_valid_timestamp is None:
                    first_valid_timestamp = relevant["t"].min()

                relevant["t"] -= first_valid_timestamp  # Normalize timestamps to start from 0

                max_timestamp_after_normalization = max(max_timestamp_after_normalization, relevant["t"].max())

                if len(relevant) < MIN_EVENTS_PER_CHUNK:
                    continue

                structured_dtype = np.dtype([("t", "<f4"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])
                structured_events = np.array(list(zip(relevant["t"], relevant["x"], relevant["y"], relevant["p"])),
                                             dtype=structured_dtype)
                events.append(structured_events)

        if events:
            events = np.concatenate(events, axis=0)
            print(f"Total events collected: {events.shape[0]}")
            print(f"First event timestamp after normalization: {events[0]['t']}")
        else:
            print(f"No valid events found for {dat_file}, skipping save")
            continue

        with open(save_path, "wb") as f:
            pickle.dump((events, class_id), f)

        print(f"Successfully saved: {save_path}")

    print(f"Max timestamp after normalization across all files: {max_timestamp_after_normalization}")


if __name__ == "__main__":
    for split in SPLITS:
        process_and_save(split)
