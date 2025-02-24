"""
Prophesee Event-Based .dat File Processing for GraphHD Encoding

This module processes event-based data stored in Prophesee's .dat format and converts it
into a structured format suitable for GraphHD encoding. It also filters relevant events
using bounding box (.npy) labels to extract only the necessary data for gesture classification.

### Features:
- Handles both "_cd.dat" and "_td.dat" file formats automatically.
- Filters events based on gesture occurrence time from the bounding box labels.
- Returns events in (timestamp, (x, y), polarity) format for direct use in GraphHD encoding.
(t, (x, y), p)

### Example Usage:
from DAT_loadergrasp import EventDatasetLoader

# Define dataset path and split (e.g., "train", "val", "test")
dataset = EventDatasetLoader(root_dir="/space/chair-nas/tosy/Gen3_Chifoumi_DAT", split="train")

# Get a sample (returns filtered events and corresponding class ID)
filtered_events, class_id = dataset[0]

print(f"Class ID: {class_id}")
print(f"First 5 Events: {filtered_events[:5]}")
"""

import os
import numpy as np
import glob
from torch.utils.data import Dataset
from metavision_core.event_io import EventsIterator
import random


class GRASP_DAT_EventLoader(Dataset):
    def __init__(self, root_dir, split, delta_t=10000, shuffle= True ):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.delta_t = 10000 # to not kill memory
        self.file_pairs = self._get_file_pairs()
        if shuffle: random.shuffle(self.file_pairs)


    def _get_file_pairs(self):
        pairs = []
        all_files = sorted(glob.glob(os.path.join(self.split_dir, '*_bbox.npy')))
        for bbox_file in all_files:
            base_name = bbox_file.replace('_bbox.npy', '')
            dat_file_cd = base_name + "_cd.dat"
            dat_file_td = base_name + "_td.dat"

            if os.path.exists(dat_file_cd):
                pairs.append((dat_file_cd, bbox_file))
            elif os.path.exists(dat_file_td):
                pairs.append((dat_file_td, bbox_file))
        return pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        dat_file, bbox_file = self.file_pairs[idx]
        bbox_data = np.load(bbox_file, allow_pickle=True)
        start_time = int(bbox_data[0][0])  # First row, first element bound
        end_time = int(bbox_data[-1][0])  # Last row, first element
        class_id = int(bbox_data[0][5])  # Class id
        # Load and filter
        filtered_events = []
        iterator = EventsIterator(input_path=dat_file, mode="delta_t", delta_t=self.delta_t)
        for events in iterator:
            relevant_events = events[(events["t"] >= start_time) & (events["t"] <= end_time)]
            formatted_events = [(t, (x, y), p) for x, y, p, t in relevant_events]  # GraspHD p is -1 1, ours 0 1, should the same.
            filtered_events.extend(formatted_events)

        return filtered_events, class_id
