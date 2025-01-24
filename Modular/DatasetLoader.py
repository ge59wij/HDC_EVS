import os
import glob
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class EventDatasetLoader(Dataset):
    def __init__(self, root_dir, split, max_time):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.max_time = max_time
        self.label_map = self._load_label_map()
        self.num_classes = len(self.label_map)
        self.h5_files = sorted(glob.glob(os.path.join(self.split_dir, '*.h5')))
        self.file_pairs = self._get_file_pairs()

    def _load_label_map(self):
        json_path = os.path.join(self.root_dir, "label_map_dictionary.json")
        with open(json_path, "r") as f:
            return {str(k): v for k, v in json.load(f).items()}

    def _get_file_pairs(self):
        pairs = []
        for h5_file in self.h5_files:
            npy_file = h5_file.replace('.h5', '_bbox.npy')
            if os.path.exists(npy_file):
                pairs.append((h5_file, npy_file))
        return pairs

    def __getitem__(self, idx):
        h5_path, npy_path = self.file_pairs[idx]
        with h5py.File(h5_path, 'r') as f:
            tensor_data = f['data'][:]
        bbox_data = np.load(npy_path, allow_pickle=True)
        class_id = int(str(int(bbox_data[0][5])))

        return torch.tensor(tensor_data, dtype=torch.float32), torch.tensor(class_id)



    def __len__(self):
        return len(self.file_pairs)
