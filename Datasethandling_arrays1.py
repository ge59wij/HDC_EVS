import os
import glob
import h5py
import json
import numpy as np


class EventDatasetLoader:
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.label_map = self._load_label_map()
        self.h5_files = glob.glob(os.path.join(self.split_dir, '*.h5'))
        self.file_pairs = self._get_file_pairs()
        print(f"Found {len(self.file_pairs)} pairs in {self.split_dir}.")

    def _load_label_map(self):
        """
        Loads label_map_dictionary.json.: {'0': 'paper', '1': 'rock', '2': 'scissor'}
        Returns: dict: Mapping of label names to class IDs (integers)
        """
        json_path = os.path.join(self.root_dir, "label_map_dictionary.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Label map file not found: {json_path}")
        with open(json_path, "r") as f:
            label_map = json.load(f)
        print(f"Loaded label map: {label_map}")
        return label_map

    def _get_file_pairs(self):
        """
        Returns: List of tuples: [(h5_file, npy_file), ...]
        """
        pairs = []
        for h5_file in self.h5_files:
            npy_file = h5_file.replace('.h5', '_bbox.npy')
            if os.path.exists(npy_file):
                pairs.append((h5_file, npy_file))
        return pairs

    def load_sample(self, h5_path, npy_path):
        """
        Loads a single sample from an HDF5 file and its corresponding NPY file
        Returns: tuple: (tensor_data, label_id)
        """
        # Load event tensor
        with h5py.File(h5_path, 'r') as f:
            tensor_data = f['data'][:]

        # Load bbox data
        bbox_data = np.load(npy_path, allow_pickle=True)
        if len(bbox_data) == 0:
            raise ValueError(f"Empty bbox data in {npy_path}")

        # Extract class name and map it to class ID
        class_name = str(int(bbox_data[0][5]))   #class name is the 6*th element
        class_id = self.label_map.get(class_name, -1)  # Default to -1 if not found
        if class_id == -1:
            raise ValueError(f"Class {class_name} not found in label map!")
        class_id = int(class_name)    # Convert Class string to int for pytensors
        return tensor_data, class_id

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        h5_path, npy_path = self.file_pairs[idx]
        return self.load_sample(h5_path, npy_path)


if __name__ == "__main__":
    dataset_path = ("/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/")
    train_loader = EventDatasetLoader(dataset_path, "train")


    ''''
    # Iterate through samples (debug)
    for i in range(len(train_loader)):
        data, label = train_loader[i]
        print(f"Sample {i}: Data shape {data.shape}, Label {label}")
    #'''