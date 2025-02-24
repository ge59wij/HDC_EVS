import os
import glob
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict
import tabulate
from tabulate import tabulate


class EventDatasetLoader(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str,
                 max_time: int = 150,  ###########
                 transform=None) -> None:
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.max_time = max_time
        self.transform = transform
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load label map and files
        self.label_map = self._load_label_map()
        self.h5_files = sorted(glob.glob(os.path.join(self.split_dir, '*.h5')))
        self.file_pairs = self._get_file_pairs()

        print(f"Found {len(self.file_pairs)} pairs in {self.split_dir}")
        print(f"Label map: {self.label_map}")

    def _load_label_map(self) -> Dict[str, str]:
        json_path = os.path.join(self.root_dir, "label_map_dictionary.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Label map file not found: {json_path}")

        try:
            with open(json_path, "r") as f:
                label_map = json.load(f)
                return {str(k): v for k, v in label_map.items()}
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {json_path}")

    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for h5_file in self.h5_files:
            npy_file = h5_file.replace('.h5', '_bbox.npy')
            if os.path.exists(npy_file):
                pairs.append((h5_file, npy_file))
            else:
                print(f"Warning: Missing NPY file for {h5_file}")
        return pairs

    def load_sample(self, h5_path: str, npy_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Load event tensor
            with h5py.File(h5_path, 'r') as f:
                tensor_data = f['data'][:]  # Shape [T, 2, H, W]

            # Load bbox data
            bbox_data = np.load(npy_path, allow_pickle=True)
            if len(bbox_data) == 0:
                raise ValueError(f"Empty bbox data in {npy_path}")

            # Extract and validate class
            class_name = str(int(bbox_data[0][5]))
            if class_name not in self.label_map:
                raise ValueError(f"Class {class_name} not found in label map!")
            class_id = int(class_name)

            # Handle time dimension
            T, C, H, W = tensor_data.shape
            if T < self.max_time:
                padding = np.zeros((self.max_time - T, C, H, W), dtype=tensor_data.dtype)
                tensor_data = np.concatenate((tensor_data, padding), axis=0)
            elif T > self.max_time:
                tensor_data = tensor_data[:self.max_time]

            # Convert to tensor
            tensor_data = torch.tensor(tensor_data, dtype=torch.float32)

            return tensor_data, torch.tensor(class_id, dtype=torch.long)

        except Exception as e:
            print(f"Error loading {h5_path}: {str(e)}")
            raise

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h5_path, npy_path = self.file_pairs[idx]
        return self.load_sample(h5_path, npy_path)
def print_summary_table(train_dataset, val_dataset, test_dataset, batch_size, num_epochs, dimensions, height, width):
    table = [
        ["Training Samples", len(train_dataset)],
        ["Validation Samples", len(val_dataset)],
        ["Test Samples", len(test_dataset)],
        ["Batch Size", batch_size],
        ["Number of Epochs", num_epochs],
        ["Hypervector Dimensions", dimensions],
        ["Frame HeightxWidth", height, width],
    ]
    print("\nDataset and Training Configuration Summary:")
    print(tabulate(table))

    #train_loader = EventDatasetLoader(dataset_path, "train") #train, test, val

    '''
    # Debugging:
    indices_to_print = [1, 1001, 2043]

    for i in indices_to_print:
        try:
            data, label = train_loader[i]
            print(f"Sample {i}: Data shape {data.shape} ), Label {label}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            break
    
    Output examples:
    1. with details:
    Loading H5 file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_01P_0_0_2250000_4125000.h5
            Loading NPY file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_01P_0_0_2250000_4125000_bbox.npy
            Loaded event tensor with shape: (180, 2, 120, 160)
            Loaded bbox data: 
            [( 460000, 10., 10., 620., 460., 0, 1., 0)
             ( 470000, 10., 10., 620., 460., 0, 1., 0)
            .
            .
             (1180000, 10., 10., 620., 460., 0, 1., 0)
             (1190000, 10., 10., 620., 460., 0, 1., 0)]
            Mapped class name '0' to class ID: paper
            Sample 0: Data shape torch.Size([180, 2, 120, 160]), Label 0
    
    2. without: all gestures correctly mapped
    Loaded label map: {'0': 'paper', '1': 'rock', '2': 'scissor'}
    Found 2046 pairs in /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train.
    Loading H5 file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_01P_0_0_4125000_6200000.h5
    Loading NPY file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_01P_0_0_4125000_6200000_bbox.npy
    Loaded event tensor with shape: (190, 2, 120, 160)
    Mapped class name '0' to class ID: paper
    Sample 1: Data shape torch.Size([190, 2, 120, 160]) ), Label 0
    Loading H5 file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/rock_left_close_slow_sitting_recording_022_2021-09-14_15-20-11.h5
    Loading NPY file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/rock_left_close_slow_sitting_recording_022_2021-09-14_15-20-11_bbox.npy
    Loaded event tensor with shape: (263, 2, 120, 160)
    Mapped class name '1' to class ID: rock
    Sample 1001: Data shape torch.Size([263, 2, 120, 160]) ), Label 1
    Loading H5 file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/scissors_right_far_slow_standing_recording_025_2021-09-14_15-43-08.h5
    Loading NPY file: /space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/scissors_right_far_slow_standing_recording_025_2021-09-14_15-43-08_bbox.npy
    Loaded event tensor with shape: (121, 2, 120, 160)
    Mapped class name '2' to class ID: scissor
    Sample 2043: Data shape torch.Size([121, 2, 120, 160]) ), Label 2
            
                '''