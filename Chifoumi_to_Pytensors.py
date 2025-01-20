import os
import glob
import h5py
import json
import numpy as np
import torch


class EventDatasetLoader:
    """
    Dataset loader for RNN-style event dataset (Prophesee style), handling HDF5 (.h5 ) event tensors and NPY label files.
    Single root dir should contain train/val/test splits, label_map_dictionary.json
    """

    def __init__(self, root_dir, split):
        """
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = self._load_label_map()
        self.h5_files = glob.glob(os.path.join(self.split_dir, '*.h5'))
        self.file_pairs = self._get_file_pairs()
        print(f"Found {len(self.file_pairs)} pairs in {self.split_dir}.")
        print("Loading Dataset")

    def _load_label_map(self):
        """
        Loads the label mapping from label_map_dictionary.json.: {'0': 'paper', '1': 'rock', '2': 'scissor'}

        Returns:
            dict: Mapping of label names to class IDs (integers)
        """
        json_path = os.path.join(self.root_dir, "label_map_dictionary.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Label map file not found: {json_path}")
        with open(json_path, "r") as f:
            label_map = json.load(f)
       # print(f"Loaded label map: {label_map}")
        return label_map

    def _get_file_pairs(self):
        """
        Matches HDF5 files with corresponding NPY files.
        Returns:
            List of tuples: [(h5_file, npy_file)]
        """
        pairs = []
        for h5_file in self.h5_files:
            npy_file = h5_file.replace('.h5', '_bbox.npy')
            if os.path.exists(npy_file):
                pairs.append((h5_file, npy_file))
        return pairs

    def load_sample(self, h5_path, npy_path):
        """
        Loads a single sample from an HDF5 file and its corresponding NPY file, convert to Pytorch Tensor, Maps class,
        Returns:
            tuple: (tensor_data, label_id)
        """
       # print(f"Loading H5 file: {h5_path}")
        #print(f"Loading NPY file: {npy_path}")
        # Load event tensor
        with h5py.File(h5_path, 'r') as f:
            tensor_data = f['data'][:]  # Shape [T, 2, H, W]

        # Load bbox data
        bbox_data = np.load(npy_path, allow_pickle=True)
        if len(bbox_data) == 0:
            raise ValueError(f"Empty bbox data in {npy_path}")

        # Extract class name and map it to class ID
        class_name = str(int(bbox_data[0][5]))  # Assuming the class name is the 6th element
        class_id = self.label_map.get(class_name, -1)
        if class_id == -1:
            raise ValueError(f"Class {class_name} not found in label map!")
        class_id = int(class_name)

        MAX_TIME = 150

        T, C, H, W = tensor_data.shape
        if T < MAX_TIME:
            # Pad with zeros along the time dimension
            padding = np.zeros((MAX_TIME - T, C, H, W), dtype=tensor_data.dtype)
            tensor_data = np.concatenate((tensor_data, padding), axis=0)
        elif T > MAX_TIME:
            # Truncate along the time dimension
            tensor_data = tensor_data[:MAX_TIME]

        return torch.tensor(tensor_data, dtype=torch.float32), torch.tensor(class_id, dtype=torch.long)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        h5_path, npy_path = self.file_pairs[idx]
        return self.load_sample(h5_path, npy_path)


if __name__ == "__main__":
    dataset_path = "/space/chair-nas/tosy/Simple_chifoumi/" #"/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/"


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