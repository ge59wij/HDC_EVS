import os
import glob
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
'''
Chifoumi_h5_loading_pad.py:

Input: h5 + npy files for labels
HDataset: Reads each [T, 2, 120, 160] into a float tensor, reads the label as an integer from 5th npy array. 
Returns (tensor, label).

DataLoader + collate_fn: Batche (tensor, label) pairs together by padding the time dimension to match the largest T in the batch, then returns (padded_batch, labels_batch).
Detailed:
    goes to each h5 file (normalized, 1000ms time bins):
    read a 4d array of shape [T,2,120,160] and find the label from the 5th el of npy. 120x160 the pixel grid, 1/4 camera grid.
        example: sample [t, c, :, :], 120×160 2D array (matrix) for a specific time bin t and a specific polarity channel c (0 or 1)
        Each number is how many events occurred at that pixel location in that time bin. (time bin either 10,000 or 1000 microseconds available)    
    Each call to dataset[i] returns a tuple (tensor_data, class_id) (before batching)
    
    tensor_data: Pytorch float tensor [T,2,120,160] and class_id is int.

Collate pads 0s depending on batch size, it takes biggest T in that batch. needeed as we stack em later.
    When PyTorch stacks them, two main outputs from the collate_fn: (batch_data_tensor, batch_label_tensor).
    Stacking the padded tensors results in shape: [batch_size, max_T, 2, 120, 160]
    Labels stacked into a batch_labels tensor of shape batch_size: 8,16..
            example: 311 is max_T
            batch_data.shape  =>  [8, 311, 2, 120, 160]
            batch_labels      =>  tensor([0, 1, 2, 2, 0, 0, 0, 2])  here each el corresponds to a class id
###########################
Next: feed (padded_batch, labels) to hdc encoder.

--delta-t 10000               Each time bin (T) represents 10,000 microseconds (10ms) of event accumulation.
--preprocess histo_quantized 
--neg_bit_len_quantized 4     The data uses 8-bit quantization, with 4 bits for negative events (off-polarity).
--total_bit_len_quantized 8   
--normalization_quantized     Normalization was applied to scale event counts.
                             Most likely dividing by a max event count (e.g., 2^8 = 256 or 2^4 = 16 for neg-bit length).
--num-workers 32  
--height_width 120 160


###########################
'''


class EventDatasetLoader(Dataset):
    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
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

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """
        Returns:
          - tensor_data: shape [T, 2, 120, 160]
          - class_id: int
        """
        h5_path, npy_path = self.file_pairs[idx]
        # 1) Load data from .h5
        with h5py.File(h5_path, 'r') as f:
            # shape: [T, 2, 120, 160]
            tensor_data = f['data'][:]
        # Convert numpy array -> torch tensor (float32)
        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        # 2) Load label from .npy (bbox_data)
        # bbox_data[0][5] holds the class_id
        bbox_data = np.load(npy_path, allow_pickle=True)
        class_id = int(bbox_data[0][5])
        return tensor_data, class_id


def collate_fn(batch):
    """
    batch: List of (sequence_tensor, class_id).
      - sequence_tensor: shape [T, 2, 120, 160]
      - class_id: int
    Returns:
      - padded batch tensor of shape: [B, max_T, 2, 120, 160]
      - labels tensor of shape [B]
    """
    sequences, labels = zip(*batch)  # separate into two tuples
    # 1) Find max sequence length T in this batch
    lengths = [seq.shape[0] for seq in sequences]
    max_length = max(lengths)
    # 2) Pad each sequence to max_length along the time dimension with -2
    padded_sequences = []
    for seq in sequences:
        t = seq.shape[0]
        pad_amount = max_length - t
        # seq shape is [T, 2, 120, 160]
        # In F.pad, specify padding in reverse order:
        # (W_left, W_right, H_left, H_right, C_left, C_right, T_left, T_right)
        # We only want to pad time dimension (T), so:
        padded_seq = F.pad(seq, (0, 0, 0, 0, 0, 0, 0, pad_amount), mode='constant', value=-2)
        padded_sequences.append(padded_seq)
    # 3) Stack padded sequences => shape [B, max_T, 2, 120, 160]
    batch_tensor = torch.stack(padded_sequences, dim=0)
    # 4) Convert labels to a torch tensor => shape [B]
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return batch_tensor, label_tensor


if __name__ == "__main__":
    root_dir = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized"
    split = "train"
    dataset = EventDatasetLoader(root_dir=root_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_fn, num_workers=4)

    '''
    sample_data, sample_label, sample_label_str = dataset[0]
    print("Shape of sample_data:", sample_data.shape)  # e.g., [T, 2, 120, 160]
    print("Numeric label (class_id):", sample_label)  # e.g., 0
    print("String label (class_label):", sample_label_str)  # e.g., "rock"
    '''
    sample_data, sample_label = dataset[1500]


    with h5py.File("/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/rock_200211_105303_0_0.h5", "r") as f:

        print("HDF5 Attributes:")
        for key, value in f.attrs.items():
            print(f"{key}: {value}")

        print("\nDataset Keys:", list(f.keys()))  # Lists all datasets in the file
        if 'data' in f:
            print("Data y shape:", f['data'].shape)
    sample1, _ = dataset[0]  # First sample
    sample2, _ = dataset[10]  # Another sample
    sample3, _ = dataset[100]  # Yet another sample
    print("Before unnormalization (normalized):", sample_data[:, 0, 60, 80])  # Polarity 0 at (60,80)
    unnormalized_sample = sample_data * sample_data.max().item()
    print("After unnormalization:", unnormalized_sample[:, 0, 60, 80])

    print("Max event count in sample 1:", sample1.max().item())
    print("Max event count in sample 2:", sample2.max().item())
    print("Max event count in sample 3:", sample3.max().item())

    print("Shape of x sample_data:", sample_data.shape)  # e.g., [T, 2, 120, 160]
    print("Label (class_id):", sample_label)
    print(sample_data[7])
    pixel_x, pixel_y = 60, 80  # Choose a random pixel
    print(sample_data[:, 0, pixel_x, pixel_y])  # Events at (60,80) for polarity 0 (off)
    print(sample_data[:, 1, pixel_x, pixel_y])  # Events at (60,80) for polarity 1 (on)
    for t in range(sample_data.shape[0]):
        print(f"T={t} | Min: {sample_data[t].min().item()}, Max: {sample_data[t].max().item()}")
    #Mean is the average value of all event counts in that sample
#Max is the highest event count in a single !pixel! for any time bin/polarity.
#Min is the lowest event count 

    mean_over_time = sample_data.mean(dim=0)  # Shape: [2, 120, 160]
    print("Mean over time shape:", mean_over_time.shape)  # Expect [2, 120, 160]
    print("Mean over time (example pixel at 60,80):", mean_over_time[:, 60, 80])
    mean_event_count_per_timestep = sample_data.sum(dim=[2, 3]).mean(dim=0)  # Shape: [2]
    print("Mean event count per time step (on/off):", mean_event_count_per_timestep)
    print("Max event count:", sample_data.max().item())  # Should be ~1.0 if fully normalized
    print("Min event count:", sample_data.min().item())  # Should be 0.0 if background

    print("Data max:", sample_data.max().item())
    print("Data min:", sample_data.min().item())


    for i, (batch_data, batch_labels) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("  batch_data shape:", batch_data.shape)  # [#batches, max_T_in_this_batch, 2, 120, 160]
        print("  batch_labels:", batch_labels)
        break  # just show the first batch for demonstration