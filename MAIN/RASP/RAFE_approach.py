import os
import glob
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import torchhd
from torchhd import embeddings, bind, multiset, normalize
from torchhd.models import Centroid
from tqdm import tqdm


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
        h5_path, npy_path = self.file_pairs[idx]
        with h5py.File(h5_path, 'r') as f:
            tensor_data = f['data'][:]  # Shape: [T, 2, 120, 160]
        tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        bbox_data = np.load(npy_path, allow_pickle=True)
        class_id = int(bbox_data[0][5])
        return tensor_data, class_id


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, label_tensor


class GraspHDEncoder(nn.Module):
    """maps an input event tensor of shape [T, 2, H, W] into a high-dimensional hypervector by incorporating spatial and temporal correlations.
   For each time bin and polarity (0: off, 1: on):
     1. The spatial frame (H x W) is partitioned into non-overlapping blocks of size kxk
     2. For each block, we sum the events (yielding an event count).
     3. The scalar count is mapped to a hypervector using a random projection followed by a sinusoidal activation (mimicking RAFE).
     4. This hypervector is bound with a pre-generated block (position) hypervector, a polarity seed, and a time seed.
   All the bound hypervectors across time, polarity, and space are then bundled together via multiset and quantized.
   """

    def __init__(self, out_features, max_time_bins=100, block_size=8, H=120, W=160):
        super(GraspHDEncoder, self).__init__()
        self.out_features = out_features
        self.block_size = block_size
        self.H = H
        self.W = W
        # Determine number of blocks in height and width
        self.H_blocks = H // block_size
        self.W_blocks = W // block_size
        self.num_blocks = self.H_blocks * self.W_blocks

        # Projection layer: maps scalar count to HD vector.
        self.projection = embeddings.Projection(1, out_features)
        # (We apply a sinusoidal non-linearity after projection.)

        # Position (block) seeds: one hypervector per block.
        self.position = embeddings.Random(self.num_blocks, out_features)
        # Reshape so that each block has its seed; will index using row-major order.

        # Time seeds for up to max_time_bins time bins.
        self.time = embeddings.Random(max_time_bins, out_features)

        # Polarity seeds
        self.polarity = embeddings.Random(2, out_features)

    def forward(self, x):
        # x: [T, 2, H, W]
        T, channels, H, W = x.shape
        hv_list = []
        # Process each time bin.
        for t in range(T):
            # Skip time bins with no events.
            if x[t].sum() == 0:
                continue
            # Get time seed (if t exceeds max_time_bins, use the last seed).
            t_idx = t if t < self.time.weight.shape[0] else self.time.weight.shape[0] - 1
            time_hv = self.time.weight[t_idx]
            # Process each polarity.
            for p in range(2):
                # x[t, p] is of shape [H, W]
                event_map = x[t, p]  # [H, W]
                # Use unfold to aggregate non-overlapping blocks.
                # Reshape to [1, 1, H, W] so that unfold works.
                event_map_unsq = event_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                blocks = F.unfold(event_map_unsq, kernel_size=self.block_size, stride=self.block_size)
                # blocks: [1, block_size*block_size, num_blocks]
                block_sums = blocks.sum(dim=1).squeeze(0)  # [num_blocks]
                # Reshape block_sums to [H_blocks, W_blocks]
                block_sums = block_sums.view(self.H_blocks, self.W_blocks)
                # Process each block
                for i in range(self.H_blocks):
                    for j in range(self.W_blocks):
                        count = block_sums[i, j].unsqueeze(0).unsqueeze(0)  # shape [1, 1]
                        # Project the scalar count to a hypervector.
                        proj = self.projection(count)  # [1, out_features]
                        # Apply a sinusoidal non-linearity.
                        activation = torch.sin(proj)  # [1, out_features]
                        # Get the block's position seed.
                        # Convert 2D index (i,j) to linear index.
                        lin_idx = i * self.W_blocks + j
                        pos_hv = self.position.weight[lin_idx]
                        # Bind the projected activation with the block seed.
                        hv = torchhd.bind(pos_hv, activation.squeeze(0))
                        # Bind with the polarity seed.
                        pol_hv = self.polarity.weight[p]
                        hv = torchhd.bind(pol_hv, hv)
                        # Finally, bind with the time seed.
                        hv = torchhd.bind(time_hv, hv)
                        hv_list.append(hv)
        if len(hv_list) == 0:
            sample_hv = torch.zeros(self.out_features, device=x.device)
        else:
            sample_hv = multiset(torch.stack(hv_list, dim=0))
            sample_hv = torchhd.normalize(sample_hv)
        return sample_hv


class RAFEEncoder(nn.Module):
    """
    Encodes a [T, 2, 120, 160] event tensor into a hypervector using a RAFE-inspired method.
    For each time bin and for each polarity (0 for off, 1 for on):
      - Sums the events over spatial dimensions to get an event count.
      - Uses a random projection (torchhd.embeddings.Projection) to map the scalar count into a hypervector.
      - Applies a sinusoidal non-linearity (simulating the RAFE non-linear activation).
      - Binds the resulting hypervector with a time seed and a polarity seed.
    The bound hypervectors are then bundled (via multiset) and hard-quantized to produce the final HD vector.
    """

    def __init__(self, out_features, max_time_bins):
        super(RAFEEncoder, self).__init__()
        self.out_features = out_features
        # Projection for mapping the scalar event count (shape: [1]) to HD space.
        self.projection = embeddings.Projection(1, out_features)
        # We'll use a sinusoidal activation to mimic the non-linearity in RAFE.
        # (Alternatively, one might use torch.sin directly.)
        self.sinusoid = nn.Identity()  # We'll apply torch.sin directly in forward.
        # Random seeds for time bins
        self.time = embeddings.Random(max_time_bins, out_features)
        # Two fixed random hypervectors for polarity: 0 (off) and 1 (on)
        self.polarity = embeddings.Random(2, out_features)

    def forward(self, x):
        # x: Tensor of shape [T, 2, 120, 160]
        T = x.shape[0]
        hv_list = []
        for t in range(T):
            # Skip padded/empty time bins
            if x[t].sum() == 0:
                continue
            for p in range(2):
                # Sum events spatially for the given time bin and polarity.
                event_count = x[t, p, :, :].sum().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]
                # Map the scalar event count to a hypervector using random projection.
                proj = self.projection(event_count)  # Shape: [1, out_features]
                # Apply sinusoidal activation (RAFE non-linearity)
                encoded = torch.sin(proj)  # Shape: [1, out_features]
                # Retrieve the corresponding time seed (if t exceeds max_time_bins, use last available)
                time_index = t if t < self.time.weight.shape[0] else self.time.weight.shape[0] - 1
                time_hv = self.time.weight[time_index]
                # Retrieve the polarity seed (p=0: off, p=1: on)
                polarity_hv = self.polarity.weight[p]
                # Bind the hypervectors: first bind polarity and encoded event, then bind with time.
                bound_hv = bind(time_hv, bind(polarity_hv, encoded.squeeze(0)))
                hv_list.append(bound_hv)
        if len(hv_list) == 0:
            sample_hv = torch.zeros(self.out_features, device=x.device)
        else:
            sample_hv = torchhd.multiset(torch.stack(hv_list, dim=0))
            sample_hv = torchhd.normalize(sample_hv)
        return sample_hv


#################################

def main():
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized"
    train_split = "train"
    test_split = "val"

    samples_train = 100
    samples_test = 40

    train_dataset = EventDatasetLoader(root_dir, train_split)
    test_dataset = EventDatasetLoader(root_dir, test_split)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    DIMENSIONS = 8000
    MAX_TIME_BINS = 900  # Maximum expected number of time bins
    BLOCK_SIZE = 4  # Block size for spatial segmentation.
    H, W = 120, 160

    encoder = GraspHDEncoder(DIMENSIONS, max_time_bins=MAX_TIME_BINS, block_size=BLOCK_SIZE, H=H, W=W).to(device)

    # Initialize the RAFE encoder (using a RAFE-inspired method from HD_DVS)
    # encoder = RAFEEncoder(DIMENSIONS, max_time_bins=MAX_TIME_BINS).to(device)

    # Centroid
    num_classes = train_dataset.num_classes
    model = Centroid(DIMENSIONS, num_classes).to(device)

    print("Training class prototypes...")
    with torch.no_grad():
        count_train = 0
        for batch_data, batch_labels in tqdm(train_loader, desc="Training"):
            batch_data = batch_data.to(device)  # Shape: [B, T, 2, 120, 160]
            batch_labels = batch_labels.to(device)
            B = batch_data.shape[0]

            # Process each sample in the batch
            for i in range(B):
                # If we've reached our training sample limit, stop.
                if samples_train is not None and count_train >= samples_train:
                    break

                sample = batch_data[i]  # Shape: [T, 2, 120, 160]
                label = batch_labels[i]
                sample_hv = encoder(sample)
                model.add(sample_hv.unsqueeze(0), label.unsqueeze(0))
                count_train += 1

            # If we've reached our training sample limit, break the outer loop as well.
            if samples_train is not None and count_train >= samples_train:
                break

    print("Evaluating on test set...")
    correct = 0
    total = 0
    with torch.no_grad():
        model.normalize()
        count_test = 0
        for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            B = batch_data.shape[0]

            for i in range(B):
                # If we've reached our testing sample limit, stop.
                if samples_test is not None and count_test >= samples_test:
                    break

                sample = batch_data[i]
                true_label = batch_labels[i].item()
                sample_hv = encoder(sample)
                outputs = model(sample_hv.unsqueeze(0), dot=True)
                predicted_label = outputs.argmax(dim=1).item()
                if predicted_label == true_label:
                    correct += 1
                total += 1
                count_test += 1

            if samples_test is not None and count_test >= samples_test:
                break

    accuracy = correct / total if total > 0 else 0
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
