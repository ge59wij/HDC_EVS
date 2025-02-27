import os
import h5py
import numpy as np
import torch
import torchhd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchhd.models import Centroid
from BASE_HIST import HDHypervectorGenerators
from HIST_Encoder import HISTEncoder
import random

BACKGROUND_LABEL = 404
dataset_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/Bin Labeled/"
Train_split = "VAL BIN LABELED"
DIMS = 2000
BIN_SIZE = 30
SLIDING_WINDOW = False
MAX_SAMPLES = 20
BATCH_SIZE = 1
WINDOW_SIZE = 30
STRIDE = 27
NUM_RANDOM_VECTORS = 50  # How many random vectors we compare


class HDF5Dataset(Dataset):
    def __init__(self, dataset_folder, max_samples=None):
        self.files = [
            os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.h5')
        ]
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, "r") as f:
            event_data = torch.tensor(f["data"][:], dtype=torch.float32)  # Shape: (T, 2, 120, 160)
            class_ids = torch.tensor(f["labels"][:], dtype=torch.int32)  # Shape: (T,)
        return event_data, class_ids, file_path


def process_sample(event_data, class_ids, encoder, bundle_size=30, stride=3):
    """Encodes a sample, bundling every `bundle_size` bins into one gesture vector.
       Background is collected across the whole sample and bundled into ONE vector.
    """
    BACKGROUND_LABEL = 404
    T = event_data.shape[0]  # Total bins
    gesture_hvs, background_hvs = [], []
    background_list = []  # Collect all background vectors

    # Process bins in sliding window fashion
    i = 0
    while i + bundle_size <= T:
        end_bin = i + bundle_size
        bin_range = slice(i, end_bin)
        current_class_ids = class_ids[bin_range]
        unique_classes = torch.unique(current_class_ids)

        print(f"[DEBUG] Encoding WINDOW {i}-{end_bin}")

        if BACKGROUND_LABEL in unique_classes and len(unique_classes) > 1:
            gesture_mask = current_class_ids != BACKGROUND_LABEL
            background_mask = current_class_ids == BACKGROUND_LABEL
            gesture_bins = event_data[bin_range][gesture_mask]
            background_bins = event_data[bin_range][background_mask]

            if gesture_bins.shape[0] > 0:
                gesture_hv = encoder.encode_window(gesture_bins, i, bundle_size)
                gesture_hvs.append((gesture_hv, int(unique_classes[unique_classes != BACKGROUND_LABEL][0])))

            if background_bins.shape[0] > 0:
                background_hv = encoder.encode_window(background_bins, i, bundle_size)
                background_list.append(background_hv)

        else:
            hv = encoder.encode_window(event_data[bin_range], i, bundle_size)
            gesture_hvs.append((hv, int(unique_classes[0])))

        print(f"[DEBUG] Finished encoding window {i}-{end_bin}")
        i = end_bin - stride  # Move to the next window, starting at (end - stride)

    # Bundle all collected background hypervectors
    if len(background_list) > 0:
        stacked_background = torch.stack(background_list)
        final_background_hv = torchhd.multiset(stacked_background)
        final_background_hv = torchhd.normalize(final_background_hv)
        background_hvs.append((final_background_hv, BACKGROUND_LABEL))
        print(f"[DEBUG] Final Background Vector Created, {len(background_list)} parts combined!")


    return gesture_hvs, background_hvs


def plot_heatmap(vectors_matrix, class_labels):
    """Generates a heatmap for similarity visualization for randomly selected vectors."""
    class_labels_tensor = torch.tensor(class_labels)

    selected_indices = random.sample(range(len(vectors_matrix)), min(NUM_RANDOM_VECTORS, len(vectors_matrix)))
    selected_vectors = torch.stack([vectors_matrix[i] for i in selected_indices])
    selected_labels = [class_labels[i] for i in selected_indices]

    similarity_matrix = torchhd.cosine_similarity(selected_vectors, selected_vectors).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, fmt=".2f", cmap="coolwarm",
                xticklabels=selected_labels, yticklabels=selected_labels,
                cbar=True, square=True, linewidths=0.5)
    plt.title(f"Cosine Similarity Heatmap ({NUM_RANDOM_VECTORS} Random Samples)")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")
    plt.show()


def main():
    device = "cpu"
    print(f"Using device: {device}")

    dataset_train = HDF5Dataset(os.path.join(dataset_path, Train_split), max_samples=MAX_SAMPLES)
    dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    encoder = HISTEncoder(height=120, width=160, dims=DIMS, device=device, threshold=1 / 16, window_size=WINDOW_SIZE,
                          stride=STRIDE)

    encoded_vectors = []
    class_labels = []

    for event_data, class_ids, filename in tqdm(dataloader, desc="Encoding Samples"):
        gesture_hvs, background_hvs = process_sample(event_data.squeeze(0), class_ids.squeeze(0), encoder,
                                                     bundle_size=BIN_SIZE)
        for hv, class_id in gesture_hvs + background_hvs:
            encoded_vectors.append(hv)
            class_labels.append(class_id)

    encoded_vectors = torch.stack(encoded_vectors)
    plot_heatmap(encoded_vectors, class_labels)


if __name__ == "__main__":
    main()
