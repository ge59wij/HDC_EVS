##doesnt train, dat loader encode
#

import os
import struct
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import mmap
from tqdm import tqdm
import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import torchhd
from torchhd import embeddings, bind, normalize, multiset
from torchhd.models import Centroid

class EventDatasetLoader(Dataset):
    def __init__(self, split, device="cuda", num_workers=4):
        self.root_dir = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT/"
        self.split = split
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.split_dir = os.path.join(self.root_dir, split)
        self.event_files = sorted([
            os.path.join(self.split_dir, file)
            for file in os.listdir(self.split_dir)
            if file.endswith(".dat")
        ])
        print(f"Using {self.device} device")
        print(f"Found {len(self.event_files)} files in {self.split_dir}")
        if not self.event_files:
            raise ValueError(f"No .dat files found in the {split} directory: {self.split_dir}")
        self.label_map = self._load_label_map()
    def _load_label_map(self):
        json_path = os.path.join(self.root_dir, "label_map_dictionary.json")
        with open(json_path, "r") as f:
            return {v.lower(): int(k) for k, v in json.load(f).items()}

    def _parse_dat_file(self, file_path, labels):
        print(f"Loading {file_path}...")
        events = []
        with open(file_path, "rb") as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mmapped_file.seek(0)

            while mmapped_file.readline().startswith(b"%"):
                continue

            data = mmapped_file.read()

            for i in range(0, len(data), 8):
                if i + 8 > len(data):
                    break
                word = struct.unpack("<Q", data[i:i + 8])[0]
                timestamp = (word >> 32) & 0xFFFFFFFF
                polarity = (word >> 28) & 0xF
                y = (word >> 14) & 0x3FFF
                x = word & 0x3FFF

                if any(
                    x_min <= x <= x_max and y_min <= y <= y_max
                    for (_, x_min, y_min, x_max, y_max, *_) in labels
                ):
                    events.append((timestamp, polarity, x, y))

        print(f"Loaded {len(events)} filtered events from {file_path}")
        return events

    def _load_labels(self, file_path):
        npy_path = file_path.replace("_td.dat", "_bbox.npy").replace("_cd.dat", "_bbox.npy")
        if os.path.exists(npy_path):
            print(f"Attempting to load labels from: {npy_path}")
            try:
                labels = np.load(npy_path, allow_pickle=True)
                print(f"Loaded labels from {npy_path}, total: {len(labels)}")
                return labels
            except Exception as e:
                print(f"Warning: Failed to load {npy_path} - {e}")
                return None
        else:
            print(f"Label file not found: {npy_path}")
            return None

    def process_file_in_memory(self, file_path):
        try:
            print(f"Processing file: {file_path}")  # Debugging statement
            labels = self._load_labels(file_path)
            if labels is not None:
                events = self._parse_dat_file(file_path, labels)
            else:
                events = []
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            events, labels = [], []
        return events, labels

    def preprocess_all_in_memory(self):
        print("Starting dataset processing in memory...")
        results = []

        # Process files in memory with reduced workers (testing with 4 workers)
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for file_path in self.event_files:
                futures.append(executor.submit(self.process_file_in_memory, file_path))

            for f in tqdm(futures, total=len(self.event_files)):
                events, labels = f.result()
                results.append((events, labels))

        print("Dataset processing complete!")
        return results

class Encoding2(nn.Module):
    def __init__(self, dimensions, max_time, height, width, num_classes):
        super(Encoding2, self).__init__()
        self.dimensions = dimensions
        self.max_time = max_time
        self.height = height
        self.width = width
        self.num_classes = num_classes

        self.spatial = torchhd.embeddings.Random(width * height, dimensions, vsa="MAP")
        self.temporal = torchhd.embeddings.Random(max_time, dimensions, vsa="MAP")
        self.centroid_model = Centroid(dimensions, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def encode(self, data):
        batch_size, time_bins, channels, height, width = data.shape
        data = data.flatten(3).flatten(2)
        num_levels = 256
        data = (data / data.max(dim=-1, keepdim=True).values) * (num_levels - 1)
        indices = data.round().long()
        indices = indices.clamp(min=0, max=num_levels - 1)

        hv = []
        for t in range(min(time_bins, self.max_time)):
            t_tensor = torch.tensor([t], dtype=torch.long, device=indices.device)
            temporal_hv = self.temporal(t_tensor)
            spatial_hv = self.spatial(indices[:, t])
            spatial_hv = multiset(spatial_hv)
            combined_hv = bind(spatial_hv, temporal_hv)
            hv.append(combined_hv)
        hv = multiset(torch.stack(hv, dim=1))
        return normalize(hv)
def generate_heatmap(all_data, encoder):
    print("Generating heatmap of class clusters...")
    encoded_samples = []

    for events, labels in all_data:
        events_tensor = torch.tensor(events).to(encoder.device)
        encoded_sample = encoder.encode(events_tensor)
        encoded_samples.append(encoded_sample)
    encoded_samples = torch.cat(encoded_samples, dim=0)
    similarity_matrix = torch.mm(encoded_samples, encoded_samples.t())
    similarity_matrix = similarity_matrix.cpu().numpy()
    sns.heatmap(similarity_matrix, cmap="YlGnBu", xticklabels=range(len(all_data)), yticklabels=range(len(all_data)))
    plt.title("Class Hypervector Similarity Heatmap")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.show()

if __name__ == "__main__":
    loader = EventDatasetLoader(split="train", device="cuda", num_workers=4)  # Reduced num_workers for testing
    all_data = loader.preprocess_all_in_memory()
    encoder = Encoding2(dimensions=10000, max_time=100, height=120, width=160, num_classes=3).to(loader.device)
    generate_heatmap(all_data, encoder)
