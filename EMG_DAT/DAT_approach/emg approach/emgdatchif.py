import os
import pickle
import glob
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
import torchmetrics
import numpy as np
torch.set_printoptions(sci_mode=False)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DIMENSIONS = 6000
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_LEVELS = 90  # Time encoding levels
N_GRAM_SIZE = 4  # Temporal encoding granularity
CHUNK_SIZE = 10000
MAX_TRAIN_SAMPLES = 600
MAX_TEST_SAMPLES = 200

event_dtype = np.dtype([
    ("t", np.float32),  # Timestamp (float32, 4 bytes)
    ("x", np.uint16),   # X-coordinate (uint16, 2 bytes)
    ("y", np.uint16),   # Y-coordinate (uint16, 2 bytes)
    ("p", np.uint8)     # Polarity (uint8, 1 byte)
])

class PickleEventDataset(data.Dataset):
    def __init__(self, root_dir, split="test", num_levels=NUM_LEVELS):
        """Loads preprocessed event data from pickle files."""
        self.root_dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.root_dir, "*.pkl")))
        self.num_levels = num_levels
        self.device = torch.device("cpu")

        # Limit dataset size
        if split == "train":
            self.files = self.files[:MAX_TRAIN_SAMPLES]
        else:
            self.files = self.files[:MAX_TEST_SAMPLES]

        # Initialize hypervector embeddings
        self.temporal = embeddings.Level(self.num_levels, DIMENSIONS).to("cpu")
        self.spatial = embeddings.Random(640 * 480, DIMENSIONS).to("cpu")  # Assuming 640x480 event resolution
        self.polarity = embeddings.Random(2, DIMENSIONS).to("cpu")  # Two polarity values (on/off)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Loads, normalizes, and encodes events into hypervectors."""
        with open(self.files[idx], "rb") as f:
            events, class_id = pickle.load(f)

        events_tensor = torch.tensor(
            np.column_stack((
                events["t"].astype(np.float32),  # timestamps are float32
                events["x"].astype(np.uint16),  #  x-coordinates are uint16
                events["y"].astype(np.uint16),  #  y-coordinates are uint16
                events["p"].astype(np.uint8)  #  polarity is uint8
            )),
            dtype=torch.float32
        )
        print("First 10 timestamps:", events["t"][:10])
        print("Data type of timestamps:", events["t"].dtype)

        print(f"Sample {idx} - Loaded {events_tensor.shape[0]} events")
        if events_tensor.shape[0] == 0:
            print(f"Skipping empty sample {idx}")
            return None

        num_events = events_tensor.shape[0]
        hv_accum = None

        for start in range(0, num_events, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_events)
            chunk = events_tensor[start:end]

            print(f"Chunk {start}:{end} - Shape: {chunk.shape}")

            if chunk.shape[0] == 0:
                print(f"Empty chunk at {start}:{end} for sample {idx}, skipping chunk")
                continue

            if chunk[:, 0].max() == chunk[:, 0].min():
                print(f"Skipping chunk {start}:{end} (identical timestamps)")
                continue
            print("Chunk structure:", chunk.shape, chunk.dtype)

            # **Temporal Encoding**
            t_norm = ((chunk[:, 0] - chunk[:, 0].min()) / (chunk[:, 0].max() - chunk[:, 0].min() + 1e-6)) * (self.num_levels - 1)
            hv_t = self.temporal(t_norm.long())

            # **Spatial Encoding**
            x = chunk[:, 1].long()
            y = chunk[:, 2].long()
            spatial_index = (x * 480 + y)
            hv_s = self.spatial(spatial_index)

            # **Polarity Encoding**
            p = chunk[:, 3].long()
            hv_p = self.polarity(p)
            #print(f"Chunk dtype: {chunk.dtype}")
            #print(f"Chunk structure: {chunk.shape}")
            #print(f"First 5 rows: {chunk[:5]}")
            # **Bind All Components**
            hv_chunk = torchhd.bind(torchhd.bind(hv_t, hv_s), hv_p)

            # **Apply N-Grams**
            hv_chunk = torchhd.ngrams(hv_chunk, n=N_GRAM_SIZE)

            if hv_accum is None:
                hv_accum = hv_chunk
            else:
                hv_accum = hv_accum + hv_chunk

        if hv_accum is None:
            print(f"No valid chunks in sample {idx}, skipping..")
            return None

        # Normalize final hypervector
        sample_hv = torchhd.normalize(hv_accum)

        return sample_hv.to(device), torch.tensor(class_id, dtype=torch.long, device=device)

def train_and_test_emg():
    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    train_dataset = PickleEventDataset(dataset_path, split="test")
    test_dataset = PickleEventDataset(dataset_path, split="test")

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Centroid(DIMENSIONS, NUM_CLASSES).to(device)
    print("Training EMG-inspired model with spatial and polarity encoding...")

    torch.cuda.empty_cache()

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Training"):
            if batch is None:
                continue
            sample_hv, target = batch
            sample_hv = sample_hv.to(torch.float32)
            model.add(sample_hv.to(device), target.to(device))
            del sample_hv, target
            torch.cuda.empty_cache()

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=NUM_CLASSES)
    print("Testing model...")

    with torch.no_grad():
        model.normalize()
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None:
                continue
            sample_hv, target = batch
            output = model(sample_hv.to(device), dot=True)
            accuracy.update(output.cpu(), target.cpu())
            del sample_hv, target, output
            torch.cuda.empty_cache()

    print(f"Testing Accuracy (Spatial + Polarity Encoding): {accuracy.compute().item() * 100:.2f}%")
    torch.cuda.empty_cache()

train_and_test_emg()
