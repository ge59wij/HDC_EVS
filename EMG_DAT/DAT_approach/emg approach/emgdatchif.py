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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DIMENSIONS = 2000
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_LEVELS = 10  # Reduced levels for time encoding
N_GRAM_SIZE = 4  # Temporal encoding granularity
MAX_TRAIN_SAMPLES = 2000  # Limit training samples to avoid SIGKILL
MAX_TEST_SAMPLES = 400  # Limit test samples to prevent memory issues
CHUNK_SIZE= 1000

class PickleEventDataset(data.Dataset):
    def __init__(self, root_dir, split="train", num_levels=NUM_LEVELS):
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

        self.temporal = embeddings.Level(self.num_levels, DIMENSIONS).to("cpu")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Loads, normalizes, and encodes"""
        with open(self.files[idx], "rb") as f:
            events, class_id = pickle.load(f)  #

        # Convert time values to tensor
        events_tensor = torch.tensor(events[:, 0], dtype=torch.float16)  # Extract only time column

        num_events = events_tensor.shape[0]
        hv_accum = None
        for start in range(0, num_events, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_events)
            chunk = events_tensor[start:end]

            # Normalize and quantize time values
            t_norm = ((chunk - chunk.min()) / (chunk.max() - chunk.min() + 1e-6)) * (self.num_levels - 1)
            t_hv = self.temporal(t_norm.long())  # Encode with torchhd Level embeddings

            # Apply n-grams to capture temporal dependencies
            chunk_hv = torchhd.ngrams(t_hv, n=N_GRAM_SIZE)

            # Aggregate chunk HVs
            if hv_accum is None:
                hv_accum = chunk_hv
            else:
                hv_accum = hv_accum + chunk_hv

        # Normalize the final hypervector
        sample_hv = torchhd.normalize(hv_accum)

        return sample_hv.to(device), torch.tensor(class_id, dtype=torch.long, device=device)

def train_and_test_emg():
    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    train_dataset = PickleEventDataset(dataset_path, split="train")
    test_dataset = PickleEventDataset(dataset_path, split="val")

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Centroid(DIMENSIONS, NUM_CLASSES).to(device)
    print("Training EMG-inspired model...")

    torch.cuda.empty_cache()

    with torch.no_grad():
        for sample_hv, target in tqdm(train_loader, desc="Training"):
            sample_hv = sample_hv.to(torch.float32)  # Reduce memory usage
            model.add(sample_hv.to(device), target.to(device))
            del sample_hv, target
            torch.cuda.empty_cache()

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=NUM_CLASSES)
    print("Testing model...")

    with torch.no_grad():
        model.normalize()
        for sample_hv, target in tqdm(test_loader, desc="Testing"):
            output = model(sample_hv.to(device), dot=True)
            accuracy.update(output.cpu(), target.cpu())

            # Free memory manually
            del sample_hv, target, output
            torch.cuda.empty_cache()

    print(f"Testing Accuracy (EMG-style, no spatial encoding): {accuracy.compute().item() * 100:.2f}%")
    torch.cuda.empty_cache()  # Free memory after training

train_and_test_emg()