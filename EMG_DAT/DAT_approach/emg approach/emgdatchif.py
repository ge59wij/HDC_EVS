import os
import pickle
import glob
import torch
import torch.utils.data as data
from tqdm import tqdm
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
import torchmetrics
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DIMENSIONS = 3000
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_LEVELS = 90
N_GRAM_SIZE = 4
CHUNK_SIZE = 10000
MAX_TRAIN_SAMPLES = 200
MAX_TEST_SAMPLES = 20

class PickleEventDataset(data.Dataset):
    def __init__(self, root_dir, split="test", num_levels=NUM_LEVELS):
        self.root_dir = os.path.join(root_dir, split)
        self.files = sorted(glob.glob(os.path.join(self.root_dir, "*.pkl")))
        self.num_levels = num_levels

        if split == "train":
            self.files = self.files[:MAX_TRAIN_SAMPLES]
        else:
            self.files = self.files[:MAX_TEST_SAMPLES]

        self.temporal = embeddings.Level(self.num_levels, DIMENSIONS).to(device)
        self.spatial = embeddings.Random(640 * 480, DIMENSIONS).to(device)
        self.polarity = embeddings.Random(2, DIMENSIONS).to(device)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            events, class_id = pickle.load(f)

        if len(events["t"]) == 0:
            print(f"Skipping empty sample {idx}")
            return None

        events_tensor = torch.tensor(
            np.column_stack((
                events["t"].astype(np.float32),
                events["x"].astype(np.uint16),
                events["y"].astype(np.uint16),
                events["p"].astype(np.uint8)
            )),
            dtype=torch.float32,
            device=device
        )

        num_events = events_tensor.shape[0]
        hv_accum = None

        for start in range(0, num_events, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_events)
            chunk = events_tensor[start:end]

            if chunk.shape[0] == 0 or chunk[:, 0].max() == chunk[:, 0].min():
                continue

            # **Temporal Encoding**
            t_norm = ((chunk[:, 0] - chunk[:, 0].min()) / (chunk[:, 0].max() - chunk[:, 0].min() + 1e-6)) * (self.num_levels - 1)
            t_norm = t_norm.clamp(0, self.num_levels - 1)  # Ensure valid index
            hv_t = self.temporal(t_norm.long())

            # **Spatial Encoding (Fixing Indexing Issue)**
            x = chunk[:, 1].long().clamp(0, 639)
            y = chunk[:, 2].long().clamp(0, 479)
            spatial_index = (x * 480 + y).clamp(0, 307199)  # Ensure valid range
            hv_s = self.spatial(spatial_index)

            # **Polarity Encoding**
            p = chunk[:, 3].long().clamp(0, 1)  # Ensure 0 or 1
            hv_p = self.polarity(p)

            # **Bind & N-Grams**
            hv_chunk = torchhd.bind(torchhd.bind(hv_t, hv_s), hv_p)

            torch.cuda.empty_cache()
            hv_chunk = hv_chunk.cpu()
            hv_chunk = torchhd.ngrams(hv_chunk, n=N_GRAM_SIZE)
            hv_chunk = hv_chunk.to(device)

            if hv_accum is None:
                hv_accum = hv_chunk
            else:
                hv_accum = hv_accum + hv_chunk

            del hv_chunk
            torch.cuda.empty_cache()

        if hv_accum is None:
            return None

        return torchhd.normalize(hv_accum), torch.tensor(class_id, dtype=torch.long, device=device)

def train_and_test_emg():
    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    train_dataset = PickleEventDataset(dataset_path, split="val")
    test_dataset = PickleEventDataset(dataset_path, split="test")

    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Centroid(DIMENSIONS, NUM_CLASSES).to(device)
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=NUM_CLASSES)
    print("Training..")

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Training"):
            if batch is None:
                continue
            sample_hv, target = batch
            model.add(sample_hv.to(device), target.to(device))
            del sample_hv, target
            torch.cuda.empty_cache()

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

    print(f"Testing Accuracy: {accuracy.compute().item() * 100:.2f}%")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_and_test_emg()
