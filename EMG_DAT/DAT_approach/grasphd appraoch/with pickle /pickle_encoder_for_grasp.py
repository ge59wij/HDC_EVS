import torch
import torchhd
import numpy as np
import os
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from grasphdencoding import GraspHDEventEncoder
from torch.utils.data import DataLoader, TensorDataset
import torchhd.functional as functional
import random

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)

def main():
    device = "cuda"  # if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    train_split, test_split = "train", "test"
    max_samples = 14
    DIMS = 300
    K = 3
    Timewindow = 500000  # Time bin size
    num_classes = 3
    batches = 1

    # Load datasets (NOW OUTPUTS TENSORS)
    train_loader, train_max_time = load_pickle_dataset(dataset_path, split=train_split, max_samples=max_samples, batch_size= batches, device=device)
    test_loader, test_max_time  = load_pickle_dataset(dataset_path, split=test_split, max_samples=max_samples,batch_size= batches, device=device)

    # Ensure max_time covers both train and test
    max_time = max(train_max_time, test_max_time)
    print(f"[INFO] Adjusted Global Maximum Timestamp: {max_time} µs")

    encoder = GraspHDEventEncoder(480, 640, DIMS, Timewindow, K, device, max_time)
    classifier = torchhd.models.Centroid(DIMS, num_classes, device=device)

    with torch.no_grad():
        for batch_events, batch_labels in tqdm(train_loader, desc="Encoding Train Batches"):
            encoded_batch = encoder.encode_grasphd(batch_events, batch_labels)
            classifier.add(encoded_batch, batch_labels)

    classifier.normalize()
    evaluate_model(classifier, encoder, test_loader, device)


def load_pickle_dataset(dataset_path, split, max_samples, batch_size, device):
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pkl')]
    random.shuffle(files)

    event_list, label_list = [], []
    max_time = 0

    for file in files[:max_samples]:
        with open(file, 'rb') as f:
            events, class_id = pickle.load(f)

        if isinstance(events, np.ndarray) and events.dtype.names is not None:
            events = np.column_stack((events["t"], events["x"], events["y"], events["p"]))

        events_tensor = torch.tensor(events, dtype=torch.float32, device=device)
        label_tensor = torch.tensor(class_id, dtype=torch.long, device=device)

        sample_max_time = torch.max(events_tensor[:, 0]) if events_tensor.numel() > 0 else 0
        max_time = max(max_time, sample_max_time.item())

        event_list.append(events_tensor)
        label_list.append(label_tensor)

    dataset = TensorDataset(torch.stack(event_list), torch.stack(label_list))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(event_list)} samples from {split} split. Max timestamp: {max_time} µs")
    return data_loader, max_time


def evaluate_model(classifier, encoder, test_loader, device):
    print("\n Running accuracy test on test batches...\n")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_events, batch_labels in test_loader:
            encoded_test_samples = encoder.encode_grasphd(batch_events, batch_labels)
            similarities = classifier(encoded_test_samples)
            predicted_labels = torch.argmax(similarities, dim=1)
            correct += (predicted_labels == batch_labels).int().sum().item()
            total += batch_labels.numel()

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[RESULT] Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")


if __name__ == "__main__":
    main()
