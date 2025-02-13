import torch
import torchhd
import numpy as np
import os
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from grasphdencoding import GraspHDEventEncoder
import torchhd.functional as functional
import random
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)

def main():
    device = "cpu"  # if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    dataset_path = "/space/chair-nas/tosy/pt_chifoumi"
    train_split, test_split = "train", "test"
    max_samples_train = 100
    max_samples_test = 22
    DIMS = 4000
    K = 4
    Timewindow = 1000000  # Time bin size #ideally?
    num_classes = 3
    batches = 1
    Encoding_Class = GraspHDEventEncoder
    encoding_mode = "encode_grasphd"
    print(f"Encoding mode: {encoding_mode}")


    # Load datasets (NOW OUTPUTS TENSORS)
    train_loader, train_max_time = load_tensor_dataset(dataset_path, split=train_split, max_samples=max_samples_train, batch_size= batches, device=device)
    test_loader, test_max_time = load_tensor_dataset(dataset_path, split=test_split, max_samples=max_samples_test,batch_size= batches, device=device)

    max_time = max(train_max_time, test_max_time)
    print(f"DEBUG: Passing Timewindow = {Timewindow}")
    print(f"[INFO] Adjusted Global Maximum Timestamp: {max_time} µs")

    encoder = Encoding_Class(480, 640, DIMS, Timewindow, K, device, max_time)
    classifier = torchhd.models.Centroid(DIMS, num_classes, device=device)
    encoding_methods = {
        "encode_grasphd": encoder.encode_grasphd,
        "nsumming": encoder.encode_grasp_n_summing,
    }
    encode_method = encoding_methods.get(encoding_mode)
    if encode_method is None:
        raise ValueError(f"Invalid encoding mode '{encoding_mode}'. Choose from: {list(encoding_methods.keys())}")

    with torch.no_grad():
        for batch_events, batch_labels in tqdm(train_loader, desc=f"Encoding Train Batches ({encoding_mode})"):
            encoded_batch = encode_method(batch_events, batch_labels)  # Dynamically call selected method
            classifier.add(encoded_batch, batch_labels)
    classifier.normalize()
    evaluate_model(classifier, encoder, test_loader, device)

def load_tensor_dataset(dataset_path, split, max_samples, batch_size, device):
    """
    Loads event tensors and labels directly from .pt files.
    Ensures correct padding if batch_size > 1.
    """
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pt')]

    event_list, label_list = [], []
    max_time = 3600000 #1960000

    for file in files[:max_samples]:
        data = torch.load(file)  # Load from .pt
        events_tensor, label_tensor = data["events"], data["label"]

        # Ensure label is correctly shaped
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)  # Convert scalar to tensor

        # Move to device
        events_tensor = events_tensor.to(device)
        label_tensor = label_tensor.to(device)

        # Update max_time dynamically
        #max_time = max(max_time, events_tensor[:, 0].max().item() if events_tensor.numel() > 0 else 0)

        event_list.append(events_tensor)
        label_list.append(label_tensor)

    # **Only pad if batch_size > 1**
    if batch_size > 1:
        padded_events = pad_sequence(event_list, batch_first=True, padding_value=0)
        dataset = TensorDataset(padded_events, torch.cat(label_list))
    else:
        dataset = list(zip(event_list, label_list))  # List of (event_tensor, label_tensor) tuples

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Loaded {len(event_list)} samples from {split}. Max timestamp: {max_time} µs.")
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
