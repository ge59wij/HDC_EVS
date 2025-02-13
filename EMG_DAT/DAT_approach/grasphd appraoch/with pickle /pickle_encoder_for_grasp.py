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
    device = "cpu"
    dataset_path = "/space/chair-nas/tosy/pt_chifoumi"
    train_split, test_split = "train", "test"
    max_samples_train = 100
    max_samples_test = 22
    DIMS = 4000
    K = 4
    Timewindow = 1000000
    num_classes = 3
    batches = 1

    Encoding_Class = GraspHDEventEncoder
    encoding_mode = "encode_grasphd"  # Options: "encode_grasphd", "nsumming"
    training_mode = "centroid"  # Options: "centroid", "iterative"

    print(f"Using Encoding Mode: {encoding_mode} | Training Mode: {training_mode} | Device: {device}")

    # Load datasets (NOW OUTPUTS TENSORS)
    train_loader, train_max_time = load_tensor_dataset(dataset_path, train_split, max_samples_train, batches, device)
    test_loader, test_max_time = load_tensor_dataset(dataset_path, test_split, max_samples_test, batches, device)

    max_time = max(train_max_time, test_max_time)
    print(f"[INFO] Adjusted Global Maximum Timestamp: {max_time} µs")

    encoder = Encoding_Class(480, 640, DIMS, Timewindow, K, device, max_time)
    encoding_methods = {
        "encode_grasphd": encoder.encode_grasphd,
        "nsumming": encoder.encode_grasp_n_summing, }
    encode_method = encoding_methods.get(encoding_mode)
    if encode_method is None: raise ValueError(
        f"Invalid encoding mode '{encoding_mode}'. Choose from: {list(encoding_methods.keys())}")

    # Initialize class vectors for iterative update
    class_vectors = torchhd.random(num_classes, DIMS, "MAP", device=device)
    train_model(train_loader, encode_method, training_mode, class_vectors)
    evaluate_model(class_vectors, test_loader, encode_method)


def load_tensor_dataset(dataset_path, split, max_samples, batch_size, device):
    """
    Loads event tensors and labels directly from .pt files.
    """
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pt')]

    event_list, label_list = [], []
    max_time = 3600000

    for file in files[:max_samples]:
        data = torch.load(file)
        events_tensor, label_tensor = data["events"], data["label"]
        if label_tensor.dim() == 0:
            label_tensor = label_tensor.unsqueeze(0)
        events_tensor = events_tensor.to(device)
        label_tensor = label_tensor.to(device)
        event_list.append(events_tensor)
        label_list.append(label_tensor)

    dataset = TensorDataset(torch.stack(event_list), torch.stack(label_list))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), max_time


def train_model(train_loader, encode_method, training_mode, class_vectors):
    """
    Train using either centroid-based or iterative update(grasphd paper).
    """
    print(f"\n[TRAINING] Using {training_mode} method...\n")

    if training_mode == "centroid":
        classifier = torchhd.models.Centroid(class_vectors.shape[1], class_vectors.shape[0],
                                             device=class_vectors.device)
        with torch.no_grad():
            for batch_events, batch_labels in tqdm(train_loader, desc="Training (Centroid)"):
                encoded_batch = encode_method(batch_events, batch_labels)
                classifier.add(encoded_batch, batch_labels)
        classifier.normalize()

    elif training_mode == "iterative":
        num_epochs = 5  # Adjustable
        for epoch in range(num_epochs):
            for batch_events, batch_labels in tqdm(train_loader, desc=f"Training Iterative (Epoch {epoch + 1})"):
                encoded_batch = encode_method(batch_events, batch_labels)
                for i in range(len(encoded_batch)):
                    prediction = torch.argmax(torchhd.cosine_similarity(class_vectors, encoded_batch[i]))
                    if prediction != batch_labels[i]:
                        class_vectors[batch_labels[i]] += encoded_batch[i]
                        class_vectors[prediction] -= encoded_batch[i]
        class_vectors = torchhd.normalize(class_vectors)


def evaluate_model(class_vectors, test_loader, encode_method):
    """
    cosine similarity.
    """
    print("\n[TESTING] Evaluating Model...\n")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_events, batch_labels in test_loader:
            encoded_test_samples = encode_method(batch_events, batch_labels)
            similarities = torchhd.cosine_similarity(class_vectors, encoded_test_samples)
            predicted_labels = torch.argmax(similarities, dim=1)
            correct += (predicted_labels == batch_labels).int().sum().item()
            total += batch_labels.numel()

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[RESULT] Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")


if __name__ == "__main__":
    main()
