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
    max_samples_train = 10
    max_samples_test = 23
    DIMS = 4000
    K = 4
    Timewindow = 1000000
    num_classes = 3
    batches = 1

    Encoding_Class = GraspHDEventEncoder
    encoding_mode = "encode_grasphd"  # Options: "encode_grasphd", "nsumming"
    training_mode = "adaptive"  # "centroid" or "adaptive"
    use_iterative_retrain = True

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

    # Initialize class vectors for adaptive update
    class_vectors = torchhd.random(num_classes, DIMS, "MAP", device=device)
    class_vectors = train_model(train_loader, encode_method, training_mode, class_vectors, use_iterative_retrain)
    evaluate_model(class_vectors, test_loader, encode_method)


def load_tensor_dataset(dataset_path, split, max_samples, batch_size, device):
    """
    Loads event tensors and labels directly from .pt files.
    """
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pt')]

    event_list, label_list, mask_list = [], [], []
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
        mask_list.append(torch.ones(events_tensor.shape[0], dtype=torch.bool, device=device))  # Create mask

    # Pad event tensors and masks
    padded_events = pad_sequence(event_list, batch_first=True, padding_value=0)
    padded_mask = pad_sequence(mask_list, batch_first=True, padding_value=False)  # False = ignored

    dataset = TensorDataset(padded_events, padded_mask, torch.stack(label_list))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), max_time


def train_model(train_loader, encode_method, training_mode, class_vectors, use_iterative_retrain=False):
    """
    Train using either centroid-based or adaptive update (grasphd paper).
    Adaptive learning updates class hypervectors based on similarity.
    Iterative retraining (if enabled) refines them further.
    """
    print(f"\n[TRAINING] Using {training_mode} method | Iterative Retraining: {use_iterative_retrain}\n")

    l = 0.1  # Learning rate

    if training_mode == "centroid":
        classifier = torchhd.models.Centroid(class_vectors.shape[1], class_vectors.shape[0],
                                             device=class_vectors.device)
        with torch.no_grad():
            for batch_events, batch_mask, batch_labels in tqdm(train_loader, desc="Training (Centroid)"):
                encoded_batch = encode_method(batch_events, batch_mask, batch_labels)  # ✅ Pass `batch_mask`
                classifier.add(encoded_batch, batch_labels)
        classifier.normalize()

    elif training_mode == "adaptive":
        # **Adaptive Training (Initial Learning)**
        for batch_events, batch_mask, batch_labels in tqdm(train_loader, desc="Training Adaptive"):
            batch_labels = batch_labels.view(-1)  # Ensure batch_labels is at least 1D
            encoded_batch = encode_method(batch_events, batch_mask, batch_labels)  #  Pass `batch_mask`
            for i in range(len(encoded_batch)):
                w_correct = torchhd.cosine_similarity(class_vectors[batch_labels[i]], encoded_batch[i])
                class_vectors[batch_labels[i]] += l * w_correct * encoded_batch[i]

        # **Iterative Retraining (Optional)**
        if use_iterative_retrain:
            for epoch in range(4):  # Number of epochs
                for batch_events, batch_mask, batch_labels in tqdm(train_loader,
                                                                   desc=f"Iterative Retraining (Epoch {epoch + 1})"):
                    encoded_batch = encode_method(batch_events, batch_mask, batch_labels)  # ✅ Pass `batch_mask`
                    for i in range(len(encoded_batch)):
                        prediction = torch.argmax(torchhd.cosine_similarity(class_vectors, encoded_batch[i]))

                        if prediction != batch_labels[i]:  # Only update if misclassified
                            w_correct = torchhd.cosine_similarity(class_vectors[batch_labels[i]], encoded_batch[i])
                            w_wrong = torchhd.cosine_similarity(class_vectors[prediction], encoded_batch[i])

                            class_vectors[batch_labels[i]] += l * (1 - w_correct) * encoded_batch[i]
                            class_vectors[prediction] -= l * (1 - w_wrong) * encoded_batch[i]

        class_vectors = torchhd.normalize(class_vectors)

    print(f"\n[INFO] Final Class Hypervectors (Norm Check): {torch.norm(class_vectors, dim=1)}")
    return class_vectors


def evaluate_model(class_vectors, test_loader, encode_method):
    """
    cosine similarity.
    """
    print("\n[TESTING] Evaluating Model...\n")
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_events, batch_mask, batch_labels in test_loader:
            encoded_test_samples = encode_method(batch_events, batch_mask, batch_labels)  # Pass the mask
            similarities = torchhd.cosine_similarity(class_vectors, encoded_test_samples)
            predicted_labels = torch.argmax(similarities, dim=1)
            correct += (predicted_labels == batch_labels).int().sum().item()
            total += batch_labels.numel()

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n[RESULT] Test Accuracy: {accuracy:.2f}% ({correct}/{total} correct)\n")


if __name__ == "__main__":
    main()
