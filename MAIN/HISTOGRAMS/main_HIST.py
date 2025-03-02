import os
import h5py
import numpy as np
import torch
import torchhd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import defaultdict
from torchhd.models import Centroid
from HIST_Encoder import HISTEncoder
import random
from sklearn.model_selection import train_test_split
import collections
from sklearn.model_selection import train_test_split
from collections import Counter
import torchmetrics

# Constants
BACKGROUND_LABEL = 404
dataset_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/Bin Labeled/"
Train_split = "VAL BIN LABELED"
test_dataset = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/test/"
DIMS = 4000
BATCH_SIZE = 1
NUM_RANDOM_VECTORS = 50
EVENT_THRESHOLD = 100  # Bins with total ON+OFF events < threshold => background (404)


WINDOW_SIZE = 10  # Total bins processed together
NGRAM_SIZE = 4  # Temporal context depth inside window
OVERLAP = 2  # Stride = window_size - overlap
def bin_labeling(files):
    """
    Processes the test dataset and labels each bin based on event count.
    If ON+OFF event count < EVENT_THRESHOLD, assign label 404 (background).
    Otherwise, assign the sample's class ID to all bins.
    """
    print("\n[INFO] Processing test dataset for per-bin labeling...")

    labeled_files = []
    for file in tqdm(files, desc="Labeling Bins"):
        with h5py.File(file, "r+") as f:
            event_data = f["data"][:]  # Shape: (T, 2, H, W)
            class_id = f["class_id"][()]  # Integer class label for full sample

            # Compute total event count per bin (sum ON & OFF)
            total_events_per_bin = np.sum(event_data, axis=(1, 2, 3))  # Shape: (T,)

            # Assign per-bin labels
            bin_labels = np.where(total_events_per_bin < EVENT_THRESHOLD, BACKGROUND_LABEL, class_id)
            if "labels" in f:
                del f["labels"]  # Delete existing labels if present
            f.create_dataset("labels", data=bin_labels, dtype="int32")

            labeled_files.append(file)

    print("[INFO] Test dataset labeling complete!")
    return labeled_files


def load_dataset(dataset_folder, is_test=False, max_test_samples=30):
    """Loads the dataset and applies bin labeling for test dataset.
       Limits the test dataset to `max_test_samples` while ensuring class balance.
    """
    files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.h5')]

    if is_test:
        # 1. **Get class labels for each file BEFORE labeling**
        class_labels = {}
        for file in files:
            with h5py.File(file, "r") as f:
                class_labels[file] = int(f["class_id"][()])  # Ensure it's a Python integer

        # 2. **Sort samples into class-wise bins**
        class_buckets = defaultdict(list)
        for file, class_id in class_labels.items():
            class_buckets[class_id].append(file)

        # 3. **Select samples for each class in a balanced way**
        balanced_test_files = []
        num_classes = len(class_buckets)
        samples_per_class = max_test_samples // num_classes  # Distribute samples evenly

        for class_id, file_list in class_buckets.items():
            selected_files = random.sample(file_list, min(samples_per_class, len(file_list)))
            balanced_test_files.extend(selected_files)

        # **Ensure we do not exceed max_test_samples**
        if len(balanced_test_files) > max_test_samples:
            balanced_test_files = balanced_test_files[:max_test_samples]

        print(f"[INFO] Selected {len(balanced_test_files)} balanced test samples across {num_classes} classes.")

        # 4. **Apply bin labeling only to the selected test samples**
        files = bin_labeling(balanced_test_files)

    return files


def process_sample(event_data, class_ids, encoder):
    '''Uses a sliding window approach: calls encode_window() from HISTEncoder for each window.'''
    T = event_data.shape[0]
    gesture_hvs = []

    # **Sliding window processing**
    for window_idx, start in enumerate(range(0, T - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP)):
        end = start + WINDOW_SIZE
        window_data = event_data[start:end]
        window_labels = class_ids[start:end]

        #print(f"[DEBUG] Processing window {window_idx} ({start}-{end}) of sample with {T} bins.")

        gesture_hv = encoder.encode_window(window_data, window_labels)

        if gesture_hv is not None:
            dominant_class = window_labels[window_labels != BACKGROUND_LABEL].mode().values.item()
            gesture_hvs.append((gesture_hv, dominant_class))

    if len(gesture_hvs) == 0:
        print("[WARNING] No valid gesture HVs found in this sample.")
        return [(torch.zeros(encoder.dims, device=encoder.device), -1)]  # **Return an empty vector**

    return gesture_hvs  # list of encoded HVs
def validate(dataloader, encoder):
    ''': Extracts class-wise hvs for visualizing encoded data.'''
    class_vectors = defaultdict(list)
    for data, labels, _ in dataloader:
        gestures = process_sample(data.squeeze(0), labels.squeeze(0), encoder)
        for hv, cls in gestures:
            class_vectors[cls].append(hv)

    if not class_vectors:  # Ensure it's not empty
        print("[ERROR] No hypervectors generated.")
        return [], []

    return list(class_vectors.values()), list(class_vectors.keys())
def plot_heatmap(vectors_matrix, class_labels):
    unique_classes = set(class_labels)
    selected_vectors = []
    selected_labels = []
    num_per_class = 6
    for cls in unique_classes:
        indices = [i for i, lbl in enumerate(class_labels) if lbl == cls]
        sampled_indices = random.sample(indices, min(num_per_class, len(indices)))  # Pick `num_per_class` samples
        for idx in sampled_indices:
            selected_vectors.append(vectors_matrix[idx])
            selected_labels.append(cls)

    selected_vectors = torch.stack(selected_vectors).cpu()
    similarity_matrix = torchhd.cosine_similarity(selected_vectors, selected_vectors).cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=selected_labels, yticklabels=selected_labels,
                cbar=True, square=True, linewidths=0.5)

    plt.title(f"Cosine Similarity Heatmap (One Sample Per Class)")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")
    plt.show()
def train_model(dataloader, encoder):
    centroids = defaultdict(list)

    all_vectors = []
    all_labels = []

    for data, labels, _ in dataloader:
        hvs = process_sample(data.squeeze(0), labels.squeeze(0), encoder)
        for hv, cls in hvs:
            centroids[cls].append(hv)
            all_vectors.append(hv)  # Store for analysis
            all_labels.append(cls)  # Store corresponding labels

    print("Classes found in training:", centroids.keys())

    model = torchhd.models.Centroid(DIMS, len(centroids))
    with torch.no_grad():
        for cls, vectors in centroids.items():
            centroid = torch.stack(vectors).mean(dim=0)
            model.add(centroid.unsqueeze(0), torch.tensor([cls]))

    return model, all_vectors, all_labels  #

def evaluate_model(dataloader, encoder, model):
    correct, total = 0, 0
    classwise_correct = defaultdict(int)
    classwise_total = defaultdict(int)

    cos_acc, dot_acc, hamming_acc = 0, 0, 0  # Store accuracies for each metric

    for data, labels, _ in dataloader:
        hvs = process_sample(data.squeeze(0), labels.squeeze(0), encoder)
        for hv, true_cls in hvs:
            # 🔹 Cosine Similarity
            cos_similarities = torchhd.cosine_similarity(hv, model.weight)
            cos_pred = cos_similarities.argmax().item()

            #  Dot Product Similarity
            dot_similarities = torch.matmul(hv, model.weight.T)
            dot_pred = dot_similarities.argmax().item()

            #  Hamming Distance Similarity
            hamming_distances = torch.cdist(hv.unsqueeze(0), model.weight, p=0).squeeze(0)
            hamming_pred = hamming_distances.argmin().item()  # Lower Hamming distance = more similar

            # Track accuracy for all 3
            if cos_pred == true_cls:
                cos_acc += 1
            if dot_pred == true_cls:
                dot_acc += 1
            if hamming_pred == true_cls:
                hamming_acc += 1

            classwise_total[true_cls] += 1
            total += 1

    print("\n Classification Accuracy:")
    print(f"  - Cosine Similarity: {cos_acc / total:.3f}")
    print(f"  - Dot Product: {dot_acc / total:.3f}")
    print(f"  - Hamming Distance: {hamming_acc / total:.3f}")

    return cos_acc / total, dot_acc / total, hamming_acc / total

def compute_intra_inter_class_similarities(encoded_vectors, class_labels):
    unique_classes = set(class_labels)
    class_vectors = defaultdict(list)

    for vec, label in zip(encoded_vectors, class_labels):
        class_vectors[label].append(vec)

    intra_class_sim = []
    inter_class_sim = []

    for cls, vectors in class_vectors.items():
        vectors = torch.stack(vectors)

        #  Compute intra-class similarity (cosine)
        intra_sim = torchhd.cosine_similarity(vectors, vectors).mean().item()
        intra_class_sim.append(intra_sim)

        #Compute inter-class similarity
        for other_cls, other_vectors in class_vectors.items():
            if cls != other_cls:
                other_vectors = torch.stack(other_vectors)
                inter_sim = torchhd.cosine_similarity(vectors, other_vectors).mean().item()
                inter_class_sim.append(inter_sim)

    # Print results
    print("\n Similarity Analysis:")
    print(f"  - Intra-Class Similarity (Avg): {sum(intra_class_sim) / len(intra_class_sim):.3f}")
    print(f"  - Inter-Class Similarity (Avg): {sum(inter_class_sim) / len(inter_class_sim):.3f}")





class HDF5Dataset(Dataset):
    def __init__(self, file_list):
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with h5py.File(file_path, "r") as f:
            event_data = torch.tensor(f["data"][:], dtype=torch.float32)
            class_ids = torch.tensor(f["labels"][:], dtype=torch.int32)
        return event_data, class_ids, file_path
def main():
    device = "cpu"
    print(f"Using device: {device}")

    train_files = load_dataset(os.path.join(dataset_path, Train_split), is_test=False)
    test_files = load_dataset(test_dataset, is_test=True, max_test_samples=30)
  # Apply labeling for test data

    dataset_train = HDF5Dataset(train_files)
    dataset_test = HDF5Dataset(test_files)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle in test

    encoder = HISTEncoder(
        height=120, width=160,
        dims=DIMS, device=device, threshold=1 / 16,
        window_size=WINDOW_SIZE, n_gram=NGRAM_SIZE
    )

    print("\n Training Model...\n")
    model, encoded_vectors, class_labels = train_model(dataloader_train, encoder)
    compute_intra_inter_class_similarities(encoded_vectors, class_labels)
    cos_acc, dot_acc, hamming_acc = evaluate_model(dataloader_test, encoder, model)
    print("\n Encoding of train: Scores:")
    print(f"  - Cosine Similarity: {cos_acc * 100:.2f}%")
    print(f"  - Dot Product: {dot_acc * 100:.2f}%")
    print(f"  - Hamming Distance: {hamming_acc * 100:.2f}%")
    if encoded_vectors:
        plot_heatmap(torch.stack(encoded_vectors), class_labels)
    else:
        print("[ERROR] No encoded vectors available for heatmap.")

################training:
    with torch.no_grad():
        for event_data, class_ids, filename in tqdm(dataloader_train, desc="Training"):
            hvs = process_sample(event_data.squeeze(0), class_ids.squeeze(0), encoder)

            for hv, class_id in hvs:
                if hv.shape != (DIMS,):  # Debugging shape
                    print(f"[ERROR] HV has incorrect shape: {hv.shape}, expected {(DIMS,)}")

                class_id_tensor = torch.tensor([class_id], dtype=torch.long, device=hv.device)  # Ensure tensor
                if class_id_tensor.shape != (1,):
                    print(f"[ERROR] Class ID tensor shape mismatch: {class_id_tensor.shape}, expected (1,)")

                model.add(hv.unsqueeze(0), class_id_tensor)
    print("\n Evaluating Model on Test Data...\n")
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=3)

    with torch.no_grad():
        model.normalize()
        for event_data, class_ids, filename in tqdm(dataloader_test, desc="Testing"):
            hvs = process_sample(event_data.squeeze(0), class_ids.squeeze(0), encoder)
            for hv, class_id in hvs:
                output = model(hv, dot=True)
                predicted_class = torch.argmax(output).unsqueeze(0)  # Select most similar class
                accuracy.update(predicted_class.cpu(), torch.tensor([class_id]))

    print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
    print("Model Centroids Stored:", model.weight.shape)
    print("Classes in Model:", list(range(model.weight.shape[0])))

    validate(dataloader_train, encoder)

    evaluate_model(dataloader_test, encoder, model)
    plot_heatmap(torch.stack(encoded_vectors), class_labels)


if __name__ == "__main__":
    main()
