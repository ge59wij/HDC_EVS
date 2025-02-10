import os
import pickle
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from grasphdencoding_seedhvs import GraspHDseedEncoder
from grasphdencoding import GraspHDEventEncoder
import torchhd
import random
import matplotlib.pyplot as plt
from torchhd.utils import plot_pair_similarity
import numpy as np
import torchhd.functional as functional



def load_pickle_dataset(dataset_path, split, max_samples):
    """
    Load event data and class labels from pickle files.
    Returns:
        List[Tuple]: A list of tuples (events, class_id), where events are the event tuples (t, (x, y), p).
    """
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pkl')]

    # Shuffle the file list before loading
    random.shuffle(files)

    dataset = []
    for file in files[:max_samples]:  # Take max_samples after shuffling
        with open(file, 'rb') as f:
            events, class_id = pickle.load(f)
        dataset.append((events, class_id))

    print(f"Loaded {len(dataset)} samples from {split} split.")
    return dataset


def main():
    """Encode samples from pickle files and visualize similarities with a heatmap."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    split_name = "val"
    max_samples = 15
    DIMS = 6000
    K= 200
    dataset = load_pickle_dataset(dataset_path, split=split_name, max_samples=max_samples)
    random.shuffle(dataset)

    encoder = GraspHDEventEncoder(height=480, width=640, dims=DIMS, k=K, device=device)

    encoded_vectors = []
    class_labels = []
    for sample_id, (events, class_id) in tqdm(enumerate(dataset), total=len(dataset), desc="Encoding Samples"):
        print(f"Encoding Sample {sample_id} | Class: {class_id}")
        formatted_events = [
            (
                float(event["t"]),  # Ensure timestamp is float32
                (int(event["x"]), int(event["y"])),  # Ensure x, y are uint16 -> int
                int(event["p"])  # Ensure polarity is uint8 -> int
            )
            for event in events
        ]
        # Encode events
        encoded_sample = encoder.encode_temporal(formatted_events, class_id)
        encoded_sample = encoded_sample.to(device).squeeze()
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)
    print("\nEncoding Complete. Generating similarity heatmap...\n")
    # Convert encoded vectors to a matrix
    encoded_matrix = torch.stack(encoded_vectors)
    # Compute cosine similarity
    similarity_matrix = torchhd.cosine_similarity(encoded_matrix, encoded_matrix)
    plot_with_parameters(similarity_matrix, class_labels, K, DIMS, max_samples)

    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(similarity_matrix.cpu().numpy(), annot=True, cmap="coolwarm", fmt=".2f",
                          xticklabels=class_labels, yticklabels=class_labels, cbar=True)
    plt.title("Cosine Similarity Heatmap of Encoded Samples")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")
    plt.show()
    # Analyze intra-class and inter-class similarities
    analyze_similarities(similarity_matrix, class_labels)
    print('\n-----------\n')
    torchhd_visualizations(encoded_matrix, class_labels)
    visualize_position_hvs(encoder, stride=10)

def analyze_similarities(similarity_matrix, class_labels):
    class_labels = torch.tensor(class_labels)
    unique_labels = torch.unique(class_labels)

    print("\nIntra-class Similarities:")
    for label in unique_labels:
        indices = torch.where(class_labels == label)[0]
        intra_class_sim = similarity_matrix[indices][:, indices].mean().item()
        print(f"Label {label.item()}: {intra_class_sim:.4f}")

    print("\nInter-class Similarities:")
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                indices_i = torch.where(class_labels == label_i)[0]
                indices_j = torch.where(class_labels == label_j)[0]
                inter_class_sim = similarity_matrix[indices_i][:, indices_j].mean().item()
                print(f"Between Label {label_i.item()} and Label {label_j.item()}: {inter_class_sim:.4f}")
def visualize_position_hvs(encoder, stride=10):
    """Visualize correlations between hypervectors for neighboring pixels."""
    positions = [(x, y) for x in range(0, encoder.height, stride) for y in range(0, encoder.width, stride)]
    hv_matrix = []

    for x, y in positions:
        hv = encoder.get_position_hv(x, y)
        hv_matrix.append(hv.cpu().numpy())

    hv_matrix = np.array(hv_matrix)

    # Compute cosine similarity between all hypervectors
    similarity_matrix = np.corrcoef(hv_matrix)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap="coolwarm", square=True)
    plt.title("Cosine Similarity Between Position Hypervectors")
    plt.show()

def torchhd_visualizations(encoded_vectors, class_labels):
    class_labels_str = [f"Class {label}" for label in class_labels]

    # Create a plot for pairwise similarity
    fig, ax = plt.subplots(figsize=(12, 10))
    plot = plot_pair_similarity(encoded_vectors, ax=ax)

    # Add class labels as ticks
    ax.set_xticks(range(len(class_labels_str)))
    ax.set_xticklabels(class_labels_str, rotation=90)
    ax.set_yticks(range(len(class_labels_str)))
    ax.set_yticklabels(class_labels_str)

    # Title and axis labels
    ax.set_title("TorchHD Pairwise Cosine Similarity Heatmap with Class Labels")
    ax.set_xlabel("Sample Index (Class ID)")
    ax.set_ylabel("Sample Index (Class ID)")

    plt.show()
def plot_with_parameters(similarity_matrix, class_labels, k, dims, max_samples):
    """
    Plot the cosine similarity heatmap with parameters in the title.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix.cpu().numpy(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
    )
    plt.title(f"Cosine Similarity Heatmap (k={k}, dims={dims}, samples={max_samples})")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")
    plt.show()

if __name__ == "__main__":
    main()
