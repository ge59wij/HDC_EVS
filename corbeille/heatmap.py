import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def compute_cosine_distance_matrix(hv_batch):
    """
    Compute the pairwise cosine distance matrix between hypervectors.
    hv_batch (torch.Tensor): Tensor of shape [N, D] where N is the number of hypervectors and D is the dimensionality.
    numpy.ndarray: A matrix of shape [N, N] containing cosine distances.
    """
    hv_norm = torch.nn.functional.normalize(hv_batch, p=2, dim=1)
    similarity = torch.matmul(hv_norm, hv_norm.T)
    distance = 1 - similarity
    return distance.cpu().numpy()
def compute_class_centroids(hv_batch, labels):
    """
        hv_batch (torch.Tensor): Tensor of shape [N, D] where N is the number of hypervectors and D is the dimensions
        labels (torch.Tensor): Tensor of shape [N] containing class labels.

        torch.Tensor: Tensor of shape [C, D] where C is the number of unique classes and D is the dimensionality.
    """
    unique_labels = torch.unique(labels)
    centroids = []
    for label in unique_labels:
        class_hvs = hv_batch[labels == label]
        centroid = class_hvs.mean(dim=0)
        centroids.append(centroid)
    return torch.stack(centroids)
def plot_distance_matrix(distance_matrix, labels, title="Cosine Distance Matrix", save_path=None):
    """
        distance_matrix (numpy.ndarray): 2D array of distances.
        labels (list or numpy.ndarray)
    """
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(distance_matrix, cmap="viridis", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Sample Index (class label)")
    plt.ylabel("Sample Index (class label)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
def visualize_sample_hv_distances(dataloader, encoder, sample_size, save_path=None):
    hv_list = []
    label_list = []
    collected = 0

    label_map = dataloader.dataset.label_map

    # Initialize tqdm progress bar
    total_samples = min(sample_size, len(dataloader.dataset))
    progress_bar = tqdm(total=total_samples, desc="Encoding Samples", unit="sample")

    for batch_data, batch_labels in dataloader:
        for sample, label in zip(batch_data, batch_labels):
            try:
                hv = encoder.encode_sample(sample)
                hv_list.append(hv)
                label_list.append(int(label))
                collected += 1
                progress_bar.update(1)

                if collected >= sample_size:
                    progress_bar.close()
                    break
            except Exception as e:
                print(f"Error during encoding sample: {e}")
                continue
        if collected >= sample_size:
            break

    if len(hv_list) == 0:
        print("No hypervectors collected for visualization.")
        return
    if len(hv_list) < sample_size:
        print(f"Warning: Collected only {len(hv_list)} samples, less than requested {sample_size}.")

    # Stack hypervectors and compute cosine distance matrix
    hv_batch = torch.stack(hv_list)
    distance_matrix = compute_cosine_distance_matrix(hv_batch)

    # Convert numeric labels to string labels using `label_map`
    label_strings = [label_map[str(label)] if str(label) in label_map else str(label) for label in label_list]

    # Plot the distance matrix
    plot_distance_matrix(distance_matrix, label_strings, title="Cosine Distance Matrix of Sample HVs", save_path=save_path)
