import torch
import torchhd
import numpy as np
import os
import pickle
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from grasphdencoding import GraspHDEventEncoder
from torchhd.models import Centroid
import random

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)

def main():
    device = "cpu" #if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    split_name = "train"
    max_samples = 11
    DIMS = 3000
    K = 3
    Timewindow = 30000

    dataset = load_pickle_dataset(dataset_path, split=split_name, max_samples=max_samples)
    encoder = GraspHDEventEncoder(height=480, width=640, dims=DIMS, time_subwindow=Timewindow, k=K, device=device)

    encoded_vectors, class_labels = [], []
    for sample_id, (events, class_id) in tqdm(enumerate(dataset), total=len(dataset), desc="Encoding Samples"):
        encoded_sample = encoder.encode_grasphd(events, class_id)
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)

    encoded_matrix = torch.stack(encoded_vectors)

    # **Train Centroid Classifier**
    model = Centroid(DIMS, len(set(class_labels)))
    with torch.no_grad():
        for vec, label in zip(encoded_matrix, class_labels):
            model.add(vec, label)
    model.normalize()

    similarity_matrix = torchhd.cosine_similarity(encoded_matrix, model.weight)

    plot_with_parameters(similarity_matrix, class_labels, K, Timewindow, DIMS, max_samples)

def plot_with_parameters(similarity_matrix, class_labels, k, Timewindow, dims, max_samples):
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
    plt.title(f"Cosine Similarity GRASPHD Heatmap (k={k}, dims={dims}, timewindow= {Timewindow}, samples={max_samples})")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")
    plt.show()
def load_pickle_dataset(dataset_path, split, max_samples):
    """
    Returns List[Tuple]: A list of tuples (events, class_id), where events are the event tuples (t, (x, y), p).
    """
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pkl')]
    random.shuffle(files)
    dataset = []
    for file in files[:max_samples]:
        with open(file, 'rb') as f:
            events, class_id = pickle.load(f)
        dataset.append((events, class_id))
    print(f"Loaded {len(dataset)} samples from {split} split.")
    return dataset


if __name__ == "__main__":
    main()
