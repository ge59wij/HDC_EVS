import os
import pickle
import torch
from tqdm import tqdm
import seaborn as sns
from grasphdencoding import GraspHDEventEncoder
import torchhd
import random
import matplotlib.pyplot as plt
torch.set_printoptions(sci_mode=False)
import numpy as np
np.set_printoptions(suppress=True, precision=8)


def main():
    device = "cpu", #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    split_name = "val"
    max_samples = 110
    DIMS = 5000
    K= 3
    Timewindow = 30000

    print(f"K:{K}, D:{DIMS}, #samples:{max_samples}, Timesubwindow: {Timewindow}")

    dataset = load_pickle_dataset(dataset_path, split=split_name, max_samples=max_samples)
    random.shuffle(dataset)

    encoder = GraspHDEventEncoder(height=480, width=640, dims=DIMS, time_subwindow= Timewindow , k=K, device="cpu")

    encoded_vectors = []
    class_labels = []

    for sample_id, (events, class_id) in tqdm(enumerate(dataset), total=len(dataset), desc="\n Encoding Samples"):
        print(f"Encoding Sample {sample_id} | Class: {class_id}")
        formatted_events = [
            (
                float(event["t"]),  # Ensure timestamp is float32
                (int(event["x"]), int(event["y"])),  #uint16 -> int
                int(event["p"])  #uint8 -> int
            )
            for event in events
        ]
        encoded_sample = encoder.encode_grasphd(formatted_events, class_id)
        encoded_sample = encoded_sample.squeeze() # .to(device).squeeze()
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)
    print("\nEncoding Complete. Generating similarity heatmap...\n")

    print('\n-----------\n')

    encoded_matrix = torch.stack(encoded_vectors)
    similarity_matrix = torchhd.cosine_similarity(encoded_matrix, encoded_matrix)
    plot_with_parameters(similarity_matrix, class_labels, K, Timewindow, DIMS, max_samples)
    analyze_similarities(similarity_matrix, class_labels,Timewindow , K, DIMS, max_samples)

def analyze_similarities(similarity_matrix, class_labels, Timewindow, k, dims, max_samples):
    class_labels = torch.tensor(class_labels)
    unique_labels = torch.unique(class_labels)
    print(f"\n K:{k}, Timewindow:{Timewindow}, Numpber of samples: {max_samples}, HV Dimensions: {dims}")
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
    for file in files[:max_samples]:  # take max_samples after shuffling
        with open(file, 'rb') as f:
            events, class_id = pickle.load(f)
        dataset.append((events, class_id))
    print(f"Loaded {len(dataset)} samples from {split} split.")
    return dataset


if __name__ == "__main__":
    main()
