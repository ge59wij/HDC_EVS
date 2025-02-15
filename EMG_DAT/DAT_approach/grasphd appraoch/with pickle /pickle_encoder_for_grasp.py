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
import torchmetrics
import torchhd.utils
import matplotlib.pyplot as plt

TRAINING_METHOD = "adaptive"  # Options: "centroid", "adaptive", "iterative"
LEARNING_RATE = 0.5  # For adaptive & iterative methods

def main():
    device = "cuda" #if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    max_samples_train = 100
    max_samples_test = 20
    DIMS = 4000
    K = 3
    Timewindow = 50000

    dataset_train = load_pickle_dataset(dataset_path, split="train", max_samples=max_samples_train)
    dataset_test = load_pickle_dataset(dataset_path, split="test", max_samples=max_samples_test)

    max_time_train = get_max_time(dataset_train)
    max_time_test = get_max_time(dataset_test)
    max_time = max(max_time_train, max_time_test)
    print(f"[INFO] Computed max_time: {max_time} (Train: {max_time_train}, Test: {max_time_test})")

    encoder = GraspHDEventEncoder(height=480, width=640, dims=DIMS, time_subwindow=Timewindow, k=K, device=device, max_time= max_time)

    # **Encode Training Data**
    encoded_vectors, class_labels = [], []
    for sample_id, (events, class_id) in tqdm(enumerate(dataset_train), total=len(dataset_train),
                                              desc="Encoding Samples"):
        encoded_sample = encoder.encode_grasphd(events, class_id)
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)

    # **Batch Processing: Convert Lists to Tensor**
    encoded_matrix = torch.stack(encoded_vectors)
    print(f"Encoded Matrix Stats - Min: {encoded_matrix.min()}, Max: {encoded_matrix.max()}")
    label_tensor = torch.tensor(class_labels, dtype=torch.long, device=device)  # Keep labels as tensor

    model = train_model(encoded_matrix, label_tensor, DIMS, len(set(class_labels)), TRAINING_METHOD)

    '''
    # **Train Centroid Classifier (Batch)**
    model = Centroid(DIMS, len(set(class_labels)))
    with torch.no_grad():
        model.add(encoded_matrix, label_tensor)  # Batch adding instead of looping one by one
        model.normalize()
    '''

    # **Testing Phase**
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=len(set(class_labels)))
    encoded_test_vectors, test_labels = [], []

    for sample_id, (events, class_id) in tqdm(enumerate(dataset_test), total=len(dataset_test),
                                              desc="Encoding Test Samples"):
        encoded_sample = encoder.encode_grasphd(events, class_id)
        encoded_test_vectors.append(encoded_sample)
        test_labels.append(class_id)

    encoded_test_matrix = torch.stack(encoded_test_vectors)
    test_label_tensor = torch.tensor(test_labels, dtype=torch.long, device=device)

    # **Compute Similarity & Track Accuracy**
    with torch.no_grad():
        output = model(encoded_test_matrix, dot=True)  #
        accuracy.update(output.cpu(), test_label_tensor.cpu())

    print(f"Testing Accuracy: {(accuracy.compute().item() * 100):.3f}%")

    # **Plot Similarity Heatmap**
    #similarity_matrix = torchhd.cosine_similarity(encoded_matrix, model.weight)
    #plot_with_parameters(similarity_matrix, class_labels, K, Timewindow, DIMS, max_samples_train)
    # **Visualizations**
    plot_pairwise_similarity(encoded_matrix, class_labels)



def plot_pairwise_similarity(encoded_matrix, class_labels):
    plt.figure(figsize=(10, 8))
    torchhd.utils.plot_pair_similarity(encoded_matrix)
    num_samples = encoded_matrix.shape[0]
    tick_positions = np.arange(num_samples)  # Tick positions
    class_labels_str = [str(label) for label in class_labels]  # Convert to string labels
    plt.xticks(tick_positions, class_labels_str, rotation=90)
    plt.yticks(tick_positions, class_labels_str)
    plt.xlabel("Class ID")
    plt.ylabel("Class ID")
    plt.title("Pairwise Cosine Similarity of Encoded Samples")
    plt.colorbar()
    plt.show()


def train_model(encoded_matrix, label_tensor, dims, num_classes, method):
    """Train a model using the specified method."""
    model = Centroid(dims, num_classes)

    with torch.no_grad():
        if method == "centroid":
            print("Training with Centroid")
            model.add(encoded_matrix, label_tensor)  # Centroid learning
        elif method == "adaptive":
            print("Training with AdaptHD")
            model.add_adapt(encoded_matrix, label_tensor, lr=LEARNING_RATE)  # Adaptive learning
        elif method == "iterative":
            print("Training with Iterative Learning (OnlineHD)")
            model.add_online(encoded_matrix, label_tensor, lr=LEARNING_RATE)  # Iterative refinement
        else:
            raise ValueError("Invalid training method. Choose from 'centroid', 'adaptive', or 'iterative'.")
    model.normalize()
    return model
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
    plt.title(f"Cosine Similarity GRASPHD Heatmap (k={k}, dims={dims}, timewindow={Timewindow}, samples={max_samples})")
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
def get_max_time(dataset):
    """
    Extracts the maximum timestamp from the dataset,stored as the first element in the event tuples.
    """
    max_time = 0
    for events, _ in dataset:
        if len(events) > 0:  # Properly check if events list is non-empty
            last_timestamp = events[-1][0]  # First element of the last tuple
            max_time = max(max_time, last_timestamp)
    return max_time

if __name__ == "__main__":
    main()