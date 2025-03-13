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
from sklearn.manifold import TSNE
import torchmetrics
import datetime
import json

BACKGROUND_LABEL = 404
LOGS_DIR = "/space/chair-nas/tosy/logs_encodings_histogram/"
dataset_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/Bin Labeled/"
Train_split = "VAL BIN LABELED"
test_dataset = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/test/"
NUM_TRAIN_METHODS = ["centroid", "adaptive", "iterative"]



DEBUG_MODE = False
K = 6
DEFAULT_HEATMAP_SAMPLES = 30  # Maximum samples per class for similarity calculations
DIMS = 6000  ##at least 500 for permuation/thermometer
EVENT_THRESHOLD = 1/ 16  # if below this, then counted as no gesture #for bin labeling in the test data and encoding
WINDOW_SIZE = 100
NGRAM_SIZE = 10
OVERLAP = 2
method_encoding = "thermometer"  # "thermometer" or "linear" or "eventhd_timeinterpolation" "eventhd_timepermutation"


def create_run_directory():
    """Creates a unique directory for each run and returns its path."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(LOGS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
def bin_labeling(files):
    """Processes the test dataset and labels each bin based on event count."""
    print("\n[INFO] Processing test dataset for per-bin labeling...")
    labeled_files = []
    for file in tqdm(files, desc="Labeling Bins"):
        with h5py.File(file, "r+") as f:
            event_data = f["data"][:]  # Shape: (T, 2, H, W)
            class_id = f["class_id"][()]  # Integer class label for full sample
            total_events_per_bin = np.sum(event_data, axis=(1, 2, 3))  # Shape: (T,)
            bin_labels = np.where(total_events_per_bin < EVENT_THRESHOLD, BACKGROUND_LABEL, class_id)
            if "labels" in f:
                del f["labels"]  # Delete existing labels if present
            f.create_dataset("labels", data=bin_labels, dtype="int32")
            labeled_files.append(file)
    print("[INFO] Test dataset labeling complete!")
    return labeled_files
def load_dataset(dataset_folder, is_test=False, max_test_samples=30):
    """Loads dataset and applies bin labeling if test."""
    files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.h5')]
    if is_test:
        files = bin_labeling(files[:max_test_samples])
    return files
def process_sample(event_data, class_ids, encoder, sample_name, debug):
    """
    Encodes event data into hypervectors using sliding windows.

    Args:
        event_data (torch.Tensor): Event data [T, 2, H, W]
        class_ids (torch.Tensor): Class IDs for each bin
        encoder (HISTEncoder): Encoder instance
        sample_name (str): Sample name for debugging

    Returns:
        list: List of (hypervector, class_id) tuples
    """
    T = event_data.shape[0]
    gesture_hvs = []

    if debug:
        print(f"\n[SAMPLE] Processing {sample_name} with {T} time bins")
        print(f"[SAMPLE] Creating windows with size={WINDOW_SIZE}, overlap={OVERLAP}")

    window_count = 0
    for start in range(0, T - WINDOW_SIZE + 1, WINDOW_SIZE - OVERLAP):
        end = start + WINDOW_SIZE
        if debug:
            print(f"\n[WINDOW] Processing window {window_count + 1} (bins {start}-{end - 1})")

        window_data = event_data[start:end]
        window_labels = class_ids[start:end]

        # Count non-background labels for debugging
        valid_bins = (window_labels != BACKGROUND_LABEL).sum().item()
        if debug:
            print(f"[WINDOW] Window has {valid_bins}/{WINDOW_SIZE} non-background bins")

        # Encode the window
        gesture_hv = encoder.encode_window(window_data, window_labels)

        if gesture_hv is not None:
            # Get the most common non-background label
            valid_labels = window_labels[window_labels != BACKGROUND_LABEL]
            if len(valid_labels) > 0:
                label_counts = torch.bincount(valid_labels)
                id_class = label_counts.argmax().item()
                gesture_hvs.append((gesture_hv, id_class))

                if debug:
                    print(f"[WINDOW] Successfully encoded window with class ID: {id_class}")
        else:
            if debug:
                print("[WINDOW] Failed to encode window (no valid gesture data)")

        window_count += 1

    if debug:
        print(f"[SAMPLE] Created {len(gesture_hvs)} encoded windows")

    return gesture_hvs if gesture_hvs else [(torch.zeros(encoder.dims), -1)]
def save_run_info(run_dir, params, metrics, results):
    """Saves run parameters and final results in the given run directory."""
    filename = os.path.join(run_dir, "run_info.json")

    run_data = {
        "parameters": params,
        "metrics": metrics,
        "model_results": results
    }

    with open(filename, "w") as f:
        json.dump(run_data, f, indent=4)

    print(f"[INFO] Run details saved to {filename}")
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
def encode_dataset(dataloader, encoder, debug):
    """
    Encodes the entire dataset before training.

    Args:
        dataloader (DataLoader): Dataset loader
        encoder (HISTEncoder): Encoder instance
        debug (bool): Enable detailed debugging

    Returns:
        tuple: (encoded_vectors, class_labels)
    """
    encoded_vectors, class_labels = [], []
    for batch_idx, (data, labels, file_path) in enumerate(tqdm(dataloader, desc="Encoding Dataset")):
        sample_name = os.path.basename(file_path[0])

        if debug:
            print(f"\n[DATASET] Processing sample {batch_idx + 1}: {sample_name}")
            unique_labels = torch.unique(labels).tolist()
            print(f"[DATASET] Sample has labels: {unique_labels}")

        hvs = process_sample(
            data.squeeze(0),
            labels.squeeze(0),
            encoder,
            sample_name=sample_name,
            debug=debug and batch_idx < 3  # Only debug the first 3 samples in detail
        )

        valid_hvs = [(hv, cls) for hv, cls in hvs if cls != -1]

        if debug:
            print(f"[DATASET] Sample produced {len(valid_hvs)} valid hypervectors")

        for hv, cls in valid_hvs:
            encoded_vectors.append(hv)
            class_labels.append(cls)

    if not encoded_vectors:
        raise ValueError("No valid hypervectors were generated from the dataset")

    return torch.stack(encoded_vectors), class_labels
def train_model(encoded_vectors, class_labels, method, debug):
    """Trains a model using different methods."""
    unique_classes = sorted(list(set(class_labels)))
    num_classes = len(unique_classes)

    if debug:
        print(f"\n[TRAINING] Using method: {method}")
        print(f"[TRAINING] Number of classes: {num_classes}")
        print(f"[TRAINING] Unique classes: {unique_classes}")
        print(f"[TRAINING] Total vectors: {len(encoded_vectors)}")

        # Count vectors per class
        class_counts = defaultdict(int)
        for cls in class_labels:
            class_counts[cls] += 1
        print(f"[TRAINING] Vectors per class: {dict(class_counts)}")

    model = Centroid(DIMS, num_classes)
    labels_tensor = torch.tensor(class_labels, dtype=torch.long)

    with torch.no_grad():
        if method == "centroid":
            model.add(encoded_vectors, labels_tensor)
        elif method == "adaptive":
            model.add_adapt(encoded_vectors, labels_tensor, lr=0.5)
        elif method == "iterative":
            model.add_online(encoded_vectors, labels_tensor, lr=0.5)

    model.normalize()

    if debug:
        # Evaluate on training data
        with torch.no_grad():
            sim = model(encoded_vectors)
            preds = sim.argmax(1)
            correct = (preds == labels_tensor).sum().item()
            acc = correct / len(labels_tensor) * 100
            print(f"[TRAINING] Training accuracy: {acc:.2f}%")

    return model


def compute_intra_inter_class_similarities(vectors, labels, run_dir, filename="similarity_heatmap.png",
                                           sample_limit=DEFAULT_HEATMAP_SAMPLES, debug=False):
    """
    Computes class-wise intra/inter similarities with detailed per-class analysis.

    Args:
        vectors (torch.Tensor): Encoded hypervectors
        labels (list): Class labels
        run_dir (str): Directory to save visualizations
        filename (str): Filename for the heatmap
        sample_limit (int): Maximum number of samples per class to use for similarity computation
        debug (bool): Enable detailed debugging

    Returns:
        tuple: (intra_sim, inter_sim)
    """
    unique_classes = sorted(list(set(labels)))

    # Group vectors by class and limit samples per class if needed
    class_vectors = {}
    for cls in unique_classes:
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        # Limit to sample_limit vectors per class
        if len(cls_indices) > sample_limit:
            cls_indices = cls_indices[:sample_limit]
        class_vectors[cls] = vectors[cls_indices]

    intra_sim = {}
    inter_sim = {}

    if debug:
        print("\n[SIMILARITY] Intra-Class and Inter-Class Similarity Analysis:")
        print(f"[SIMILARITY] Using up to {sample_limit} samples per class")

    # Compute intra-class similarities
    for cls, vecs in class_vectors.items():
        if len(vecs) > 1:  # Need at least 2 vectors to compute similarity
            # Compute all pairwise similarities within class
            sim_matrix = torchhd.cosine_similarity(vecs, vecs)
            # Exclude self-similarities (diagonal)
            mask = ~torch.eye(len(vecs), dtype=torch.bool)
            intra = sim_matrix[mask].mean().item()
            intra_sim[cls] = intra
            if debug:
                print(f"  - Class {cls}: Intra-Class Similarity: {intra:.3f} ({len(vecs)} vectors)")

    # Compute inter-class similarities
    for cls in unique_classes:
        inter_sim[cls] = {}
        for other_cls in unique_classes:
            if cls != other_cls and len(class_vectors[cls]) > 0 and len(class_vectors[other_cls]) > 0:
                inter = torchhd.cosine_similarity(
                    class_vectors[cls], class_vectors[other_cls]
                ).mean().item()
                inter_sim[cls][other_cls] = inter
                if debug:
                    print(f"    - Class {cls} vs. Class {other_cls}: Inter-Class Similarity: {inter:.3f}")

    # Create and save similarity heatmap
    plot_path = os.path.join(run_dir, filename)
    create_similarity_heatmap(intra_sim, inter_sim, plot_path)

    return intra_sim, inter_sim


def create_similarity_heatmap(intra_sim, inter_sim, plot_path):
    """
    Creates and saves a heatmap of class similarities.

    Args:
        intra_sim (dict): Dictionary of intra-class similarities
        inter_sim (dict): Dictionary of inter-class similarities
        plot_path (str): Path to save the heatmap
    """
    classes = sorted(list(intra_sim.keys()))
    n_classes = len(classes)
    sim_matrix = np.zeros((n_classes, n_classes))

    # Fill diagonal with intra-class similarities
    for i, cls in enumerate(classes):
        sim_matrix[i, i] = intra_sim[cls]

    # Fill off-diagonal with inter-class similarities
    for i, cls1 in enumerate(classes):
        for j, cls2 in enumerate(classes):
            if cls1 != cls2:
                sim_matrix[i, j] = inter_sim[cls1].get(cls2, 0)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title("Class Similarity Matrix")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[VISUALIZATION] Similarity heatmap saved to: {plot_path}")


def visualize_hypervectors(vectors, labels, run_dir, filename="hypervector_visualization.png"):
    """Projects hypervectors to 2D using t-SNE and visualizes them."""
    print("[VISUALIZATION] Projecting hypervectors to 2D using t-SNE...")

    # Convert to numpy for t-SNE
    vectors_np = vectors.detach().cpu().numpy()

    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedded = tsne.fit_transform(vectors_np)

    # Create scatter plot
    plt.figure(figsize=(12, 10))
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        plt.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=[colors[i]],
            label=f"Class {label}",
            alpha=0.7
        )
    plot_path = os.path.join(run_dir, filename)
    plt.title("t-SNE Visualization of Gesture Hypervectors")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(run_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"[VISUALIZATION] Saved t-SNE plot: {plot_path}")


def evaluate_model(model, encoded_vectors, true_labels, method_name):
    """Evaluates model performance with detailed metrics."""
    print(f"\n[EVALUATION] Evaluating {method_name} model...")

    # Convert to tensors
    true_labels_tensor = torch.tensor(true_labels, dtype=torch.long)

    # Make predictions
    with torch.no_grad():
        similarities = model(encoded_vectors)
        pred_labels = similarities.argmax(dim=1)

    # Calculate accuracy
    correct = (pred_labels == true_labels_tensor).sum().item()
    total = len(true_labels)
    accuracy = correct / total * 100

    # Calculate per-class metrics
    unique_classes = sorted(list(set(true_labels)))
    class_metrics = {}

    for cls in unique_classes:
        cls_mask = true_labels_tensor == cls
        if cls_mask.sum() > 0:
            cls_correct = (pred_labels[cls_mask] == true_labels_tensor[cls_mask]).sum().item()
            cls_total = cls_mask.sum().item()
            cls_accuracy = cls_correct / cls_total * 100
            class_metrics[cls] = {
                "accuracy": cls_accuracy,
                "count": cls_total
            }

    # Calculate confusion matrix
    confusion = torch.zeros(len(unique_classes), len(unique_classes), dtype=torch.long)
    for pred, true in zip(pred_labels, true_labels_tensor):
        pred_idx = unique_classes.index(pred.item())
        true_idx = unique_classes.index(true.item())
        confusion[pred_idx, true_idx] += 1

    # Print results
    print(f"[EVALUATION] Overall Accuracy: {accuracy:.2f}%")
    print("[EVALUATION] Per-Class Metrics:")
    for cls, metrics in class_metrics.items():
        print(f"  - Class {cls}: Accuracy: {metrics['accuracy']:.2f}% ({metrics['count']} samples)")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion.numpy(),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_classes,
        yticklabels=unique_classes
    )
    plt.title(f"Confusion Matrix - {method_name}")
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{method_name}.png")
    plt.close()

    return {
        "accuracy": accuracy,
        "class_metrics": class_metrics,
        "confusion": confusion
    }


def compute_additional_metrics(vectors, labels, run_dir, sample_limit=DEFAULT_HEATMAP_SAMPLES):
    """
    Compute additional metrics for the encoded hypervectors:
    - Cosine similarity matrix
    - Dot product matrix

    Args:
        vectors (torch.Tensor): Encoded hypervectors
        labels (list): Class labels for each vector
        run_dir (str): Directory to save visualizations
        sample_limit (int): Maximum number of samples per class to use
    """
    unique_classes = sorted(list(set(labels)))
    num_classes = len(unique_classes)

    # Group vectors by class and limit samples per class
    class_vectors = {}
    class_samples = {}

    for cls in unique_classes:
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        if len(cls_indices) > sample_limit:
            sampled_indices = cls_indices[:sample_limit]
        else:
            sampled_indices = cls_indices

        class_vectors[cls] = vectors[sampled_indices]
        class_samples[cls] = len(sampled_indices)

    print(f"\n[METRICS] Computing metrics using up to {sample_limit} samples per class")
    for cls, count in class_samples.items():
        print(f"  - Class {cls}: {count} samples")

    # Calculate centroids for each class
    class_centroids = {cls: vecs.mean(dim=0) for cls, vecs in class_vectors.items() if len(vecs) > 0}

    # Initialize matrices for metrics between centroids
    cosine_sim_matrix = np.zeros((num_classes, num_classes))
    dot_product_matrix = np.zeros((num_classes, num_classes))

    # Initialize matrices for metrics between individual vectors (mean of all pairwise comparisons)
    pairwise_cosine_matrix = np.zeros((num_classes, num_classes))

    # Compute metrics between class centroids
    #print("\n[METRICS] Computing centroid-based metrics...")
    for i, cls1 in enumerate(unique_classes):
        for j, cls2 in enumerate(unique_classes):
            if cls1 in class_centroids and cls2 in class_centroids:
                # Cosine similarity
                cosine_sim = torchhd.cosine_similarity(
                    class_centroids[cls1].unsqueeze(0),
                    class_centroids[cls2].unsqueeze(0)
                ).item()
                cosine_sim_matrix[i, j] = cosine_sim

                # Dot product
                dot_product = torch.dot(class_centroids[cls1], class_centroids[cls2]).item()
                dot_product_matrix[i, j] = dot_product

    # Compute metrics between individual vectors (all pairwise comparisons)
    print("\n[METRICS] Computing pairwise vector metrics...")
    for i, cls1 in enumerate(unique_classes):
        for j, cls2 in enumerate(unique_classes):
            if cls1 in class_vectors and cls2 in class_vectors:
                # Pairwise cosine similarities
                cosine_sims = torchhd.cosine_similarity(
                    class_vectors[cls1], class_vectors[cls2]
                )

                # For same class, exclude self-similarities
                if i == j and len(cosine_sims) > 1:
                    mask = ~torch.eye(len(cosine_sims), dtype=torch.bool)
                    pairwise_cosine = cosine_sims[mask].mean().item()
                else:
                    pairwise_cosine = cosine_sims.mean().item()

                pairwise_cosine_matrix[i, j] = pairwise_cosine


    # Visualize centroid-based cosine similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cosine_sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=unique_classes,
        yticklabels=unique_classes
    )
    plt.title("Cosine Similarity Between Class Centroids")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "centroid_cosine_similarity.png")
    plt.savefig(plot_path)
    plt.close()

    # Visualize pairwise cosine similarity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pairwise_cosine_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=unique_classes,
        yticklabels=unique_classes
    )
    plt.title("Average Pairwise Cosine Similarity Between Vectors")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "pairwise_cosine_similarity.png")
    plt.savefig(plot_path)
    plt.close()


    # Visualize dot product matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        dot_product_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=unique_classes,
        yticklabels=unique_classes
    )
    plt.title("Dot Product Between Class Centroids")
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    plot_path = os.path.join(run_dir, "dot_product_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    # Print summary statistics
    print("\n[METRICS] Centroid-Based Metrics:")
    print("  Cosine Similarity Matrix:")
    print(f"    - Average intra-class: {np.mean([cosine_sim_matrix[i, i] for i in range(num_classes)]):.3f}")
    print(
        f"    - Average inter-class: {np.mean([cosine_sim_matrix[i, j] for i in range(num_classes) for j in range(num_classes) if i != j]):.3f}")


    print("\n[METRICS] Pairwise Vector Metrics:")
    print("  Cosine Similarity Matrix:")
    print(f"    - Average intra-class: {np.mean([pairwise_cosine_matrix[i, i] for i in range(num_classes)]):.3f}")
    print(
        f"    - Average inter-class: {np.mean([pairwise_cosine_matrix[i, j] for i in range(num_classes) for j in range(num_classes) if i != j]):.3f}")


def main(skip_training):
    torch.manual_seed(40)
    np.random.seed(40)
    random.seed(40)
    global test_dataset

    device = "cpu"
    print(f"Using device: {device}")
    print("\n[INFO] Loading datasets...")
    train_files = load_dataset(dataset_path + Train_split)

    if not skip_training:
        test_files = load_dataset(test_dataset, is_test=True, max_test_samples=30)
        print(f"[INFO] Loaded {len(train_files)} training files and {len(test_files)} test files")
        test_dataset = HDF5Dataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    run_dir = create_run_directory()

    # Save Parameters Before Training
    params = {
        "DIMS": DIMS,
        "EVENT_THRESHOLD": EVENT_THRESHOLD,
        "WINDOW_SIZE": WINDOW_SIZE,
        "NGRAM_SIZE": NGRAM_SIZE,
        "OVERLAP": OVERLAP,
        "NUM_TRAIN_METHODS": NUM_TRAIN_METHODS,
        "method_encoding": method_encoding,
        "K EVENTHD": K,
        "DEFAULT_HEATMAP_SAMPLES": DEFAULT_HEATMAP_SAMPLES,
    }

    train_dataset = HDF5Dataset(train_files)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    print("\n[INFO] Initializing HISTEncoder...")
    encoder = HISTEncoder(
        height=120,
        width=160,
        dims=DIMS,
        device=device,
        window_size=WINDOW_SIZE,
        n_gram=NGRAM_SIZE,
        threshold=EVENT_THRESHOLD,
        method_encoding=method_encoding,
        K=K,
        debug=DEBUG_MODE
    )

    # Encode training dataset
    print("\n[INFO] Encoding training dataset...")
    train_vectors, train_labels = encode_dataset(train_loader, encoder, debug=DEBUG_MODE)

    # Analyze encoded vectors before training
    print("\n[INFO] Analyzing encoded training vectors...")

    # Calculate intra and inter-class similarities using a limited number of samples per class
    print(
        f"\n[INFO] Computing intra/inter-class similarities (limiting to {DEFAULT_HEATMAP_SAMPLES} samples per class)...")
    intra_sim, inter_sim = compute_intra_inter_class_similarities(train_vectors, train_labels, run_dir=run_dir, filename="vector_similarity_heatmap.png", sample_limit=DEFAULT_HEATMAP_SAMPLES, debug=True)

    # Calculate additional metrics
    print("\n[INFO] Computing additional similarity metrics...")
    compute_additional_metrics( train_vectors, train_labels, run_dir=run_dir, sample_limit=DEFAULT_HEATMAP_SAMPLES )
    print("\n[INFO] Creating t-SNE visualization of encoded vectors...")
    visualize_hypervectors(train_vectors, train_labels, run_dir, "vector_tsne_visualization.png")

    if skip_training:
        print("\n[INFO] Skipping training as requested.")
        return
    else:
        # Train models using different methods
        print("\n[INFO] Training models...")
        models = {}
        for method in NUM_TRAIN_METHODS:
            print(f"\n[TRAINING] Training model using {method} method...")
            models[method] = train_model(train_vectors, train_labels, method, debug=DEBUG_MODE)

        # Encode test dataset
        print("\n[INFO] Encoding test dataset...")
        test_vectors, test_labels = encode_dataset(test_loader, encoder, debug=DEBUG_MODE)

        # Evaluate models
        print("\n[INFO] Evaluating models...")
        results = {}
        for method, model in models.items():
            results[method] = evaluate_model(model, test_vectors, test_labels, method)

        # Compare model performances
        print("\n[INFO] Comparing model performances:")
        for method, result in results.items():
            print(f"  - {method}: {result['accuracy']:.2f}% accuracy")

        save_run_info(run_dir, params, intra_sim, results)
        print("\n[INFO] Pipeline completed successfully!")


if __name__ == "__main__":
    main(skip_training=False)
