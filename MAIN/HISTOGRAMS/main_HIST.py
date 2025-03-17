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
import resource
import psutil
import time
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.main_enc import (
    create_unique_run_folder, _test_model, train_model, plot_heatmap,
    plot_tsne, plot_confusion_matrix, save_hyperparameters, encode_dataset)
total_memory = psutil.virtual_memory().total
safe_limit = int(total_memory * 0.9) # of total RAM
resource.setrlimit(resource.RLIMIT_AS, (safe_limit, safe_limit))
print(f"Memory limit set to: {safe_limit / 1e9:.1f} GB")

TRAIN_METHODS = ["VanillaHD", "AdaptHD", "OnlineHD", "Centroid"]
BACKGROUND_LABEL = 404
train_dataset = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/"
#dataset_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/Bin Labeled/"
Train_split = "train"
#Train_split = "VAL BIN LABELED"
test_dataset = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/test/"
DEBUG_MODE = False
DEFAULT_HEATMAP_SAMPLES = 25  # Maximum samples per class for similarity calculations
EVENT_THRESHOLD = 6 / 16  # if below this, then counted as no gesture #for bin labeling in the test data and encoding


K = 20
DIMS = 6000  ##at least 500 for permuation/thermometer
WINDOW_SIZE = 40
NGRAM_SIZE = 9
OVERLAP = 10
method_encoding = "linear"  # "thermometer" or "linear" or kxk_ngram or "eventhd_timeinterpolation" "eventhd_timepermutation" "stem_hd"
train_samples, test_samples = 100, 20


def main(skip_training):
    torch.manual_seed(40)
    np.random.seed(40)
    random.seed(40)
    global test_dataset
    global train_dataset
    debug = True
    save =True
    device = "cpu"
    print(f"Using device: {device}")
    print("\n[INFO] Loading datasets...")

    train_files = load_dataset(train_dataset + Train_split, is_test=True, max_samples = train_samples)
    train_dataset = HDF5Dataset(train_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_files = load_dataset(test_dataset, is_test=True, max_samples=test_samples)
    test_dataset = HDF5Dataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    print(f"[INFO] Loaded {len(train_dataset)} training files and {len(test_files)} test files.")

    debug2 = False
    if debug2:
        for idx in range(len(train_dataset)):
            event_data, class_ids, file_path = train_dataset[idx]  # Unpack dataset item
            print(f"[INFO] Train Sample {idx}: Class ID = {class_ids.numpy()} from {file_path}")
        for idx in range(len(test_dataset)):
            event_data, class_ids, file_path = test_dataset[idx]  # Unpack dataset item
            print(f"[INFO] Test Sample {idx}: Class ID = {class_ids.numpy()} from {file_path}")
    print("\n[INFO] Initializing HISTEncoder...")
    encoder = HISTEncoder(height=120,width=160,dims=DIMS,device=device,
                          window_size=WINDOW_SIZE,n_gram=NGRAM_SIZE,
                          threshold=EVENT_THRESHOLD,method_encoding=method_encoding
                          ,K=K,debug=DEBUG_MODE, weighting= weighting)


    encoded_train, labels_train = encode_dataset(train_loader, encoder, "TRAIN")
    encoded_test, labels_test == encode_dataset(test_loader, encoder,"TEST")

    run_dir = create_unique_run_folder("/space/chair-nas/tosy/17marsHIST/test_run/")


    models = {}
    accuracies = {}
    train_accuracies = {}
    for method in TRAIN_METHODS:
        print(f"\n[TRAINING] Training model using {method}...")
        model, train_acc = train_model(encoded_train, labels_train, method, debug=True, d=DIMS)
        models[method] = model
        train_accuracies[method] = train_acc

        train_acc, train_preds = _test_model(model, encoded_train, labels_train)
        print(f"[INFO] {method} Train Accuracy: {train_acc:.3f}%")

        test_acc, test_preds = _test_model(model, encoded_test, labels_test)
        accuracies[method] = test_acc
        print(f"[INFO] {method} Test Accuracy: {test_acc:.3f}%")

        plot_confusion_matrix(labels_train, train_preds, save=True, run_folder=run_dir, split_name=f"Train_{method}")
        plot_confusion_matrix(labels_test, test_preds, save=True, run_folder=run_dir, split_name=f"Test_{method}")

    #print("\n[INFO] Computing intra/inter-class similarities (Heatmap)...")
    #intra_sim, inter_sim = compute_intra_inter_class_similarities(train_vectors, train_labels, run_dir=run_dir, filename="vector_similarity_heatmap.png",sample_limit=20, debug=True)
    #compute_additional_metrics(train_vectors, train_labels, run_dir, sample_limit=DEFAULT_HEATMAP_SAMPLES)
    plot_tsne(encoded_train, labels_train, K, WINDOW_SIZE, DIMS, method_encoding, True, run_dir, "train")
    plot_tsne(encoded_test, labels_test, K, WINDOW_SIZE, DIMS, method_encoding, True, run_dir, "test")


     save_hyperparameters(run_dir, {
            "Methods": TRAIN_METHODS,
            "Train_Accuracies": train_accuracies,
            "Test_Accuracies": accuracies,
            "DIMS": DIMS,
            "k": K,
            "Window Size": WINDOW_SIZE,
            "Overlap": OVERLAP,
            "NGRAM"= NGRAM_SIZE,
            "Event count bin Threshhold" = EVENT_THRESHOLD,
            "Train Samples": len(train_dataset_obj),
            "Test Samples": len(test_dataset_obj),
            "Encoding Method": method_encoding,
            "Event Count weighting": weighting,

        })













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
def bin_labeling(files):
    """Processes the test dataset and assigns a class label to each bin based on event count."""
    print("\n[INFO] Processing test dataset for per-bin labeling...")
    labeled_files = []

    for file in tqdm(files, desc="Labeling Bins"):
        with h5py.File(file, "r+") as f:
            event_data = f["data"][:]  # Shape: (T, 2, H, W)  -> (time bins, polarity, height, width)
            class_id = f["class_id"][()]  # Integer class label for full sample

            # Compute total events per bin
            total_events_per_bin = np.sum(event_data, axis=(1, 2, 3))  # Shape: (T,)

            # Assign class_id to each bin (assuming all bins get the same class for now)
            #bin_labels = np.full_like(total_events_per_bin, class_id, dtype=np.int32)  # Shape: (T,)

            # If a threshold-based background label is needed:
            bin_labels = np.where(total_events_per_bin < EVENT_THRESHOLD, BACKGROUND_LABEL, class_id)

            # Overwrite or create "labels" dataset in HDF5 file
            if "labels" in f:
                del f["labels"]  # Remove existing labels if present
            f.create_dataset("labels", data=bin_labels, dtype="int32")

            labeled_files.append(file)

    print("[INFO] Test dataset labeling complete!")
    return labeled_files
def load_dataset(dataset_folder, is_test, max_samples):
    """Loads dataset and applies bin labeling if test, with random sampling."""
    files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.h5')]
    # Shuffle files to ensure random selection
    random.shuffle(files)
    #if is_test:
    files = bin_labeling(files[:max_samples])
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
def encode_dataset(dataloader, encoder, split_name, debug):
    encoded_vectors, class_labels = [], []
    for batch_idx, (data, labels, file_path) in enumerate(tqdm(dataloader, desc=f"Encoding {split_name} Dataset")):
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
            print(f"[DEBUG] Unique encoded classes: {set(class_labels)}")

    if not encoded_vectors:
        raise ValueError("No valid hypervectors were generated from the dataset")
    return torch.stack(encoded_vectors), class_labels
def compute_intra_inter_class_similarities(vectors, labels, run_dir, filename="similarity_heatmap.png",
                                           sample_limit=DEFAULT_HEATMAP_SAMPLES, debug=False):
    unique_classes = sorted(list(set(labels)))
    class_vectors = {}
    for cls in unique_classes:
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        if len(cls_indices) > sample_limit:
            cls_indices = cls_indices[:sample_limit]
        class_vectors[cls] = vectors[cls_indices]

    intra_sim = {}
    inter_sim = {}

    if debug:
        print(f"\n[SIMILARITY] Computing with up to {sample_limit} samples per class.")

    # Compute intra-class similarities
    for cls, vecs in class_vectors.items():
        if len(vecs) > 1:
            sim_matrix = torchhd.cosine_similarity(vecs, vecs)
            mask = ~torch.eye(len(vecs), dtype=torch.bool)
            intra_sim[cls] = sim_matrix[mask].mean().item()
            if debug:
                print(f"  - Class {cls}: Intra-Class Similarity: {intra_sim[cls]:.3f}")

    # Compute inter-class similarities
    for cls in unique_classes:
        inter_sim[cls] = {}
        for other_cls in unique_classes:
            if cls != other_cls:
                inter = torchhd.cosine_similarity(class_vectors[cls], class_vectors[other_cls]).mean().item()
                inter_sim[cls][other_cls] = inter
                if debug:
                    print(f"  - {cls} vs. {other_cls}: Inter-Class Similarity: {inter:.3f}")

    # Save sorted heatmap
    plot_path = os.path.join(run_dir, filename)
    create_similarity_heatmap(intra_sim, inter_sim, plot_path)

    return intra_sim, inter_sim



if __name__ == "__main__":
    main()
