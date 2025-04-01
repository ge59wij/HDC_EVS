import torch
import torchhd
import os
import pickle
from tqdm import tqdm
import random
import torchmetrics
import torchhd.utils
import torchhd.classifiers
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchhd.models import Centroid
import seaborn as sns
import gc
from collections import defaultdict


def encode_dataset(dataset, encoder, split_name, device="cuda"):
    encoded_vectors, class_labels = [], []

    for events, class_id in tqdm(dataset, desc=f"Encoding {split_name} Samples"):
        encoded_windows = encoder.process_windows(events, class_id)
        encoded_vectors.extend(encoded_windows)
        class_labels.extend([class_id] * len(encoded_windows))

    encoded_vectors = torch.stack(encoded_vectors).to(device)
    class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)

    return encoded_vectors, class_labels

def train_model(encoded_vectors, class_labels, method, debug, d):
    device = encoded_vectors.device
    unique_classes = sorted(list(set(class_labels.cpu().tolist())))
    num_classes = len(unique_classes)

    if debug:
        print(f"\n[TRAINING] Using method: {method} on {device}")
        print(f"[TRAINING] Unique classes: {unique_classes}")
        print(f"[TRAINING] Total vectors: {len(encoded_vectors)}")

    if method in ["Centroid", "Vanilla"]:
        model = torchhd.models.Centroid(in_features=d, out_features=num_classes).to(device)

        with torch.no_grad():
            for class_id in unique_classes:
                class_mask = class_labels == class_id
                class_vectors = encoded_vectors[class_mask]
                model.weight[class_id] = class_vectors.mean(dim=0)
            model.normalize()
    elif method in ["AdaptHD", "OnlineHD"]:
        model_cls = getattr(torchhd.classifiers, method)
        model = model_cls(n_features=d, n_dimensions=d, n_classes=num_classes, epochs=5, device=device)
        dataset = torch.utils.data.TensorDataset(encoded_vectors, class_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False)
        model.fit(loader)
        if hasattr(model, "model") and isinstance(model.model, torchhd.models.Centroid):
            with torch.no_grad():
                model.model.weight.data = torch.nn.functional.normalize(model.model.weight.data, dim=1)
    else:
        raise ValueError(f"Unknown training method: {method}")
    train_accuracy, _ = _test_model(model, encoded_vectors, class_labels)
    print(f"[TRAINING] {method} Training Accuracy: {train_accuracy:.3f}%")
    del encoded_vectors
    gc.collect()

    return model, train_accuracy


def _test_model(model, encoded_test, test_labels):
    device = encoded_test.device
    model.to(device)
    test_labels = test_labels.to(device)

    accuracy_metric = torchmetrics.Accuracy("multiclass", num_classes=len(set(test_labels.cpu().tolist()))).to(device)

    with torch.no_grad():
        output = model(encoded_test)
        preds = torch.argmax(output, dim=1) if output.dim() == 2 else output

        accuracy_metric.update(preds, test_labels)

    acc = accuracy_metric.compute().item() * 100
    return acc, preds.tolist()



def plot_confusion_matrix(true_labels, pred_labels, save, run_folder, split_name):
    """Generates a confusion matrix and saves it automatically."""

    true_labels = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else np.array(true_labels)
    pred_labels = pred_labels.cpu().numpy() if isinstance(pred_labels, torch.Tensor) else np.array(pred_labels)

    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(len(set(true_labels))),
                yticklabels=range(len(set(true_labels))))

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({split_name})")

    save_plot(run_folder, f"{split_name}_confusion_matrix.png")


def plot_heatmap(vectors, labels, k, Timewindow, dims, max_samples, encodingmethod, save, run_folder, split_name):
    """Plots and saves a cosine similarity heatmap."""

    class_labels_tensor = torch.tensor(labels)
    unique_classes = torch.unique(class_labels_tensor).tolist()
    num_classes = len(unique_classes)

    samples_per_class = max(1, max_samples // num_classes)
    class_to_samples = defaultdict(list)

    for idx, label in enumerate(class_labels_tensor.tolist()):
        class_to_samples[label].append(idx)

    selected_indices = []

    for cls in unique_classes:
        available_samples = class_to_samples[cls]
        selected_indices.extend(available_samples[:samples_per_class])

    selected_indices = selected_indices[:max_samples]

    selected_vectors = vectors[selected_indices]
    selected_labels = class_labels_tensor[selected_indices].tolist()

    similarity_matrix = torchhd.functional.cosine_similarity(selected_vectors, selected_vectors).cpu().numpy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, annot=True, cmap="coolwarm",
        xticklabels=selected_labels, yticklabels=selected_labels,
        cbar=True, square=True, linewidths=0.5
    )

    plt.title(
        f"Cosine Similarity Heatmap ({split_name}) | {encodingmethod} (k={k}, dims={dims}, timewindow={Timewindow})")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")

    save_plot(run_folder, f"{split_name}_{encodingmethod}_heatmap.png")



def save_plot(run_folder, filename):
    """Saves the current plot to the specified folder without showing it."""
    os.makedirs(run_folder, exist_ok=True)
    plt.savefig(os.path.join(run_folder, filename), bbox_inches="tight")
    print(f"[SAVED] Plot saved: {filename}")
    plt.close()


def save_hyperparameters(run_folder, params):
    """Saves hyperparameters to a text file."""
    os.makedirs(run_folder, exist_ok=True)
    param_file = os.path.join(run_folder, "params.txt")
    with open(param_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"[SAVED] Hyperparameters saved to: {param_file}")


def create_experiment_root(base_path):
    """Creates a top-level experiment folder that contains all runs."""
    os.makedirs(base_path, exist_ok=True)
    existing_runs = sorted([d for d in os.listdir(base_path) if d.startswith("run_") and d[4:].isdigit()])

    new_run = 1
    if existing_runs:
        new_run = max(int(d[4:]) for d in existing_runs) + 1

    experiment_folder = os.path.join(base_path, f"run_{new_run:03d}")
    os.makedirs(experiment_folder)
    print(f"[INFO] Experiment root created at: {experiment_folder}")
    return experiment_folder
def create_unique_run_folder(experiment_root, encoding_method, param_name, param_value,dataset):
    """Creates a unique run folder inside the current experiment root."""
    run_folder = os.path.join(experiment_root, encoding_method, f"{param_name}_{param_value}_{dataset}")
    os.makedirs(run_folder, exist_ok=True)
    print(f" [INFO] Run folder created at: {run_folder}")
    return run_folder


def save_pickle_file(run_folder, filename, data):
    """Saves dictionary to a pickle file."""
    os.makedirs(run_folder, exist_ok=True)
    with open(os.path.join(run_folder, filename), "wb") as f:
        pickle.dump(data, f)
    print(f" [SAVED] {filename} file saved")



def clear_cache():
    """Clears memory cache for fresh runs."""
    torch.cuda.empty_cache()
    gc.collect()
