import torch
import torchhd
import os
import pickle
from tqdm import tqdm
from grasphdencoding import Raw_events_HDEncoder
from  seperaterawencoderfromhist import Raw_events_HDEncoder_Enhanced
import random
import torchmetrics
import torchhd.utils
import torchhd.classifiers
import numpy as np
import tonic
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchhd.models import Centroid
import seaborn as sns
import gc
from collections import defaultdict

gc.collect()

#os.environ["OMP_NUM_THREADS"] = "8"
#torch.set_num_threads(8)
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)
torch.manual_seed(40)
np.random.seed(40)
random.seed(40)
import resource
#resource.setrlimit(resource.RLIMIT_AS, (10_000_000_000, 10_000_000_000))  # 2GB limit

'''
stem: Interpolation is done dimension-wise (not concatenation!) for spatial. for temp: STEMHD does use concatenation for  1D temporal interpolation.
"A proportion (1 - α) of Tj is taken from the first vector and α from the next one. The two parts are concatenated to form the new time hypervector."


stem_hd:    spatial:  dimensionwise interpolation. weighted combination of neighboring hypervectors based on distance. same as eventhd, grasphd.
            temporal: STEMHD does use concatenation for its 1D temporal interpolation.   ##########One per bin
                        A proportion (1-alpha) of Tk is taken from first vector, and alpha from tj+1 from next one.
                        concat together. 

event_hd:   spatial:    uses weighted sum per element to interpolate positions.
            temporal:   event_hd_timepermutation: Uses permutation-based encoding, time step is encoded using permutations of a base hypervector  for each t.
                        eventhd_timeinterpolation:Uses weighted sum per element for interpolation (NOT concatenation).  Multiple—one per intermediate timestamp

grasp_hd:   spatial:    weighted sum per dimension (similar to EventHD for time)
            temporal:   Uses weighted sum per element to blend temporal hypervectors (same as EventHD time interpolation). Multiple—one per intermediate timestamp
            same as eventhd timeineterpolation.

STEMHD = Uses concatenation for time, not for space. in time: One per bin
EventHD = Uses weighted sum for both space and time (no concatenation).
GraspHD = Uses weighted sum for both space and time (no concatenation).

'''
#["event_hd_timepermutation", "stem_hd" , "event_hd_timeinterpolation", kxk_ngram]:
TIME_INTERPOLATION_METHOD = "event_hd_timeinterpolation"

ENCODING_METHOD = Raw_events_HDEncoder
if TIME_INTERPOLATION_METHOD in ["thermometer","linear","kxk_ngram"]:
    ENCODING_METHOD = Raw_events_HDEncoder_Enhanced
def main():
    device = "cpu"
    print(f"Using device: {device}")
    height, width = 34, 34
    train_split = "Train"
    test_split = "Test"

    dataset_name = "nmnist"  # chifoumi
    dataset_path = "/space/chair-nas/tosy/data/"

    if dataset_name == "chifoumi":
        dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
        height = 480
        width = 640
        train_split = "Train"
        test_split = "Test"

    max_samples_train, max_samples_test = 100,20

    DIMS, K, Timewindow = 4000, 5 , 50_000

    WINDOW_SIZE_MS, OVERLAP_MS= 600000, 0
    debug = True
    save =True

    dataset_train = load__dataset(dataset_path, train_split, max_samples_train, dataset_name)
    dataset_test = load__dataset(dataset_path, test_split, max_samples_test, dataset_name)
    max_time = WINDOW_SIZE_MS
    print(f"[INFO] Sliding Window = {max_time}")
    encoder = ENCODING_METHOD(
        height=height, width=width, dims=DIMS, time_subwindow=Timewindow, k=K, device=device,
        max_time=max_time, time_method=TIME_INTERPOLATION_METHOD,
        WINDOW_SIZE_MS=WINDOW_SIZE_MS, OVERLAP_MS=OVERLAP_MS
    )
    # ------------------------ Encode Train & Test Data ------------------------
    encoded_train, labels_train = encode_dataset(dataset_train, encoder, split_name=train_split)
    encoded_test, labels_test = encode_dataset(dataset_test, encoder, split_name=test_split)
    # ------------------------ Train & Test ------------------------
    run_folder = create_unique_run_folder("/space/chair-nas/tosy/17marsraw/test_run/") if save else None

    models = {}
    accuracies = {}
    train_accuracies = {}

    for method in ["Vanilla", "AdaptHD", "OnlineHD", "Centroid"]:
        print(f"\n[TRAINING] Training model using {method}...")
        model, train_acc = train_model(encoded_train, labels_train, method, debug, DIMS)
        models[method] = model
        train_accuracies[method] = train_acc

        # **Test Model on Train Data**
        train_acc, train_preds = _test_model(model, encoded_train, labels_train)
        print(f"[INFO] {method} Train Accuracy: {train_acc:.3f}%")

        # **Test Model on Test Data**
        test_acc, test_preds = _test_model(model, encoded_test, labels_test)
        accuracies[method] = test_acc
        print(f"[INFO] {method} Test Accuracy: {test_acc:.3f}%")

        print(f"[DEBUG] Model successfully trained: {model}")
        plot_confusion_matrix(labels_train, train_preds, save, run_folder, f"Train_{method}")
        plot_confusion_matrix(labels_test, test_preds, save, run_folder, f"Test_{method}")

    plot_heatmap(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
                 run_folder, "Train")
    plot_heatmap(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
                 run_folder, train_split)



    plot_tsne(encoded_train, labels_train, K, Timewindow, DIMS, TIME_INTERPOLATION_METHOD, save,run_folder, train_split)
    plot_tsne(encoded_test, labels_test, K, Timewindow, DIMS, TIME_INTERPOLATION_METHOD, save, run_folder, test_split)
    if save:
        save_hyperparameters(run_folder, {
            "Methods": list(accuracies.keys()),
            "Train_Accuracies": train_accuracies,
            "Test_Accuracies": accuracies,
            "k": K, "Timewindow": Timewindow, "DIMS": DIMS,
            "Train_samples": len(dataset_train), "Test_samples": len(dataset_test),
            "Method": TIME_INTERPOLATION_METHOD,

        })




















def load__dataset(dataset_path, split, max_samples, dataset_name):
    """
    Loads dataset based on dataset_name.
    - Chifoumi (pickle files) -> loads & shifts timestamps.
    - NMNIST (Tonic) -> loads exactly `max_samples` per digit (0-9) using Tonic's API.
    """
    if dataset_name == "chifoumi":
        dataset = load_pickle_dataset(dataset_path, split, max_samples)
        return dataset

    elif dataset_name == "nmnist":
        dataset = tonic.datasets.NMNIST(save_to=dataset_path, train=(split == "train"))
        dataset_list = []
        dtype = np.dtype([("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", np.int8)])

        # Store counts per digit class
        class_counts = {i: 0 for i in range(10)}

        for i in range(len(dataset)):
            events, class_id = dataset[i]  # Get (events, label) directly from Tonic

            # Ensure we don't exceed max_samples per class
            if class_counts[class_id] < max_samples:
                structured_events = np.array([(t, x, y, p) for x, y, t, p in events], dtype=dtype)
                dataset_list.append((structured_events, class_id))
                class_counts[class_id] += 1

            # Stop early if we have enough samples for all digits
            if all(count >= max_samples for count in class_counts.values()):
                break

        print(f"Loaded {len(dataset_list)} NMNIST samples for {split} split.")
        return dataset_list

    else:
        raise ValueError("Unknown dataset! Choose 'chifoumi' or 'nmnist'.")

def encode_dataset(dataset, encoder, split_name):
    encoded_vectors, class_labels = [], []
    for events, class_id in tqdm(dataset, desc=f"Encoding {split_name} Samples"):
        encoded_windows = encoder.process_windows(events, class_id)     ####should work, if not suse if else.
        encoded_vectors.extend(encoded_windows)
        class_labels.extend([class_id] * len(encoded_windows))
    return torch.stack(encoded_vectors), class_labels


def train_model(encoded_vectors, class_labels, method, debug, d):
    """Trains a model using TorchHD classifiers."""
    unique_classes = sorted(list(set(class_labels)))
    num_classes = len(unique_classes)

    if debug:
        print(f"\n[TRAINING] Using method: {method}")
        print(f"[TRAINING] Number of classes: {num_classes}")
        print(f"[TRAINING] Unique classes: {unique_classes}")
        print(f"[TRAINING] Total vectors: {len(encoded_vectors)}")
        class_counts = defaultdict(int)
        for cls in class_labels:
            class_counts[cls] += 1
        print(f"[TRAINING] Vectors per class: {dict(class_counts)}")

    # **Handle Centroid & Vanilla (both are centroid-based)**
    if method in ["Centroid", "Vanilla"]:
        model = torchhd.models.Centroid(in_features=d, out_features=num_classes)
        with torch.no_grad():
            for class_id in unique_classes:
                class_mask = torch.tensor(class_labels) == class_id
                class_vectors = encoded_vectors[class_mask]
                model.weight[class_id] = class_vectors.mean(dim=0)
            model.normalize()  # Normalize for cosine similarity

    # **Handle AdaptHD & OnlineHD**
    elif method in ["AdaptHD", "OnlineHD"]:
        model_cls = getattr(torchhd.classifiers, method)
        model = model_cls(
            n_features=d,
            n_dimensions=d,
            n_classes=num_classes,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        labels_tensor = torch.tensor(class_labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(encoded_vectors, labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        model.fit(loader)  # **Remove `epochs=10` (not needed)**

        # **Normalize Centroids if using a centroid-based classifier**
        if hasattr(model, "model") and isinstance(model.model, torchhd.models.Centroid):
            with torch.no_grad():
                model.model.weight.data = torch.nn.functional.normalize(model.model.weight.data, dim=1)

    else:
        raise ValueError(f"Unknown training method: {method}")

    # **Compute Training Accuracy**
    train_accuracy, _ = _test_model(model, encoded_vectors, class_labels)
    print(f"[TRAINING] {method} Training Accuracy: {train_accuracy:.3f}%")

    return model, train_accuracy


def _test_model(model, encoded_test, test_labels):
    accuracy_metric = torchmetrics.Accuracy("multiclass", num_classes=len(set(test_labels)))
    with torch.no_grad():
        output = model(encoded_test)
        preds = torch.argmax(output, dim=1)
        accuracy_metric.update(output, torch.tensor(test_labels, dtype=torch.long))
    acc = accuracy_metric.compute().item() * 100
    return acc, preds.tolist()


def plot_heatmap(vectors, labels, k, Timewindow, dims, max_samples, encodingmethod, save, run_folder, split_name):
    """Plots and saves a cosine similarity heatmap ensuring equal selection from all available classes."""

    class_labels_tensor = torch.tensor(labels)
    unique_classes = torch.unique(class_labels_tensor).tolist()
    num_classes = len(unique_classes)

    # **Compute how many samples per class (rounded)**
    samples_per_class = max(1, max_samples // num_classes)  # Ensure at least 1 sample per class
    class_to_samples = defaultdict(list)

    # **Group sample indices by class**
    for idx, label in enumerate(class_labels_tensor.tolist()):
        class_to_samples[label].append(idx)

    selected_indices = []

    # **Step 1: Select equal samples per class**
    for cls in unique_classes:
        available_samples = class_to_samples[cls]
        selected_indices.extend(available_samples[:samples_per_class])  # Take `samples_per_class` from each class

    # **Step 2: If not enough samples were selected, fill the remaining slots**
    while len(selected_indices) < max_samples:
        for cls in unique_classes:
            if len(selected_indices) < max_samples and class_to_samples[cls]:
                selected_indices.append(class_to_samples[cls].pop(0))  # Add extra sample if available

    # **Final check: Trim excess samples**
    selected_indices = selected_indices[:max_samples]

    # **Sort for better visualization**
    selected_indices = torch.tensor(sorted(selected_indices))
    selected_vectors = vectors[selected_indices]
    selected_labels = class_labels_tensor[selected_indices].tolist()

    # Compute cosine similarity for selected samples
    similarity_matrix = torchhd.functional.cosine_similarity(selected_vectors, selected_vectors).cpu().numpy()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, annot=False, cmap="coolwarm",
        xticklabels=selected_labels, yticklabels=selected_labels,
        cbar=True, square=True, linewidths=0.5
    )

    plt.title(
        f"Cosine Similarity Heatmap ({split_name}) | {encodingmethod} (k={k}, dims={dims}, timewindow={Timewindow})")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")

    print(f"[INFO] Selected {len(selected_indices)} samples with {samples_per_class} per class (Adjusted as needed).")

    # **Show plot non-blocking**
    plt.draw()
    plt.pause(0.001)

    # Save plot
    if save and run_folder:
        save_plot(run_folder, f"{split_name}_balanced_heatmap.png")

    plt.close()


def check_centroid_similarity(model, encoded_train, labels_train):
    """Checks how similar each centroid is to its respective class samples."""

    if hasattr(model, "model") and isinstance(model.model, Centroid):
        centroids = model.model.weight.detach().cpu()  # Extract centroids correctly
    else:
        print("[WARNING] Model does not contain centroids. Skipping similarity check.")
        return

    encoded_train = encoded_train.cpu()  # Move training samples to CPU
    labels_train = torch.tensor(labels_train, dtype=torch.long, device="cpu")  # Convert labels to tensor

    num_classes = centroids.shape[0]  # Number of classes
    similarities = []

    for class_id in range(num_classes):
        # Select only the training samples belonging to the current class
        class_mask = labels_train == class_id
        class_samples = encoded_train[class_mask]  # Filter samples by class

        if class_samples.shape[0] == 0:
            print(f"[WARN] No samples found for class {class_id}")
            similarities.append(0)
            continue
        centroid = centroids[class_id].unsqueeze(0)  # Ensure centroid has shape (1, D)
        sim = torchhd.functional.cosine_similarity(class_samples, centroid).mean()
        similarities.append(sim.item())

    print("\n **Average cosine similarities of centroids with their class samples:**")
    for i, sim in enumerate(similarities):
        print(f"Class {i}: {sim:.4f}")

    return similarities

def plot_tsne(encoded_vectors, class_labels, k, Timewindow, dims,  encodingmethod, save, run_folder, split_name):
    """Plots and saves t-SNE visualization for all samples."""
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    encoded_vectors = np.array(encoded_vectors)  # Convert list to NumPy array
    reduced_vectors = tsne.fit_transform(encoded_vectors)

    # Plot t-SNE scatter
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(class_labels)
    colors = ["red", "blue", "green", "purple", "orange"]
    palette = {cls: colors[i % len(colors)] for i, cls in enumerate(unique_classes)}

    for class_id in unique_classes:
        indices = np.where(np.array(class_labels) == class_id)[0]
        plt.scatter(
            reduced_vectors[indices, 0], reduced_vectors[indices, 1],
            label=f"Class {class_id}", color=palette[class_id], alpha=0.7, edgecolors='k'
        )

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE Visualization ({split_name}) | {encodingmethod} (k={k}, dims={dims}, timewindow={Timewindow})")
    plt.legend()

    # Save plot
    if save and run_folder:
        save_plot(run_folder, f"{split_name}_tsne.png")
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, save, run_folder, split_name):
    """Generates a confusion matrix to visualize classification performance."""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(len(set(true_labels))), yticklabels=range(len(set(true_labels))))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({split_name})")

    # Save plot
    if save and run_folder:
        save_plot(run_folder, f"{split_name}_confusion_matrix.png")
    plt.close()


def save_plot(run_folder, filename):
    """Saves the current plot to the specified folder."""
    plt.savefig(os.path.join(run_folder, filename), bbox_inches="tight")
    print(f"[SAVED] Plot saved: {filename}")
def save_encoded_data(run_folder, encoded_data, labels, split_name):
    """Saves encoded data and labels."""
    save_pickle_file(run_folder, f"{split_name}_encoded.pkl", {"encoded_matrix": encoded_data.cpu(), "class_labels": labels})
def create_unique_run_folder(base_path):
    """Creates a unique folder for this run."""
    os.makedirs(base_path, exist_ok=True)
    existing_runs = sorted([d for d in os.listdir(base_path) if d.startswith("run_")])
    new_run = max([int(d.split("_")[1]) for d in existing_runs], default=0) + 1
    run_folder = os.path.join(base_path, f"run_{new_run:03d}")
    os.makedirs(run_folder)
    print(f"[INFO] Run folder created at: {run_folder}")
    return run_folder

def save_hyperparameters(run_folder, params):
    """Saves hyperparameters."""
    param_file = os.path.join(run_folder, "params.txt")
    with open(param_file, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"[SAVED] Hyperparameters saved to: {param_file}")

def save_pickle_file(run_folder, filename, data):
    """Saves dictionary to a pickle file"""
    with open(os.path.join(run_folder, filename), "wb") as f:
        pickle.dump(data, f)
        print(f" [SAVED] {filename} file saved")

def save_plot(run_folder, filename):
    plt.gcf().canvas.draw()
    plt.savefig(os.path.join(run_folder, filename), bbox_inches="tight")
    print(f" [SAVED] Plot saved: {filename}")
    plt.show()
    plt.close()
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

