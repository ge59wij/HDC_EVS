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
NUM_TRAIN_METHODS = ["Vanilla", "AdaptHD", "OnlineHD"]

# -------------------------------- Hyperparameters --------------------------------

#["event_hd_timepermutation", "stem_hd" , "event_hd_timeinterpolation"]:
TIME_INTERPOLATION_METHOD = "kxk_ngram"


TRAINING_METHOD = "AdaptHD"  # "centroid" "adaptive"
LEARNING_RATE = 0.5
ENCODING_METHOD = Raw_events_HDEncoder
if TIME_INTERPOLATION_METHOD in ["thermometer","linear","kxk_ngram"]:
    ENCODING_METHOD = Raw_events_HDEncoder_Enhanced
def main():
    device = "cpu"
    print(f"Using device: {device}")
    height, width = 34, 34
    train_split = "Train"
    test_split = "Test"

    # ------------------------ Parameters ------------------------
    dataset_name = "nmnist"  # chifoumi
    dataset_path = "/space/chair-nas/tosy/data/"

    if dataset_name == "chifoumi":
        dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
        height = 480
        width = 640
        train_split = "Train"
        test_split = "Test"

    max_samples_train, max_samples_test = 10,10

    DIMS, K, Timewindow = 4000, 5 , 50_000

    WINDOW_SIZE_MS, OVERLAP_MS= 600000, 0
    debug = True
    save =True

    # ------------------------ Load & Preprocess Dataset ------------------------
    dataset_train = load__dataset(dataset_path, train_split, max_samples_train, dataset_name)
    dataset_test = load__dataset(dataset_path, test_split, max_samples_test, dataset_name)
    max_time = WINDOW_SIZE_MS
    print(f"[INFO] Using max_time = {max_time}")

    encoder = ENCODING_METHOD(
        height=height, width=width, dims=DIMS, time_subwindow=Timewindow, k=K, device=device,
        max_time=max_time, time_method=TIME_INTERPOLATION_METHOD,
        WINDOW_SIZE_MS=WINDOW_SIZE_MS, OVERLAP_MS=OVERLAP_MS
    )

    # ------------------------ Encode Train & Test Data ------------------------
    encoded_train, labels_train = encode_dataset(dataset_train, encoder, split_name=train_split)
    encoded_test, labels_test = encode_dataset(dataset_test, encoder, split_name=test_split)
    # ------------------------ Train & Test ------------------------

    print(f"\n[TRAINING] Training model using {TRAINING_METHOD}...")
    print(
        f"[DEBUG] Encoded train shape: {encoded_train.shape if isinstance(encoded_train, torch.Tensor) else type(encoded_train)}")
    models = {}
    model = train_model(encoded_train, labels_train, TRAINING_METHOD,debug, DIMS, LEARNING_RATE)
    models[TRAINING_METHOD] = model
    print(f"[DEBUG] Model successfully trained: {model}")  # Debug check
    accuracy, preds = _test_model(model, encoded_test, labels_test)  # Now returns preds too
    print(f"Testing Accuracy: {accuracy:.3f}%")
    run_folder = create_unique_run_folder("/space/chair-nas/tosy/17marsraw/test_run/") if save else None

    plot_confusion_matrix(labels_test, preds, save, run_folder, test_split)
    plot_heatmap(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
                 run_folder, train_split)
    plot_heatmap(encoded_test, labels_test, K, Timewindow, DIMS, max_samples_test, TIME_INTERPOLATION_METHOD, save,
                 run_folder, test_split)
    plot_tsne(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
              run_folder, train_split)
    plot_tsne(encoded_test, labels_test, K, Timewindow, DIMS, max_samples_test, TIME_INTERPOLATION_METHOD, save,
                  run_folder, test_split)
    plot_tsne_with_centroids(model, encoded_train, labels_train, save, run_folder, train_split)
    plot_tsne_with_centroids(model, encoded_test, labels_test, save, run_folder, test_split)
    run_folder = create_unique_run_folder("/space/chair-nas/tosy/3.mars_after_fixes/test_run/") if save else None
    if save:
        save_encoded_data(run_folder, encoded_train, labels_train, train_split)
        save_encoded_data(run_folder, encoded_test, labels_test, test_split)
        save_hyperparameters(run_folder, {
            "k": K, "Timewindow": Timewindow, "DIMS": DIMS, "Train_samples": len(dataset_train),
            "Test_samples": len(dataset_test), "Method": TIME_INTERPOLATION_METHOD, "accuracy": accuracy, })

    if hasattr(model, "model") and isinstance(model.model, Centroid):
        print("Centroid norms:", model.model.weight.norm(dim=1))
        check_centroid_similarity(model, encoded_train, labels_train)
    else:
        print("[WARNING] Skipping centroid similarity check as model has no centroids.")


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


def train_model(encoded_vectors, class_labels, method, debug, d, lr):
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

    if method in torchhd.classifiers.__dict__:
        model_cls = getattr(torchhd.classifiers, method)
        # For these classifiers, pass the required arguments
        model = model_cls(
            n_features=d,
            n_dimensions=d,
            n_classes=num_classes,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    else:
        raise ValueError(f"Unknown training method: {method}")

    labels_tensor = torch.tensor(class_labels, dtype=torch.long)
    # Create a DataLoader from encoded vectors and labels
    dataset = torch.utils.data.TensorDataset(encoded_vectors, labels_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Use the classifier's built-in fit() method
    model.fit(loader)

    if debug:
        with torch.no_grad():
            sim = model(encoded_vectors)
            preds = sim.argmax(1)
            correct = (preds == labels_tensor).sum().item()
            acc = correct / len(labels_tensor) * 100
            print(f"[TRAINING] Training accuracy: {acc:.2f}%")

    return model


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


def plot_tsne_with_centroids(model, vectors, labels, save, run_folder, split_name):
    """Plots t-SNE with all samples & centroids to visualize separation."""

    if hasattr(model, "model") and isinstance(model.model, Centroid):
        centroids = model.model.weight.detach().cpu().numpy()
    else:
        print("[WARNING] Model does not contain centroids. Skipping t-SNE with centroids.")
        return

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)

    encoded_vectors = np.array(vectors)

    # **Step 1: Combine samples & centroids before t-SNE transformation**
    all_vectors = np.vstack([encoded_vectors, centroids])

    # **Step 2: Apply t-SNE to all data at once**
    reduced_all = tsne.fit_transform(all_vectors)

    # **Step 3: Split transformed data back into samples & centroids**
    reduced_vectors = reduced_all[:-centroids.shape[0]]
    reduced_centroids = reduced_all[-centroids.shape[0]:]

    # **Plot t-SNE scatter**
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(labels)
    colors = sns.color_palette("tab10", len(unique_classes))

    for idx, class_id in enumerate(unique_classes):
        indices = np.where(np.array(labels) == class_id)[0]
        plt.scatter(
            reduced_vectors[indices, 0], reduced_vectors[indices, 1],
            label=f"Class {class_id}", color=colors[idx], alpha=0.6, edgecolors='k'
        )

    # **Overlay Centroids**
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1],
                marker='X', s=300, c='black', linewidths=3, label="Centroids")

    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE with Centroids ({split_name})")
    plt.legend()

    if save and run_folder:
        save_plot(run_folder, f"{split_name}_tsne_centroids.png")

    plt.close()


def plot_tsne(encoded_vectors, class_labels, k, Timewindow, dims, max_samples, encodingmethod, save, run_folder, split_name):
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
def get_max_time(dataset):
    """
    Extracts the max timestamp from the dataset
    """
    max_time = 0
    for events, _ in dataset:
        if len(events) > 0:
            last_timestamp = events[-1][0]  # First element of the last tuple
            max_time = max(max_time, last_timestamp)
    return max_time


if __name__ == "__main__":
    main()




'''
    #max_time_train = get_max_time(dataset_train)   #we dont need this for sliding window approach
    #max_time_test = get_max_time(dataset_test)
    #max_time = max(max_time_train, max_time_test)
    #print(f"[INFO] Computed max_time: {max_time} (Train: {max_time_train}, Test: {max_time_test})")

    print("\n[DEBUG] First 5 events of first 3 training samples:")
    for i, (events, class_id) in enumerate(dataset_train[:3]):
        print(f"Sample {i} (Class {class_id}): {events[:3]}")
        target_timestamp = 9000
        tolerance = 2  # small window around the target timestamp
        for i, (events, class_id) in enumerate(dataset_train[:1]):  # Only first 2 samples
            filtered_events = [event for event in events if abs(event[0] - target_timestamp) <= tolerance]
            print(f"Sample {i} (Class {class_id}), Events around {target_timestamp}: {filtered_events}")



    print_debug(TIME_INTERPOLATION_METHOD, dataset_train, encoder, max_time, Timewindow, K)
    run_folder = create_unique_run_folder("/space/chair-nas/tosy/3.mars after fixes/test_run/") if save else None

    if save:
        save_encoded_data(run_folder, encoded_train, labels_train, "train")
        save_encoded_data(run_folder, encoded_test, labels_test, "test")
        save_hyperparameters(run_folder,
                             {"k": K, "Timewindow": Timewindow, "DIMS": DIMS, "Train_samples": len(dataset_train),
                              "Test_samples": len(dataset_test), "Method": TIME_INTERPOLATION_METHOD})


        #elif TIME_INTERPOLATION_METHOD == "encode_temporalpermutation":
        #    encoded_sample = encoder.encode_temporalpermutation(events, class_id)
        #elif TIME_INTERPOLATION_METHOD == "encode_accumulation_weight":
        #    encoded_sample = encoder.encode_accumulation_weight(events, class_id)
        #elif TIME_INTERPOLATION_METHOD in [ "thermometer", "permutation"]:
        #    encoded_sample = encoder.encode_accumulation(events, class_id)
'''