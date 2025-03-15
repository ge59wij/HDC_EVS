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
import seaborn as sns
import numpy as np
import tonic
import time
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchhd.models import Centroid
import seaborn as sns
import gc
gc.collect()

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)
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



# -------------------------------- Hyperparameters --------------------------------
TRAINING_METHOD = "centroid"  # "centroid" "adaptive"
LEARNING_RATE = 0.5
#["event_hd_timepermutation", "stem_hd" , "event_hd_timeinterpolation"]:
TIME_INTERPOLATION_METHOD = "thermometer"
ENCODING_METHOD = Raw_events_HDEncoder

if TIME_INTERPOLATION_METHOD in ["thermometer","linear"]:
    ENCODING_METHOD = Raw_events_HDEncoder_Enhanced

# thermometer, permutation,encode_temporalpermutation_weight


def main():
    device = "cpu"
    print(f"Using device: {device}")
    height, width = 34, 34
    split_name = "Train"
    # ------------------------ Parameters ------------------------
    dataset_name = "chifoumi"  # chifoumi
    dataset_path = "/space/chair-nas/tosy/data/"

    if dataset_name == "chifoumi":
        dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
        height = 480
        width = 640
        split_name="picked_samples"

    max_samples_train, max_samples_test = 70,30

    DIMS, K, Timewindow = 6000, 20 , 90_000

    WINDOW_SIZE_MS, OVERLAP_MS= 600_000, 200_000

    save =True

    # ------------------------ Load & Preprocess Dataset ------------------------
    dataset_train = load__dataset(dataset_path, split_name, max_samples_train, dataset_name)
    dataset_test = load__dataset(dataset_path, "picked_samples", max_samples_test, dataset_name)
    max_time = WINDOW_SIZE_MS
    print(f"[INFO] Using max_time = {max_time}")

    encoder = ENCODING_METHOD(
        height=height, width=width, dims=DIMS, time_subwindow=Timewindow, k=K, device=device,
        max_time=max_time, time_method=TIME_INTERPOLATION_METHOD,
        WINDOW_SIZE_MS=WINDOW_SIZE_MS, OVERLAP_MS=OVERLAP_MS
    )

    # ------------------------ Encode Train & Test Data ------------------------
    encoded_train, labels_train = encode_dataset(dataset_train, encoder, split_name=split_name)
    run_folder = create_unique_run_folder("/space/chair-nas/tosy/3.mars_after_fixes/test_run/") if save else None
    plot_heatmap(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
                 run_folder, "train")

    encoded_test, labels_test = encode_dataset(dataset_test, encoder, split_name="Test")

    # ------------------------ Save Data ------------------------
    run_folder = create_unique_run_folder("/space/chair-nas/tosy/3.mars_after_fixes/test_run/") if save else None
    if save:
        save_encoded_data(run_folder, encoded_train, labels_train, "train")
        save_encoded_data(run_folder, encoded_test, labels_test, "test")
        save_hyperparameters(run_folder, {
            "k": K, "Timewindow": Timewindow, "DIMS": DIMS, "Train_samples": len(dataset_train),
            "Test_samples": len(dataset_test), "Method": TIME_INTERPOLATION_METHOD
        })

    # ------------------------ Train & Test ------------------------
    model = train_model(encoded_train, labels_train, DIMS, len(set(labels_train)), TRAINING_METHOD)
    accuracy, preds = _test_model(model, encoded_test, labels_test)  # Now returns preds too
    print(f"Testing Accuracy: {accuracy:.3f}%")

    plot_heatmap(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
                 run_folder, "train")
    plot_heatmap(encoded_test, labels_test, K, Timewindow, DIMS, max_samples_test, TIME_INTERPOLATION_METHOD, save,
                 run_folder, "test")

    plot_tsne(encoded_train, labels_train, K, Timewindow, DIMS, max_samples_train, TIME_INTERPOLATION_METHOD, save,
              run_folder, "train")
    plot_tsne(encoded_test, labels_test, K, Timewindow, DIMS, max_samples_test, TIME_INTERPOLATION_METHOD, save,
              run_folder, "test")

    plot_confusion_matrix(labels_test, preds, save, run_folder, "test")


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

def train_model(encoded_matrix, labels, dims, num_classes, method):
    model = Centroid(dims, num_classes)
    with torch.no_grad():
        if method == "centroid":
            model.add(encoded_matrix, torch.tensor(labels, dtype=torch.long))
        elif method == "adaptive":
            model.add_adapt(encoded_matrix, torch.tensor(labels, dtype=torch.long), lr=LEARNING_RATE)
    model.normalize()
    return model


def _test_model(model, encoded_test, test_labels):
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=len(set(test_labels)))
    with torch.no_grad():
        output = model(encoded_test)
        preds = torch.argmax(output, dim=1)  # Get predicted class labels
        accuracy.update(output, torch.tensor(test_labels, dtype=torch.long))

    acc = accuracy.compute().item() * 100
    return acc, preds.tolist()  # Return accuracy and predictions


def plot_heatmap(vectors, labels, k, Timewindow, dims, max_samples, encodingmethod, save, run_folder, split_name):
    """Plots and saves cosine similarity heatmap with equal class representation."""
    class_labels_tensor = torch.tensor(labels)

    # Ensure balanced representation: Select 4 samples per class (or max available)
    unique_classes = list(set(labels))
    selected_indices = []
    for cls in unique_classes:
        indices = (class_labels_tensor == cls).nonzero(as_tuple=True)[0]
        selected_indices.extend(indices[:4])  # Take first 4 samples (if available)

    # Sort the selected indices for better organization
    selected_indices = torch.tensor(sorted(selected_indices))  # Ensure sorted order
    selected_vectors = vectors[selected_indices]
    selected_labels = class_labels_tensor[selected_indices].tolist()

    # Compute cosine similarity
    similarity_matrix = torchhd.functional.cosine_similarity(selected_vectors, selected_vectors).cpu().numpy()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        xticklabels=selected_labels, yticklabels=selected_labels,
        cbar=True, square=True, linewidths=0.5, annot_kws={"size": 7}
    )

    plt.title(
        f"Cosine Similarity Heatmap ({split_name}) | {encodingmethod} (k={k}, dims={dims}, timewindow={Timewindow})")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")

    # Save plot
    if save and run_folder:
        save_plot(run_folder, f"{split_name}_heatmap.png")
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