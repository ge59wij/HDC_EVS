import torch
import torchhd
import os
import pickle
from tqdm import tqdm
from grasphdencoding import GraspHDEventEncoder
from torchhd.models import Centroid
import random
import torchmetrics
import torchhd.utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)


TRAINING_METHOD = "adaptive"  # "centroid" "adaptive" "iterative"
LEARNING_RATE = 0.5
ENCODING_METHOD = GraspHDEventEncoder # xx
TIME_INTERPOLATION_METHOD = "event_hd"  # "grasp_hd", "event_hd"   encode_temporalpermutation, thermometer

def main():
    device = "cpu"
    print(f"Using device: {device}")
    dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi"
    max_samples_train = 25
    max_samples_test = 4
    DIMS = 8000
    K = 4
    Timewindow = 500000


    dataset_train = load_pickle_dataset(dataset_path, split="train", max_samples=max_samples_train)
    dataset_test = load_pickle_dataset(dataset_path, split="test", max_samples=max_samples_test)

    max_time_train = get_max_time(dataset_train)
    max_time_test = get_max_time(dataset_test)
    max_time = max(max_time_train, max_time_test)
    print(f"[INFO] Computed max_time: {max_time} (Train: {max_time_train}, Test: {max_time_test})")

    '''
    print("\n[DEBUG] First 10 events of first 10 training samples:")
    for i, (events, class_id) in enumerate(dataset_train[:3]):
        print(f"Sample {i} (Class {class_id}): {events[:25]}")
        target_timestamp = 9000
        tolerance = 2  # small window around the target timestamp
        for i, (events, class_id) in enumerate(dataset_train[:1]):  # Only first 2 samples
            filtered_events = [event for event in events if abs(event[0] - target_timestamp) <= tolerance]
            print(f"Sample {i} (Class {class_id}), Events around {target_timestamp}: {filtered_events}")
    '''


    encoder = ENCODING_METHOD(height=480, width=640, dims=DIMS, time_subwindow=Timewindow, k=K, device=device,
                              max_time= max_time, time_method= TIME_INTERPOLATION_METHOD)

    print_debug(TIME_INTERPOLATION_METHOD, dataset_train, encoder, max_time, Timewindow, K)

    encoded_vectors, class_labels = [], []

    for sample_id, (events, class_id) in tqdm(enumerate(dataset_train), total=len(dataset_train),
                                              desc="Encoding Train Samples"):
        if TIME_INTERPOLATION_METHOD == "event_hd":
            encoded_sample = encoder.encode_eventhd(events, class_id)
        elif TIME_INTERPOLATION_METHOD == "grasp_hd":
            encoded_sample = encoder.encode_grasphd(events, class_id)
        elif TIME_INTERPOLATION_METHOD == "linear_interpolation":
            encoded_sample = encoder.linear_interpolation(events, class_id)
        elif TIME_INTERPOLATION_METHOD == "encode_temporalpermutation":
            encoded_sample = encoder.encode_temporalpermutation(events, class_id)
        elif TIME_INTERPOLATION_METHOD in [ "thermometer", "permutatin"]:
            encoded_sample = encoder.encode_accumulation(events, class_id)


        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)

    encoded_matrix = torch.stack(encoded_vectors)
    print(f"Encoded Matrix Stats - Min: {encoded_matrix.min()}, Max: {encoded_matrix.max()}")

    #---------------------------Training--------------
    '''
    label_tensor = torch.tensor(class_labels, dtype=torch.long, device=device)  # Keep labels as tensor
    model = train_model(encoded_matrix, label_tensor, DIMS, len(set(class_labels)), TRAINING_METHOD)


    # **Train Centroid Classifier (Batch)**
    model = Centroid(DIMS, len(set(class_labels)))
    with torch.no_grad():
        model.add(encoded_matrix, label_tensor)  # Batch adding instead of looping one by one
        model.normalize()
    '''
    #---------------------------Testing--------------
    '''

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=len(set(class_labels)))
    
    encoded_test_vectors, test_labels = [], []

    for sample_id, (events, class_id) in tqdm(enumerate(dataset_test), total=len(dataset_test),
                                              desc="Encoding Test Samples"):
        encoded_sample = encoder.encode_eventhd(events, class_id) if TIME_INTERPOLATION_METHOD == "weighted_time" \
            else encoder.encode_grasphd(events, class_id)
        encoded_test_vectors.append(encoded_sample)
        test_labels.append(class_id)

    encoded_test_matrix = torch.stack(encoded_test_vectors)
    test_label_tensor = torch.tensor(test_labels, dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(encoded_test_matrix, dot=True)  #
        accuracy.update(output.cpu(), test_label_tensor.cpu())
    print(f"Run with D = {DIMS}, k = {K}, on {device},\n T= {Timewindow}, "
          f"L= {LEARNING_RATE} on {TRAINING_METHOD}\n {max_samples_train} trainsamples, {max_samples_test} testsmaples.")
    print(f"Testing Accuracy: {(accuracy.compute().item() * 100):.3f}%")
    #plot_with_parameters(encoded_test_matrix, test_labels, K ,Timewindow, DIMS ,max_samples_train, ENCODING_METHOD)
    
    
    '''
    plot_with_parameters(encoded_matrix, class_labels, K ,Timewindow, DIMS ,max_samples_train, TIME_INTERPOLATION_METHOD)
    plot_tsne(encoded_vectors, class_labels,K ,Timewindow, DIMS ,max_samples_train, TIME_INTERPOLATION_METHOD )



def print_debug(TIME_INTERPOLATION_METHOD, dataset, encoder, max_time, Timewindow, K):

    if TIME_INTERPOLATION_METHOD == "weighted_time":
        # Encode one sample first
        print("\n[DEBUG] Encoding First Sample")
        events, class_id = dataset[0]
        encoder.encode_eventhd(events, class_id)
        #dynamically created time keys
        eventhd_time_keys = sorted(encoder.time_hvs.keys())  # Only populated keys after encoding!
        print(f"\n[DEBUG] Retrieved {len(eventhd_time_keys)} cached time keys after first encoding.")
        # Sampled checks: First 5 consecutive timestamps, then compare across different bins
        sample_indices = list(range(5)) + [len(eventhd_time_keys) // 2, len(eventhd_time_keys) - 1]

        print("\n[DEBUG] Sampling Time Hypervector Similarities Across Different Bins:")
        for i in sample_indices[:-1]:
            t1, t2 = eventhd_time_keys[i], eventhd_time_keys[i + 1]  # Compare adjacent timestamps
            T_t1 = encoder.get_time_hv(t1)
            T_t2 = encoder.get_time_hv(t2)
            similarity = torchhd.cosine_similarity(T_t1, T_t2).item()
            print(f"Time HV similarity (t={t1} vs. t={t2}): {similarity:.4f}")

        # Compare across larger bin differences
        print("\n[DEBUG] Comparing Across Distant Time Bins:")
        distant_pairs = [
            (eventhd_time_keys[0], eventhd_time_keys[len(eventhd_time_keys) // 3]),
            (eventhd_time_keys[len(eventhd_time_keys) // 3], eventhd_time_keys[2 * len(eventhd_time_keys) // 3]),
            (eventhd_time_keys[2 * len(eventhd_time_keys) // 3], eventhd_time_keys[-1])
        ]

        for t1, t2 in distant_pairs:
            T_t1 = encoder.get_time_hv(t1)
            T_t2 = encoder.get_time_hv(t2)
            similarity = torchhd.cosine_similarity(T_t1, T_t2).item()
            print(f"Time HV similarity (t={t1} vs. t={t2}): {similarity:.4f}")   #Time HVs Are Too Similar Across Consecutive Timestamps. cos=1! too many vectors inside one bin! 1microsecomd incrementation!
            #474,365 cached time hypervectors!!!!!!!!

    if TIME_INTERPOLATION_METHOD == "grasp_hd":
        # GraspHD needs increments of 0.5 bins
        bin_keys = sorted(encoder.time_hvs.keys())  # Get all existing time bins (anchors & interpolated)
        for i in range(len(bin_keys) - 1):
            t1, t2 = bin_keys[i], bin_keys[i + 1]  # Compare adjacent bins (both anchors & interpolated)
            T_t1 = encoder.get_time_hv(t1)
            T_t2 = encoder.get_time_hv(t2)
            similarity = torchhd.cosine_similarity(T_t1, T_t2).item()
            print(f"Time HV similarity (t={t1} vs. t={t2}): {similarity:.4f}")

    print("\n[DEBUG] Checking Similarity of Position Hypervectors")

    # Iterate over the first few k-windows
    for x in range(0, 3 * K, K):
        for y in range(0, K + 1, K):
            # Define key points inside the window
            P00 = encoder.get_position_hv(x, y)  # Top-left corner
            P01 = encoder.get_position_hv(x, min(y + K, 480 - 1))  # Top-right corner
            P10 = encoder.get_position_hv(min(x + K, 640 - 1), y)  # Bottom-left corner
            P11 = encoder.get_position_hv(min(x + K, 640 - 1), min(y + K, 480 - 1))  # Bottom-right

            # Choose three sample points inside the k-window
            P_inside_1 = encoder.get_position_hv(x + K // 4, y + K // 4)  # Close to top-left
            P_inside_2 = encoder.get_position_hv(x + K // 2, y + K // 2)  # Dead center
            P_inside_3 = encoder.get_position_hv(x + 3 * K // 4, y + 3 * K // 4)  # Close to bottom-right

            # Compute similarities inside the window
            print(f"\n[Window ({x},{y}) - ({x + K},{y + K})]")
            print(f"  P00 vs P01 (Left-Right Corner): {torchhd.cosine_similarity(P00, P01).item():.4f}")
            print(f"  P00 vs P10 (Top-Bottom Corner): {torchhd.cosine_similarity(P00, P10).item():.4f}")
            print(f"  P01 vs P11 (Right Corners): {torchhd.cosine_similarity(P01, P11).item():.4f}")
            print(f"  P10 vs P11 (Bottom Corners): {torchhd.cosine_similarity(P10, P11).item():.4f}")

            print(f"  P_inside_1 vs P00 (Close to Top-Left): {torchhd.cosine_similarity(P_inside_1, P00).item():.4f}")
            print(f"  P_inside_2 vs P00 (Center vs Top-Left): {torchhd.cosine_similarity(P_inside_2, P00).item():.4f}")
            print(
                f"  P_inside_2 vs P11 (Center vs Bottom-Right): {torchhd.cosine_similarity(P_inside_2, P11).item():.4f}")
            print(
                f"  P_inside_3 vs P11 (Close to Bottom-Right): {torchhd.cosine_similarity(P_inside_3, P11).item():.4f}")


def plot_pairwise_similarity(encoded_matrix, class_labels):
    fig, ax = plt.subplots(figsize=(10, 8))
    mappable = torchhd.utils.plot_pair_similarity(encoded_matrix, ax=ax)
    plt.colorbar(mappable, ax=ax)
    num_samples = encoded_matrix.shape[0]
    tick_positions = np.arange(num_samples)
    class_labels_str = [str(label) for label in class_labels]  # Convert labels to strings
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(class_labels_str, rotation=90)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(class_labels_str)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Class ID")
    ax.set_title("Pairwise Cosine Similarity of Encoded Samples")
    plt.show()
def train_model(encoded_matrix, label_tensor, dims, num_classes, method):
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
def plot_with_parameters(vectors_matrix, class_labels, k, Timewindow, dims, max_samples, encodingmethod):

    similarity_matrix = torchhd.functional.cosine_similarity(vectors_matrix, vectors_matrix).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        square=True,
        linewidths=0.5
    )
    plt.title(f"Cosine Similarity {encodingmethod} Heatmap (k={k}| dims={dims}| timewindow={Timewindow}| samples={max_samples})")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")
    plt.show()

def plot_tsne(encoded_vectors, class_labels, k, Timewindow, dims, max_samples, encodingmethod):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)

    encoded_vectors = np.array(encoded_vectors)  #  list to NumPy array
    reduced_vectors = tsne.fit_transform(encoded_vectors)
    palette = {0: "red", 1: "blue", 2: "green"}

    # Scatter plot with fixed colors
    plt.figure(figsize=(8, 6))
    for class_id in np.unique(class_labels):
        plt.scatter(reduced_vectors[np.array(class_labels) == class_id, 0],
                    reduced_vectors[np.array(class_labels) == class_id, 1],
                    label=f"Class {class_id}",
                    color=palette[class_id], alpha=0.7, edgecolors='k')

    # Plot settings
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE Visualization, {encodingmethod} Heatmap (k={k}| dims={dims}| timewindow={Timewindow}| samples={max_samples})")
    plt.legend()
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