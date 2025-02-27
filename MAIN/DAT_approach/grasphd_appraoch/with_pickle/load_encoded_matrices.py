import os
import pickle
import torch
import torchhd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from main_enc import plot_with_parameters, plot_tsne

# Define the run folder
run_folder = "/space/chair-nas/tosy/test_run/run_001"

def load_pickle_file(filepath):
    """Loads a pickle file and returns its contents as a dictionary."""
    if not os.path.exists(filepath):
        print(f"[MISSING] File not found: {filepath}")
        return None
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"[LOADED] {filepath}")
    return data

def load_hyperparameters(filepath):
    """Loads hyperparameters from params.txt, marking missing ones explicitly."""
    if not os.path.exists(filepath):
        print(f"[MISSING] File not found: {filepath}")
        return {}

    params = {}
    with open(filepath, "r") as f:
        for line in f:
            try:
                key, value = line.strip().split(": ")
                if value.isdigit():
                    params[key] = int(value)
                elif "." in value:
                    params[key] = float(value)
                else:
                    params[key] = value
            except ValueError:
                params[key] = "MISSING"

    print(f"[LOADED] Hyperparameters from {filepath}")
    return params

def main():
    param_file = os.path.join(run_folder, "params.txt")
    params = load_hyperparameters(param_file)

    # Get parameters, mark missing ones explicitly
    k = params.get("k", "MISSING")
    time_window = params.get("Timewindow", "MISSING")
    dims = params.get("DIMS", "MISSING")
    max_samples = params.get("max_samples_train", "MISSING")
    encoding_method = params.get("encoding_method", "MISSING")

    print(f"[INFO] Loaded Parameters: k={k}, Timewindow={time_window}, DIMS={dims}, Encoding={encoding_method}")

    # Load encoded matrix and class labels
    encoded_data = load_pickle_file(os.path.join(run_folder, "encoded_matrix.pkl"))
    if encoded_data:
        encoded_matrix = torch.tensor(encoded_data.get("encoded_matrix", []))
        class_labels = encoded_data.get("class_labels", [])

        if encoded_matrix.shape[0] == 0 or len(class_labels) == 0:
            print("[MISSING] Encoded matrix or class labels are empty.")
        else:
            print(f"[INFO] Loaded Encoded Matrix: {encoded_matrix.shape}, Labels: {len(class_labels)}")

            # Plot similarity matrix
            plot_with_parameters(encoded_matrix, class_labels, k, time_window, dims, max_samples, encoding_method, save=False)

            # Plot t-SNE visualization
            plot_tsne(encoded_matrix.numpy(), class_labels, k, time_window, dims, max_samples, encoding_method, save=False)

    # Load similarity matrix (if saved)
    sim_data = load_pickle_file(os.path.join(run_folder, "similarity_matrix.pkl"))
    if sim_data:
        similarity_matrix = sim_data.get("similarity_matrix", None)
        if similarity_matrix is None:
            print("[MISSING] Similarity matrix is empty.")
        else:
            plt.figure(figsize=(12, 10))
            sns.heatmap(similarity_matrix, cmap="coolwarm", annot=False)
            plt.title("Loaded Similarity Matrix")
            plt.xlabel("Sample Index")
            plt.ylabel("Sample Index")
            plt.show()

if __name__ == "__main__":
    main()
