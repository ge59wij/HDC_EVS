import matplotlib.pyplot as plt
import seaborn as sns
from DAT_loadergrasp import GRASP_DAT_EventLoader
from grasphdencoding import GraspHDEventEncoder
import torch
from tqdm import tqdm
import torchhd


#todo: load from pickle files. batches like emg.
def main():
    """Encode samples and visualize similarities with a heatmap."""
    # Set the device
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset_path = "/space/chair-nas/tosy//space/chair-nas/tosy/preprocessed_dat_chifoumi/"
    split_name = "val"
    n = 20  # Number of samples to encode
    # Load data
    event_loader = GRASP_DAT_EventLoader(root_dir=dataset_path, split=split_name, delta_t=10000, shuffle=True)
    print(f"Total samples available in '{split_name}' split: {len(event_loader)}\n")

    encoder = GraspHDEventEncoder(height=480, width=640, dims=6000, k=6, device=device)

    encoded_vectors = []
    class_labels = []

    for sample_id, (filtered_events, class_id) in tqdm(enumerate(event_loader), total=n, desc="Encoding Samples"):
        print(f"Encoding Sample {sample_id} | Class: {class_id} | Device: {device}")

        # Ensure filtered events are moved to the correct device
        filtered_events = [(t, (x, y), p) for t, (x, y), p in filtered_events]

        # Encode filtered events
        encoded_sample = encoder.encode_temporal(filtered_events, class_id)
        encoded_sample = encoded_sample.to(device).squeeze()  # Ensure it's on the same device

        # Append results
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)

        # Stop after n samples
        if len(encoded_vectors) == n:
            break

    print("\nEncoding Complete. Generating similarity heatmap...\n")

    # Convert encoded vectors to a matrix
    encoded_matrix = torch.stack(encoded_vectors)  # Shape: [n, dims]
    print(encoded_matrix)

    # Compute cosine similarity using TorchHD
    similarity_matrix = torchhd.cosine_similarity(encoded_matrix, encoded_matrix)
    print(similarity_matrix)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(similarity_matrix.cpu().numpy(), annot=True, cmap="coolwarm", fmt=".2f",
                xticklabels=class_labels, yticklabels=class_labels, cbar=True)

    plt.title("Cosine Similarity Heatmap of Encoded Samples")
    plt.xlabel("Sample Index (Class ID)")
    plt.ylabel("Sample Index (Class ID)")

    cbar = heatmap.collections[0].colorbar
    cbar.set_label("Similarity Gradient (-1 to +1)")

    plt.show()


if __name__ == "__main__":
    main()
