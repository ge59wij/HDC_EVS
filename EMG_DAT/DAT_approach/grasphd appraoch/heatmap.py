import matplotlib.pyplot as plt
import seaborn as sns
from DAT_loadergrasp import GRASP_DAT_EventLoader
from grasphdencoding_seedhvs import GraspHDseedEncoder
from grasphdencoding import GraspHDEventEncoder
import torch
from tqdm import tqdm



def main():
    """Encode 30 samples and visualize similarities with a heatmap."""
    device = "cpu"
    dataset_path = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT/"
    split_name = "val"

    # Load dataset
    event_loader = GRASP_DAT_EventLoader(root_dir=dataset_path, split=split_name, delta_t=10000, shuffle=True)

    print(f" Total samples available in '{split_name}' split: {len(event_loader)}\n")
    first_sample, first_label = event_loader[0]
    print(f" First Sample Label: {first_label}")
    print(f"First Sample Example (first 5 events): {first_sample[:5]}\n")
    print("encoding..")
    encoded_vectors = []
    class_labels = []
    for sample_id, (filtered_events, class_id) in tqdm(enumerate(event_loader), total=30, desc="Encoding Samples"):

        # Create Encoder
        encoder = GraspHDEventEncoder(height=480, width=640, dims=8000, k=2, device=device)

        # Encode using spatial and temporal encoding
        encoded_sample = encoder.encode_temporal(filtered_events)

        # Debugging prints for shape and device check
        print(f"Sample {sample_id} Encoded | Shape: {encoded_sample.shape} | Device: {encoded_sample.device}")

        # Store results
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)

        if len(encoded_vectors) == 30:  # Stop after 30 samples
            break

    print("\n Encoding Complete. Generating similarity heatmap...\n")
    # Convert to tensor
    encoded_matrix = torch.stack(encoded_vectors)

    # Compute cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(
        encoded_matrix.unsqueeze(1), encoded_matrix.unsqueeze(0), dim=-1
    )

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix.numpy(), annot=False, cmap="coolwarm", xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title("Cosine Similarity Heatmap of Encoded Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.show()


if __name__ == "__main__":
    main()