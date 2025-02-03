import torch
import torchhd
from DAT_loadergrasp import GRASP_DAT_EventLoader
import os


def generate_random_bipolar(dims, device):
    """Generate a random bipolar hypervector."""
    return torchhd.random(1, 1, "MAP", device=device).float()

def interpolate_time_hv(T_start, T_end, alpha):
    """Interpolate between two hypervectors based on alpha position in time sub-window."""
    dims = T_start.shape[0]
    num_from_start = int((1 - alpha) * dims)
    return torch.cat((T_start[:num_from_start], T_end[num_from_start:]), dim=0)


class TorchHDEventEncoder:
    def __init__(self, sensor_height, sensor_width, dims=8000, time_subwindow=10000, device=None):
        """Hyperdimensional Encoding for Event Data."""
        self.height = sensor_height
        self.width = sensor_width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.device = device if device else torch.device("cpu")

        # Generate spatial and polarity hypervectors
        self.P = torch.empty((self.height, self.width, dims), device=self.device)
        for x in range(self.height):
            for y in range(self.width):
                self.P[x, y] = generate_random_bipolar(dims, self.device)

        self.I = torch.empty((2, dims), device=self.device)
        for p in range(2):
            self.I[p] = generate_random_bipolar(dims, self.device)

        self.T_boundaries = {}

    def get_time_boundary(self, boundary_index):
        """Retrieve or generate a seed temporal hypervector for time indexing."""
        if boundary_index not in self.T_boundaries:
            self.T_boundaries[boundary_index] = generate_random_bipolar(self.dims, self.device)
        return self.T_boundaries[boundary_index]

    def encode_timestamp(self, t):
        """Encode a timestamp into a hypervector using time sub-windows."""
        i = int(t // self.time_subwindow)
        alpha = (t - i * self.time_subwindow) / self.time_subwindow
        return interpolate_time_hv(self.get_time_boundary(i), self.get_time_boundary(i + 1), alpha)

    def encode_event(self, event):
        """Encode a single event as a hypervector."""
        t, (x, y), p = event
        return torchhd.bind(torchhd.bind(self.P[y, x], self.I[p]), self.encode_timestamp(t)).squeeze(0)

    def encode_sample(self, events):
        """Encode an entire sample of events into a single hypervector."""
        accumulator = torch.zeros(self.dims, device=self.device)
        for event in events:
            accumulator += self.encode_event(event)
        encoded = torch.sign(accumulator)
        encoded[encoded == 0] = 1  # Replace zero values with +1
        return encoded


def main():
    """Load dataset, encode 30 samples, and check similarity between class vectors."""
    device = "cpu"

    dataset_path = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT/"
    split_name = "val"

    event_loader = GRASP_DAT_EventLoader(root_dir=dataset_path, split=split_name, delta_t=10000, shuffle=True)

    #print(event_loader[0]) #correct format! :)

    encoder = TorchHDEventEncoder(sensor_height=480, sensor_width=640, dims=8000, time_subwindow=10000, device=device)

    print(f"Dataset Loaded: {len(event_loader)} samples.")

    encoded_vectors = []
    class_labels = []

    for sample_id, (filtered_events, class_id) in enumerate(event_loader):
        if len(filtered_events) == 0:
            continue  # Skip empty samples

        encoded_sample = encoder.encode_sample(filtered_events)
        encoded_vectors.append(encoded_sample)
        class_labels.append(class_id)

        print(f"Sample {sample_id} - Encoded Vector Shape: {encoded_sample.shape}, Class ID: {class_id}")

        if len(encoded_vectors) == 12:  # Limit to first x samples
            break

    # Convert to tensor for similarity analysis
    encoded_matrix = torch.stack(encoded_vectors)

    # Compute pairwise cosine similarity
    similarity_matrix = torch.nn.functional.cosine_similarity(encoded_matrix.unsqueeze(1), encoded_matrix.unsqueeze(0),
                                                              dim=-1)

    # Print similarity statistics
    print("\nCosine Similarity Between Encoded Vectors:")
    print(similarity_matrix)


if __name__ == "__main__":
    main()
