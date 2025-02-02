import torch
from torchhd import embeddings, MAPTensor
from torchhd.functional import bind, bundle, normalize
from tqdm import tqdm
from torchhd.functional import ngrams

class EventEncoder:
    def __init__(self, dimensions, height, width, num_polarities, n_gram_size):
        self.dimensions = dimensions
        self.height = height
        self.width = width
        self.num_polarities = num_polarities
        self.n_gram_size = n_gram_size
        # Initialize embeddings
        self.polarity_embeddings = embeddings.Random(num_polarities, dimensions, device='cuda')
        self.spatial_embeddings = embeddings.Random(height * width, dimensions, device='cuda')

    def encode(self, event_tensor):
        """
        Encodes the event tensor into a hypervector.
        """
        # Convert event tensor indices to long for embedding lookup
        event_tensor_indices = event_tensor[:, :, 0, 0].long()

        # Generate polarity and spatial hypervectors
        polarity_hvs = self.polarity_embeddings(event_tensor_indices)
        spatial_hvs = self.spatial_embeddings(event_tensor_indices)

        # Bind polarity and spatial hypervectors
        bound_hvs = bind(polarity_hvs, spatial_hvs)

        # Aggregate bound hypervectors
        aggregated_hv = bound_hvs[0]
        for hv in bound_hvs[1:]:
            aggregated_hv = bundle(aggregated_hv, hv)

        # Create n-grams for temporal structure
        encoded_hv = ngrams(aggregated_hv, n=self.n_gram_size)
        return normalize(encoded_hv)


def encode_dataset(dataset, encoder, device, batch_size=1):
    """
    Encodes the entire dataset into hypervectors.
    """
    encoded_hypervectors = []
    labels = []

    print("Encoding dataset...")
    for i in tqdm(range(len(dataset)), desc="Encoding Dataset"):
        event_tensor, label = dataset[i]
        event_tensor = event_tensor.to(device)

        encoded_hv = encoder.encode(event_tensor)
        encoded_hypervectors.append(encoded_hv.cpu())
        labels.append(label)

    return torch.stack(encoded_hypervectors), torch.tensor(labels)
