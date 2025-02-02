########################
'''
around 33 accuracy, similar to emg, redo encodings
'''
########################

import os
import h5py
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchhd.models import Centroid
from torchhd import embeddings
from torchhd.functional import bind, bundle, normalize, ngrams
from torchmetrics import Accuracy
from tqdm import tqdm
class EventGestureDataset(Dataset):
    def __init__(self, data_dir, label_map_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.label_map = {k: v.lower() for k, v in self.label_map.items()}
        self._load_dataset()
    def _normalize_label(self, label):
        label = label.lower()
        if label == "scissors": label = "scissor"
        return label
    def _load_dataset(self):
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.h5'):
                label_name = self._normalize_label(file_name.split('_')[0])  #not best but for now ignore npy labeles
                if label_name in self.label_map.values():
                    label = list(self.label_map.keys())[list(self.label_map.values()).index(label_name)]
                else:
                    print(f"Warning: Label '{label_name}' not found in label map.")
                    continue
                h5_path = os.path.join(self.data_dir, file_name)
                self.samples.append(h5_path)
                self.labels.append(int(label))

        print(f"Loaded {len(self.samples)} samples from {self.data_dir}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        h5_path = self.samples[idx]
        label = self.labels[idx]
        with h5py.File(h5_path, 'r') as h5_file:
            event_data = h5_file['data'][:]
        event_tensor = torch.tensor(event_data, dtype=torch.float32)

        if self.transform:
            event_tensor = self.transform(event_tensor)
        return event_tensor, label

class EventEncoder:
    def __init__(self, dimensions, height, width, num_polarities, n_gram_size):
        self.dimensions = dimensions
        self.height = height
        self.width = width
        self.num_polarities = num_polarities
        self.n_gram_size = n_gram_size
        self.polarity_embeddings = embeddings.Random(num_polarities, dimensions, device='cuda')
        self.spatial_embeddings = embeddings.Random(height * width, dimensions, device='cuda')

    def encode(self, event_tensor):
        event_tensor_indices = event_tensor[:, :, 0, 0].long()
        polarity_hvs = self.polarity_embeddings(event_tensor_indices)
        spatial_hvs = self.spatial_embeddings(event_tensor_indices)
        bound_hvs = bind(polarity_hvs, spatial_hvs)
        aggregated_hv = bound_hvs[0]
        for hv in bound_hvs[1:]:
            aggregated_hv = bundle(aggregated_hv, hv)
        if self.n_gram_size > 1:
            aggregated_hv = ngrams(aggregated_hv, n=self.n_gram_size)
        return normalize(aggregated_hv)

def encode_dataset(dataset, encoder, device, batch_size=1):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoded_hypervectors = []
    labels = []

    print("Encoding dataset...")
    for event_tensors, batch_labels in tqdm(data_loader, desc="Encoding Dataset"):
        event_tensors = event_tensors.to(device)
        batch_encoded_hvs = [encoder.encode(event_tensor) for event_tensor in event_tensors]
        encoded_hypervectors.extend(batch_encoded_hvs)
        labels.extend(batch_labels)
    return torch.stack(encoded_hypervectors), torch.tensor(labels)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    dimensions = 10000
    height, width = 120, 160
    num_polarities = 2
    n_gram_size = 4
    batch_size = 32

    data_dir = '/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train'
    label_map_path = '/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/label_map_dictionary.json'
    dataset = EventGestureDataset(data_dir, label_map_path)
    unique_labels = set(dataset.labels)
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    encoder = EventEncoder(dimensions, height, width, num_polarities, n_gram_size)
    encoded_data, labels = encode_dataset(dataset, encoder, device)
    # Create DataLoader for hypervectors
    encoded_dataset = torch.utils.data.TensorDataset(encoded_data, labels)
    data_loader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True)
    model = Centroid(dimensions, num_classes).to(device)
    print("Starting training...")
    with torch.no_grad():
        for hypervectors, batch_labels in tqdm(data_loader, desc="Training"):
            hypervectors = hypervectors.to(device)
            batch_labels = batch_labels.to(device)
            model.add(hypervectors, batch_labels)
    model.normalize()
    # Testing
    test_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    for test_hypervectors, test_labels in tqdm(data_loader, desc="Testing"):
        test_hypervectors = test_hypervectors.to(device)
        test_labels = test_labels.to(device)
        outputs = model(test_hypervectors, dot=True)
        test_accuracy.update(outputs.cpu(), test_labels.cpu())
    print(f"Testing accuracy: {test_accuracy.compute().item() * 100:.2f}%")
