import torch
import torch.nn as nn
import torchhd
from torchhd import embeddings, bind, normalize, multiset
from torchhd.models import Centroid
import torchmetrics
from tqdm import tqdm
"""
Encoding2: Spatial-Temporal Hypervector Encoding, Centroid
#Spatial and Temporal Encoding:
   - Spatial hypervectors: Initialize randomly for flattened height and width and channel
   - Temporal hypervectors: Random permutations for time bins to capture temporal dependencie
   - Binding: Combines spatial and temporal components to encode spatial-temporal patterns
   - Aggregation: Multiset operation aggregates hvs across time bins. (torchhd.normalize)
   - binary (-1 or 1) hvs, (efficient similarity computation)
#Classification with Centroid:
   - Centroid model handles class hypervector bundling using `.add()`.
   - Normalization ensures centroids are ready for inference.
   - Predictions are made using dot-product similarity.
#Validation and Metrics:
   - validation after each training epoch.
   - `torchmetrics.Accuracy` for multiclass classification evaluation.
#Training:
   - Encode spatial-temporal patterns into hvs.
   - Train the Centroid model by bundling class-specific hvs.
   - Normalize centroids after training.
#Validation:
   - after each epoch.
#Testing:
   -dot-product similarity.
"""


class Encoding2(nn.Module):
    def __init__(self, dimensions, max_time, height, width, batch_size, num_epochs, num_classes):
        super(Encoding2, self).__init__()
        self.dimensions = dimensions
        self.max_time = max_time
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self.spatial = torchhd.embeddings.Random(width * height, dimensions, vsa="MAP") #initialize random hvs for spatial dimensions (flattened height and width) and time bins
        self.temporal = torchhd.embeddings.Random(max_time, dimensions, vsa="MAP")

        self.centroid_model = Centroid(dimensions, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def encode(self, data):
        """
    Encode spatial, temporal, and channel components into hypervectors.
        data: Input histogram tensor of shape (batch_size, time_bins, channels, height, width)
    Returns: Encoded hypervectors of shape (batch_size, dimensions)
    """
        #print(f"Data shape: {data.shape}") ####debug torch.Size([1, 172, 2, 120, 160]) [batch, timebins, channels, height and weight)
        batch_size, time_bins, channels, height, width = data.shape

        data = data.flatten(3).flatten(2)  # Combine height and width   # Combine channels with spatial dimensions

        num_levels = 256

        # Normalize and discretize data into indices
        data = (data / data.max(dim=-1, keepdim=True).values) * (num_levels - 1)
        indices = data.round().long()  # Convert to integer indices


        hv = []

        # Store encoded hvs for each time bin

        for t in range(time_bins): # Uses indices for spatial embedding and bind with temporal hypervector
            t_tensor = torch.tensor([t], dtype=torch.long, device=data.device)
            temporal_hv = self.temporal(t_tensor)
            spatial_hv = bind(self.spatial(indices[:, t]), temporal_hv)
            assert indices[:, t].min() >= 0, "Indices contain negative values!"
            assert indices[:, t].max() < (
                        self.width * self.height), f"Indices exceed spatial embedding range! Max index: {indices[:, t].max()}"
            #spatial_hv = bind(self.spatial(indices), self.temporal(t)) #bind(self.spatial(data[:, t]), self.temporal[t])  # Spatial + temporal hvs binding => spatial-temporal
            hv.append(spatial_hv)

        hv = multiset(torch.stack(hv, dim=1))  # Aggregate across time bins "Multiset of input hvs"
        return normalize(hv)  #  Normalize to -1 and 1

    def train(self, train_loader, val_loader):
        """Train the Centroid model using encoded hypervectors and validate."""
        print("Starting training...")

        with torch.no_grad():
            for epoch in range(self.num_epochs):
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

                # Training loop
                for data, targets in tqdm(train_loader, desc="Training"):
                    data, targets = data.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)

                    sample_hv = self.encode(data)  # Encode input data
                    self.centroid_model.add(sample_hv, targets)  # Add to centroids

                self.centroid_model.normalize()  # Normalize centroids for inference

                # Validation step
                print("Validating...")
                self.validate(val_loader)

    def validate(self, val_loader):
        """Validate the Centroid model and compute accuracy."""
        validation_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(
            next(self.parameters()).device)

        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data, targets = data.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)

                sample_hv = self.encode(data)  # Encode validation samples
                predictions = self.centroid_model(sample_hv, dot=True)  # Classify samples

                validation_accuracy.update(predictions.cpu(), targets.cpu())

        print(f"Validation accuracy: {validation_accuracy.compute().item() * 100:.2f}%")

    def evaluate(self, test_loader):
        """Evaluate the Centroid model and compute accuracy."""
        print("Starting evaluation...")

        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc="Testing"):
                data, targets = data.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)

                sample_hv = self.encode(data)  # Encode test samples
                predictions = self.centroid_model(sample_hv, dot=True)  # Classify samples

                self.accuracy.update(predictions.cpu(), targets.cpu())

        print(f"Testing accuracy: {self.accuracy.compute().item() * 100:.2f}%")
