import torch
from tqdm import tqdm
import torchhd

#torchh centroid approach

def train_one_epoch(dataloader, encoder, dimensions, num_classes, device):
    print("Starting training...")
    centroids = torchhd.models.Centroid(dimensions, num_classes).to(device)
    for batch_data, batch_labels in tqdm(dataloader, desc="Training Progress", unit="batch"):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        # Encode samples into hypervectors
        hv_batch = torch.stack([encoder.encode_sample(sample) for sample in batch_data])
        # Update centroids with the batch
        centroids.add(hv_batch, batch_labels)
    print("Training complete.")
    return centroids

def validate_one_epoch(dataloader, encoder, centroids, device):
    print("Starting validation...")
    correct = 0
    total = 0
    centroids = centroids.to(device)
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Validation Progress", unit="batch"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            # Encode samples into hypervectors
            hv_batch = torch.stack([encoder.encode_sample(sample) for sample in batch_data])
            # Predict using cosine similarity
            similarities = torchhd.functional.cosine_similarity(hv_batch, centroids())
            predictions = torch.argmax(similarities, dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

def evaluation(dataloader, encoder, centroids, device):
    print("Starting evaluation...")
    correct = 0
    total = 0
    centroids = centroids.to(device)
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Evaluation Progress", unit="batch"):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            hv_batch = torch.stack([encoder.encode_sample(sample) for sample in batch_data])
            #cosine similarity
            similarities = torchhd.functional.cosine_similarity(hv_batch, centroids())
            predictions = torch.argmax(similarities, dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%}")