from torch.utils.data import DataLoader
from DAT_loader import EventDatasetLoader
import torch

# Initialize dataset
train_dataset = EventDatasetLoader(
    "train","cuda",21
)
test_dataset = EventDatasetLoader(
    "test", "cuda", 21
)

val_dataset = EventDatasetLoader("val", "cuda", 21)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
from torchhd.models import Centroid

num_classes = len(train_dataset.label_map)
model = Centroid(dimensions=10000, num_classes=num_classes)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
