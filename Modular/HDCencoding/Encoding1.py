import torch
import torchhd
from torch.utils.data import DataLoader
from torchhd.models import Centroid
import torchmetrics
from tqdm import tqdm


class Encoding1:
    def __init__(self, dimensions, max_time, height, width, device=None):
        self.dimensions = dimensions
        self.max_time = max_time
        self.height = height
        self.width = width
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Centroid(dimensions, 3).to(self.device)
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(self.device)

    def train(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        print("Training started...")
        for epoch in range(10):  # Example: 10 epochs
            for hist_tensor, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                hist_tensor = hist_tensor.to(self.device)
                label = label.to(self.device)

                # Encoding logic
                hv = self.encode(hist_tensor)
                self.model.add(hv.unsqueeze(0), label)

        self.model.normalize()
        print("Training completed.")

        self._validate(val_loader)

    def encode(self, hist_tensor):
        # Custom encoding logic (simplified for brevity)
        return torch.rand(self.dimensions, device=self.device)

    def _validate(self, val_loader):
        print("Validating...")
        self.model.eval()
        self.accuracy_metric.reset()

        with torch.no_grad():
            for hist_tensor, label in val_loader:
                hist_tensor = hist_tensor.to(self.device)
                label = label.to(self.device)
                hv = self.encode(hist_tensor)
                preds = self.model(hv.unsqueeze(0), dot=True).argmax(dim=1)
                self.accuracy_metric.update(preds, label)

        val_acc = self.accuracy_metric.compute().item()
        print(f"Validation Accuracy: {val_acc:.2%}")

    def evaluate(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print("Testing...")
        self._validate(test_loader)
