import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchhd
from torchhd.embeddings import Random
from torchhd.models import Centroid
import torchmetrics
from tqdm import tqdm  # For the progress bar
from Chifoumi_to_Pytensors import EventDatasetLoader

DIMENSIONS = 8000   # HD dimension (tweak as you like)
MAX_TIME   = 600    # maximum time bins expected
HEIGHT     = 120
WIDTH      = 160
NUM_EPOCHS = 1
##############################################################################
####################SpatioTemporal Encoder
##############################################################################
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, dimensions, max_time, height, width, device=None):
        super().__init__()
        self.dimensions = dimensions
        self.max_time = max_time
        self.height = height
        self.width = width
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # (time, polarity, position) random embeddings
        self.time_embed = Random(
            num_embeddings=self.max_time,
            embedding_dim=self.dimensions,
            vsa="MAP",
            device="cpu",  # store on CPU to avoid big GPU memory usage?
            requires_grad=False,
        )

        self.polarity_embed = Random(
            num_embeddings=2,  # ON/OFF polarity
            embedding_dim=self.dimensions,
            vsa="MAP",
            device="cpu",
            requires_grad=False,
        )

        self.position_embed = Random(
            num_embeddings=self.height * self.width,
            embedding_dim=self.dimensions,
            vsa="MAP",
            device="cpu",
            requires_grad=False,
        )

    def forward(self, hist_tensor: torch.Tensor) -> torch.Tensor:
        """
        hist_tensor: shape [T, 2, H, W], on GPU ( move the embeddings on the fly).
        Returns: HV shape [dimensions] (MAP).
        """
        T, C, H, W = hist_tensor.shape
        gesture_hv = torch.zeros(self.dimensions, device=hist_tensor.device)

        # We'll move embeddings from CPU->GPU as needed:
        time_weight = self.time_embed.weight.to(hist_tensor.device)
        pol_weight = self.polarity_embed.weight.to(hist_tensor.device)
        pos_weight = self.position_embed.weight.to(hist_tensor.device)

        for t in range(T):
            t_idx = min(t, self.max_time - 1)  # clamp
            time_hv = time_weight[t_idx]

            for p in range(C):  # 0 or 1
                pol_hv = pol_weight[p]
                plane = hist_tensor[t, p]  # shape [H, W]
                coords = torch.nonzero(plane, as_tuple=False)

                for loc in coords:
                    y, x = loc[0].item(), loc[1].item()
                    intensity = plane[y, x].item()
                    pos_idx = y * self.width + x

                    pos_hv = pos_weight[pos_idx]
                    # bind
                    event_hv = torchhd.functional.bind(pos_hv, pol_hv)
                    event_hv = torchhd.functional.bind(event_hv, time_hv)
                    gesture_hv += (event_hv * intensity)

        gesture_hv = torch.sign(gesture_hv)
        return gesture_hv

##############################################################################
#######training
##############################################################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/"
    print("Loading Dataset")
    train_dataset = EventDatasetLoader(dataset_path, "train")
    val_dataset   = EventDatasetLoader(dataset_path, "val")
    test_dataset  = EventDatasetLoader(dataset_path, "test")

    #
    #batch_size=1 to avoids complicated padding for variable T, add padding later
    #multiple workers to speed up reading.
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # SpatioTemporal MAP encoder
    encoder = SpatioTemporalEncoder(
        DIMENSIONS, MAX_TIME, HEIGHT, WIDTH, device=device
    )

    # HD Centroid model (3 classes: paper, rock, scissor), for now built in centroid
    model = Centroid(DIMENSIONS, 3).to(device)

    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)

    ############################################################################
    # Training Loop
    ############################################################################
    print("Starting training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=True)

        for (hist_tensor, label) in train_iter:
            hist_tensor = hist_tensor.squeeze(0)  # shape [T, 2, H, W]
            # Move to GPU
            hist_tensor = hist_tensor.to(device)
            label = label.to(device)
            # Encode
            gesture_hv = encoder(hist_tensor)
            # Add to model
            model.add(gesture_hv.unsqueeze(0), label)

    print("Finished training. Normalizing centroids...")
    model.normalize()

    ############################################################################
    # Validation Loop
    ############################################################################
    print("Validating...")
    model.eval()
    accuracy_metric.reset()

    val_iter = tqdm(val_loader, desc="Validation", leave=True)
    with torch.no_grad():
        for (hist_tensor, label) in val_iter:
            hist_tensor = hist_tensor.squeeze(0).to(device)
            label = label.to(device)

            hv = encoder(hist_tensor)
            outputs = model(hv.unsqueeze(0), dot=True)
            preds = outputs.argmax(dim=1)
            accuracy_metric.update(preds, label)

    val_acc = accuracy_metric.compute().item()
    print(f"Validation Accuracy: {val_acc*100:.2f}%")

    ############################################################################
    # Test Loop
    ############################################################################
    print("Testing...")
    model.eval()
    accuracy_metric.reset()

    test_iter = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for (hist_tensor, label) in test_iter:
            hist_tensor = hist_tensor.squeeze(0).to(device)
            label = label.to(device)

            hv = encoder(hist_tensor)
            outputs = model(hv.unsqueeze(0), dot=True)
            preds = outputs.argmax(dim=1)
            accuracy_metric.update(preds, label)

    test_acc = accuracy_metric.compute().item()
    print(f"Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
