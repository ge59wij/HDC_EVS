import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader

data_dir = '/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train'
label_map_path = '/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/label_map_dictionary.json'
class EventGestureDataset(Dataset):
    def __init__(self, data_dir, label_map_path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        # Normalize the label map
        self.label_map = {k: v.lower() for k, v in self.label_map.items()}
        self._load_dataset()

    def _normalize_label(self, label):
        label = label.lower()
        if label == "scissors":
            label = "scissor"
        return label

    def _load_dataset(self):
        """Loads the dataset by reading .h5 and .npy files and mapping them to labels."""
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.h5'):
                # Extract and normalize the label from the file name
                label_name = self._normalize_label(file_name.split('_')[0])
                if label_name in self.label_map.values():
                    label = list(self.label_map.keys())[list(self.label_map.values()).index(label_name)]
                else:
                    print(f"Warning: Label '{label_name}' not found in label map.")
                    continue
                # Add the sample path and label
                h5_path = os.path.join(self.data_dir, file_name)
                bbox_path = os.path.join(self.data_dir, file_name.replace('.h5', '_bbox.npy'))
                self.samples.append((h5_path, bbox_path))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h5_path, bbox_path = self.samples[idx]
        label = self.labels[idx]
        # Load event data from .h5 file
        with h5py.File(h5_path, 'r') as h5_file:
            event_data = h5_file['data'][:]
        if os.path.exists(bbox_path):
            bbox_data = np.load(bbox_path)
        else:
            bbox_data = None
        # Convert event data to tensor
        event_tensor = torch.tensor(event_data, dtype=torch.float32)
        if self.transform:
            event_tensor = self.transform(event_tensor)
        return event_tensor, label

dataset = EventGestureDataset(data_dir, label_map_path)
batch_size = 36
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def test_dataset():
    print(f"Dataset size: {len(dataset)}")
    for i in range(3):  # Print first 3 samples
        event_tensor, label = dataset[i]
        print(f"Sample {i} - Event Tensor Shape: {event_tensor.shape}, Label: {label}")


def test_label_recognition():
    print(f"Dataset size: {len(dataset)}")
    for i in range(10):  # Check 5 samples
        event_tensor, label = dataset[i][:2]  # Ignore bbox_data if unused
        gesture_name = dataset.label_map[str(label)]
        print(f"Sample {i}: Label ID {label} -> Gesture Name: {gesture_name}")
import matplotlib.pyplot as plt
def plot_event_tensor(event_tensor, label, label_map, title="Event Tensor Sample"):
    gesture_name = label_map[str(label)]
    # Sum along the temporal dimension to get a spatial representation
    event_image = event_tensor.sum(dim=0).numpy()  # Shape: [2, 120, 160]
    # Plot both polarities
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(event_image[0], cmap='gray')
    axes[0].set_title(f"Positive Polarity - {gesture_name}")
    axes[1].imshow(event_image[1], cmap='gray')
    axes[1].set_title(f"Negative Polarity - {gesture_name}")
    plt.suptitle(title)
    plt.show()


def test_plot_event_tensor_with_name():
    """
    Test plotting of event tensors with gesture names from the dataset.
    """
    # Load label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    event_tensor, label = dataset[7]
    plot_event_tensor(event_tensor, label, label_map, title="Gesture Visualization")


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def visualize_event_tensor_as_video(event_tensor, label, label_map):
    """
    Visualizes the event tensor as a video by iterating over temporal bins.
 Event data tensor of shape [T, 2, H, W].
    """
    # Convert tensor to numpy array
    event_data = event_tensor.numpy()
    gesture_name = label_map[str(label)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Gesture: {gesture_name} (Label: {label})", fontsize=16)

    def update(frame_idx):
        for ax in axes:
            ax.clear()

        # Plot positive polarity
        axes[0].imshow(event_data[frame_idx, 0], cmap='gray', vmin=0, vmax=np.max(event_data))
        axes[0].set_title(f"Positive Polarity - Frame {frame_idx}")

        # Plot negative polarity
        axes[1].imshow(event_data[frame_idx, 1], cmap='gray', vmin=0, vmax=np.max(event_data))
        axes[1].set_title(f"Negative Polarity - Frame {frame_idx}")

    # Create animation
    anim = FuncAnimation(fig, update, frames=event_data.shape[0], interval=100)
    plt.show()


# Testing with a sample
def test_visualize_video():
    # Load label map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)

    # Select a sample
    event_tensor, label = dataset[100]
    visualize_event_tensor_as_video(event_tensor, label, label_map)




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    test_dataset()
    test_plot_event_tensor_with_name()
    test_visualize_video()

