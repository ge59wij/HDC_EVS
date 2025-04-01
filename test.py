from torchhd.functional import hard_quantize
import torch
import torchhd
import os
from torchhd.functional import normalize
import torch
import numpy as np
'''
tensor = torch.tensor([0.2, -0.5, 0.7])
binary_tensor = torchhd.normalize(tensor)
print(binary_tensor)


tensor = torch.tensor([0.2, -0.5, 0.7])
binary_tensor = normalize(tensor)
print(binary_tensor)

num_workers = os.cpu_count()
print(num_workers)


np.set_printoptions(suppress=True, precision=8)
torch.set_printoptions(sci_mode=False, precision=5)


def xtest_torchhd_operations():
    dims = 10000  # Dimensionality of hypervectors
    device = "cpu"

    print("\nGenerating Random Hypervectors...\n")
    hv_1 = torchhd.random(1, dims, "MAP", device=device).squeeze(0)
    hv_2 = torchhd.random(1, dims, "MAP", device=device).squeeze(0)

    print("First 10 values of hv_1:", hv_1[:10])
    print("First 10 values of hv_2:", hv_2[:10])
    print("Mean hv_1:", hv_1.mean().item())
    print("Std hv_1:", hv_1.std().item())

    print("\nChecking if Values are Truly Bipolar (-1,1)...")
    unique_values = torch.unique(hv_1).tolist()
    print("Unique values in hv_1:", unique_values)

    print("\nApplying torchhd.normalize()...\n")
    hv_1_norm = torchhd.normalize(hv_1)
    print("First 10 values after normalization:", hv_1_norm[:10])
    print("Mean after normalization:", hv_1_norm.mean().item())
    print("Std after normalization:", hv_1_norm.std().item())

    print("\nTesting Bundling (Summation)\n")
    bundled_hv = torchhd.bundle(hv_1, hv_2)
    print("First 10 values of bundled HV:", bundled_hv[:10])
    print("Mean bundled:", bundled_hv.mean().item())
    print("Std bundled:", bundled_hv.std().item())

    print("\nTesting Binding (Multiplication/XOR-like Operation)\n")
    bound_hv = torchhd.bind(hv_1, hv_2)
    print("First 10 values of bound HV:", bound_hv[:10])
    print("Mean bound:", bound_hv.mean().item())
    print("Std bound:", bound_hv.std().item())

    print("\nInterpolation Test (For Time Encoding)\n")
    alpha = 0.3
    interpolated_hv = (1 - alpha) * hv_1 + alpha * hv_2
    print("First 10 values of interpolated HV:", interpolated_hv[:10])
    print("Mean interpolated:", interpolated_hv.mean().item())
    print("Std interpolated:", interpolated_hv.std().item())


if __name__ == "__main__":
    xtest_torchhd_operations()

print("\n Generating random bipolar hvs:")
hv_1 = torchhd.random(1000, 1000, "MAP")  # 1000 hypervectors, 1000 dimensions
hv_2 = torchhd.random(1000, 1000, "MAP")
# Stats before normalization
print("\nBefore Normalization:")
print(f"Mean hv_1: {hv_1.mean().item():.6f}")
print(f"Std hv_1: {hv_1.std().item():.6f}")
print(f"Unique values in hv_1: {torch.unique(hv_1)}")  # Should be only -1 and 1

# =============================
# 2. NORMALIZATION TEST
# =============================
print("\n applying torchhdnormalize")
hv_1_norm = torchhd.normalize(hv_1)
print("\nAfter Normalization:")
print(f"Mean after normalization: {hv_1_norm.mean().item():.6f}")
print(f"Std after normalization: {hv_1_norm.std().item():.6f}")
print(f"Unique values after normalization: {torch.unique(hv_1_norm)}")

# Does normalization change anything? If it doesn't, then it **might not be needed**.

# =============================
# 3. TESTING BUNDLING
print("\n Testing Bundling (Summation of Hypervectors)...")

bundled_hv = torchhd.bundle(hv_1, hv_2)

print("\nBefore Normalization (Bundled HV):")
print(f"Mean bundled: {bundled_hv.mean().item():.6f}")
print(f"Std bundled: {bundled_hv.std().item():.6f}")
print(f"Unique values in bundled HV: {torch.unique(bundled_hv)}")  # Should have values other than -1,1

# =============================
# 4. NORMALIZING BUNDLED HYPERVECTOR
# =============================

print("\n Normalizing Bundled Hypervector...")

bundled_hv_norm = torchhd.normalize(bundled_hv)

print("\nAfter Normalization (Bundled HV):")
print(f"Mean bundled normalized: {bundled_hv_norm.mean().item():.6f}")
print(f"Std bundled normalized: {bundled_hv_norm.std().item():.6f}")
print(f"Unique values in bundled HV after normalization: {torch.unique(bundled_hv_norm)}")

# Does this bring values back to -1,1? Or does it still contain fractional values?

# =============================
# 5. TESTING INTERPOLATION (FOR TIME ENCODING)
# =============================

print("\n Interpolation Test (time encdoing")

T_1 = torchhd.random(1, 1000, "MAP")  # Simulating time vector at T1
T_2 = torchhd.random(1, 1000, "MAP")  # Simulating time vector at T2

# Interpolating between T_1 and T_2 (50%)
alpha_t = 0.5
interpolated_hv = alpha_t * T_1 + (1 - alpha_t) * T_2

print("\nBefore Binarization (Interpolated HV):")
print(f"Mean interpolated: {interpolated_hv.mean().item():.6f}")
print(f"Std interpolated: {interpolated_hv.std().item():.6f}")
print(f"Unique values in interpolated HV: {torch.unique(interpolated_hv)[:10]} (Showing 10 values)")  # Should be fractional

# =============================
# 6. BINARIZING INTERPOLATED VECTOR
# =============================
print("\n applying binarization (threshloding at 0)")

binarized_hv = torch.where(interpolated_hv > 0, torch.tensor(1.0), torch.tensor(-1.0))

print("\nAfter Binarization (Interpolated HV):")
print(f"Mean binarized: {binarized_hv.mean().item():.6f}")
print(f"Std binarized: {binarized_hv.std().item():.6f}")
print(f"Unique values in binarized HV: {torch.unique(binarized_hv)}")  # Should be only -1 and 1


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
import torchmetrics

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000
IMG_SIZE = 28
NUM_LEVELS = 1000
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones
Datasets
Next, we load the training and testing datasets:

transform = torchvision.transforms.ToTensor()

train_ds = MNIST("../data", train=True, transform=transform, download=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST("../data", train=False, transform=transform, download=True)
test_ld = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
In addition to the various datasets available in the Torch ecosystem, such as MNIST, the torchhd.datasets module provides an interface to several commonly used datasets in HDC. Such interfaces inherit from PyTorch’s dataset class, ensuring interoperability with other datasets.

Training
To perform the training, we start by defining an encoding. In addition to specifying the basis-hypervectors sets, a core part of learning is the encoding function. In the example below, we use random-hypervectors and level-hypervectors to encode the position and value of each pixel, respectively:

class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
encode = encode.to(device)

num_classes = len(train_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)
Having defined the model, we iterate over the training samples to create the class-vectors:

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = encode(samples)
        model.add(samples_hv, labels)
Testing
With the model trained, we can classify the testing samples by encoding them and comparing them to the class-vectors:

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

with torch.no_grad():
    model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%"


import torch
import torchhd
import numpy as np

# Suppress scientific notation in prints
np.set_printoptions(suppress=True, precision=8)
torch.set_printoptions(sci_mode=False)

print("\nGenerating Random Bipolar Hypervectors...")
hv_1 = torchhd.random(1, 1000, "MAP").squeeze(0)
hv_2 = torchhd.random(1, 1000, "MAP").squeeze(0)

print("\nBefore Applying ensure_vsa_tensor:")
print(f"Mean hv_1: {hv_1.mean().item():.6f}")
print(f"Std hv_1: {hv_1.std().item():.6f}")
print(f"Unique values in hv_1: {torch.unique(hv_1)}")

print("\nApplying torchhd.ensure_vsa_tensor()...")
hv_1_vsa = torchhd.ensure_vsa_tensor(hv_1, "MAP")

print("\nAfter Applying ensure_vsa_tensor:")
print(f"Mean hv_1_vsa: {hv_1_vsa.mean().item():.6f}")
print(f"Std hv_1_vsa: {hv_1_vsa.std().item():.6f}")
print(f"Unique values in hv_1_vsa: {torch.unique(hv_1_vsa)}")

# Test on a bundled vector (values -2, 0, 2)
bundled_hv = torchhd.bundle(hv_1, hv_2)
print("\nBefore ensure_vsa_tensor (Bundled HV):")
print(f"Mean bundled: {bundled_hv.mean().item():.6f}")
print(f"Std bundled: {bundled_hv.std().item():.6f}")
print(f"Unique values in bundled HV: {torch.unique(bundled_hv)}")

bundled_hv_vsa = torchhd.ensure_vsa_tensor(bundled_hv, "MAP")
print("\nAfter ensure_vsa_tensor (Bundled HV):")
print(f"Mean bundled normalized: {bundled_hv_vsa.mean().item():.6f}")
print(f"Std bundled normalized: {bundled_hv_vsa.std().item():.6f}")
print(f"Unique values in bundled HV after ensure_vsa_tensor: {torch.unique(bundled_hv_vsa)}")

# Test on an interpolated HV (which includes -1, 0, 1)
interpolated_hv = 0.6 * hv_1 + 0.4 * hv_2
print("\nBefore ensure_vsa_tensor (Interpolated HV):")
print(f"Mean interpolated: {interpolated_hv.mean().item():.6f}")
print(f"Std interpolated: {interpolated_hv.std().item():.6f}")
print(f"Unique values in interpolated HV: {torch.unique(interpolated_hv)}")

interpolated_hv_vsa = torchhd.ensure_vsa_tensor(interpolated_hv, "MAP")
print("\nAfter ensure_vsa_tensor (Interpolated HV):")
print(f"Mean interpolated normalized: {interpolated_hv_vsa.mean().item():.6f}")
print(f"Std interpolated normalized: {interpolated_hv_vsa.std().item():.6f}")
print(f"Unique values in interpolated HV after ensure_vsa_tensor: {torch.unique(interpolated_hv_vsa)}")
'''
import torch
'''
# Grid size and hypervector dimension
height, width = 480, 640
k_values = [2, 3, 4, 5]  # Testing different k sizes
d = 512  # Hypervector dimension


# Function to compute number of corner hypervectors based on k
def compute_corner_hvs(k):
    num_x_corners = (width // k) + 1
    num_y_corners = (height // k) + 1
    return num_x_corners, num_y_corners, num_x_corners * num_y_corners


# Function to compute the interpolation proportions for a given (x, y) and k
def compute_interpolation_proportions(x, y, k):
    x0, y0 = (x // k) * k, (y // k) * k  # Bottom-left corner
    x1, y1 = x0 + k, y0 + k  # Top-right corner

    cx0, cy0 = x0 // k, y0 // k  # Bottom-left corner index
    cx1, cy1 = cx0 + 1, cy0 + 1  # Top-right corner index

    # Compute distance-based proportions
    alpha_x = (x - x0) / k  # Distance from left
    alpha_y = (y - y0) / k  # Distance from bottom

    # Compute the proportions based on distance
    bottom_left = (1 - alpha_x) * (1 - alpha_y)
    bottom_right = alpha_x * (1 - alpha_y)
    top_left = (1 - alpha_x) * alpha_y
    top_right = alpha_x * alpha_y

    return {
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
        "top_left": top_left,
        "top_right": top_right,
        "corners_used": [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    }


# Test different k values
test_results = []

for k in k_values:
    num_x_corners, num_y_corners, total_corners = compute_corner_hvs(k)

    # Test for some normal and edge-case pixels
    test_pixels = [(0, 0), (1, 1), (k // 2, k // 2), (k - 1, k - 1), (k, k), (width - 1, height - 1),
                   (width // 2, height // 2)]
    pixel_results = {pixel: compute_interpolation_proportions(pixel[0], pixel[1], k) for pixel in test_pixels}

    test_results.append({
        "k": k,
        "num_x_corners": num_x_corners,
        "num_y_corners": num_y_corners,
        "total_corners": total_corners,
        "interpolation_tests": pixel_results
    })

# Display results
'''
'''
import pickle

file_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/test/scissor_200210_150548_0_0_td.dat.pkl"



with open(file_path, "rb") as f:
    data = pickle.load(f)

events = data[0]  # Extract event data

print(f"Data type: {events.dtype}")
print(f"First 5 events:\n{events[:5]}")

import h5py
import numpy as np
import os

H5_SAMPLE = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/train/scissors_right_far_slow_standing_recording_026_2021-09-14_15-51-13.h5"
with h5py.File(H5_SAMPLE, "r") as f:
    event_data = f["data"][:]  # Load event tensor
    print(f"Shape: {event_data.shape}")

    first_bin = event_data[0]  # First time bin
    last_bin = event_data[-1]  # Last time bin

    print("\nFirst Time Bin (T=0):")
    print(first_bin)

    print("\nLast Time Bin (T=T-1):")
    print(last_bin)

    print(f"\nTotal T bins: {event_data.shape[0]}")
'''
import h5py
import numpy as np

file_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/Bin Labeled/VAL BIN LABELED/paper_left_far_slow_sitting_recording_023_2021-09-14_15-28-09.h5"

with h5py.File(file_path, "r") as f:
    event_data = f["data"][:]  # Shape: (T, 2, H, W)


print(f"Shape: {event_data.shape}")
print("Min value:", event_data.min())
print("Max value:", event_data.max())
print("Unique values (if few):", np.unique(event_data)[:20])

print("Sample values from first time bin:")
print(event_data[0]) 

