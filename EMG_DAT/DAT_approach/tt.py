import pickle
import torch
import os

dataset_path = "/space/chair-nas/tosy/dattensors_chifoumi/test"
files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.pkl')]

if not files:
    print("No pickle files found!")
    exit()

with open(files[0], "rb") as f:
    events_tensor, label_tensor = pickle.load(f)
print()
print(f"Events Tensor Type: {type(events_tensor)} | Shape: {events_tensor.shape if isinstance(events_tensor, torch.Tensor) else 'N/A'}")
print(f"Label Tensor Type: {type(label_tensor)} | Shape: {label_tensor.shape if isinstance(label_tensor, torch.Tensor) else 'N/A'}")
print(label_tensor)
'''
# Re-import necessary libraries after execution state reset
import pandas as pd

# Given image dimensions
height, width = 480, 640

# Function to compute corners and interpolated positions for a given k
def compute_hvs(k):
    num_rows = (height // k) + 1
    num_cols = (width // k) + 1
    num_corners = num_rows * num_cols
    num_interpolated = (height * width) - num_corners
    return num_corners, num_interpolated

# Create a table for different k values
k_values = list(range(2, 21, 2)) + list(range(25, 101, 5))  # Small steps for small k, larger for big k
data = []

for k in k_values:
    num_corners, num_interpolated = compute_hvs(k)
    data.append([k, num_corners, num_interpolated])

# Convert to a DataFrame for better visualization
df = pd.DataFrame(data, columns=["k", "Corner HVs", "Interpolated HVs"])

# Display the table
import ace_tools as tools
tools.display_dataframe_to_user(name="Corner & Interpolated HVs Table", dataframe=df)
'''
import torchhd

x = torchhd.random(3, 6)
sum_x = torch.sum(x, dim=0)
multiset_x = torchhd.multiset(x)
print("Sum:\n", sum_x)
print("Multiset:\n", multiset_x)
