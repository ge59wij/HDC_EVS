import tonic
import numpy as np
import tonic.transforms as transforms
import matplotlib.pyplot as plt
import random

# Load NMNIST dataset
dataset = tonic.datasets.NMNIST(save_to="/space/chair-nas/tosy/data", train=True)
sensor_size = tonic.datasets.NMNIST.sensor_size  # Sensor resolution (34, 34)

# Randomly select an index
random_index = random.randint(0, len(dataset) - 1)
events, label = dataset[random_index]

# Extract event data
timestamps = events["t"]
x_coords = events["x"]
y_coords = events["y"]
polarities = events["p"]

# Define binning parameters
bin_size = 10000  # 10ms bins
max_time = np.max(timestamps)
bins = np.arange(0, max_time, bin_size)

# Store frames for plotting
frames = []
for i in range(len(bins) - 1):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i + 1])
    frames.append((x_coords[mask], y_coords[mask], polarities[mask]))

# Plot the frames
num_frames = min(len(frames), 4)
fig, axes = plt.subplots(1, num_frames, figsize=(12, 5))

for i in range(num_frames):
    ax = axes[i]
    x, y, p = frames[i]
    ax.scatter(x[p == 1], y[p == 1], s=15, color="blue", label="ON Events")
    ax.scatter(x[p == 0], y[p == 0], s=15, color="red", label="OFF Events")

    ax.set_xlim(0, sensor_size[1])  # Width
    ax.set_ylim(0, sensor_size[0])  # Height
    ax.invert_yaxis()
    ax.set_title(f"Bin {i+1}/{num_frames} ({bins[i]/1000:.1f}ms)")
    ax.axis("off")

fig.suptitle(f"NMNIST Sample (Class {label})", fontsize=18)
plt.tight_layout()
plt.show()
