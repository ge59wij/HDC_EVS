import tonic
import numpy as np
import tonic.transforms as transforms
import matplotlib.pyplot as plt


# Load NMNIST from your local path
dataset = tonic.datasets.NMNIST(save_to="/space/chair-nas/tosy/data", train=True)

# Fetch one sample
events, target = dataset[8000]

sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=3)
frames = frame_transform(events)
print(sensor_size)

fig, axes = plt.subplots(1, len(frames))
for axis, frame in zip(axes, frames):
    axis.imshow(frame[1] - frame[0])
    axis.axis("off")
plt.tight_layout()
#plt.show()



# Fetch one sample

# Print length
for i in range(5):  # Check first 5 samples
    events, target = dataset[i]
    print(f"Sample {i} (Class {target}) has {len(events)} events.")
sample_lengths = [len(dataset[i][0]) for i in range(len(dataset))]

print(f"Total Samples: {len(sample_lengths)}")
print(f"Min Events: {min(sample_lengths)}")
print(f"Max Events: {max(sample_lengths)}")
print(f"Average Events: {sum(sample_lengths) / len(sample_lengths):.2f}")
# Check time duration for first 5 samples
for i in range(100):
    events, target = dataset[i]

    timestamps = events["t"]  # Extract timestamps
    min_time = timestamps.min()
    max_time = timestamps.max()
    duration = max_time - min_time  # Total duration of the sample

    print(f"Sample {i} (Class {target}):")
    print(f"  - Start Time: {min_time} µs")
    print(f"  - End Time: {max_time} µs")
    print(f"  - Duration: {duration} µs\n")