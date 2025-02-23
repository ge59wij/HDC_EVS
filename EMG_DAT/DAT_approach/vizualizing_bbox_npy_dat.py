import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from metavision_core.event_io import EventsIterator

# Paths
base_path = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT/test/"
file_name = "rock_200212_135956_0_0"
dat_file_path = os.path.join(base_path, file_name + "_td.dat")  # Could also be "_td.dat"
bbox_file_path = os.path.join(base_path, file_name + "_bbox.npy")

# Load bounding box for filtering
bbox_data = np.load(bbox_file_path)
start_time = int(bbox_data[0][0])  # First row, first element
end_time = int(bbox_data[-1][0])  # Last row, first element
gesture_label = int(bbox_data[0][5])  # Class ID

print(f"\n Bounding Box Data first and last row for {file_name}:")
print(bbox_data[0], "\n")
print(bbox_data[-1], "\n")

print(f"\n Loaded Bounding Box Data for {file_name}")
print(f"   → Start Time: {start_time} µs")
print(f"   → End Time: {end_time} µs")
print(f"   → Gesture Label: {gesture_label}\n")

# Load all events
original_events = []
filtered_events = []

iterator = EventsIterator(input_path=dat_file_path, mode="delta_t", delta_t=1000)

for events in iterator:
    original_events.append(events)  # Store raw events
    relevant_events = events[(events["t"] >= start_time) & (events["t"] <= end_time)]
    filtered_events.append(relevant_events)  # Store filtered events

print(f"Total original events: {sum([len(e) for e in original_events])}")
print(f"Total filtered events: {sum([len(e) for e in filtered_events])}")

print("\nLets PLOT")

# Setup visualization with two plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].set_title("Original Events")
axes[1].set_title("Filtered Events")

for ax in axes:
    ax.set_xlim(0, 640)  # Assuming width=640
    ax.set_ylim(0, 480)  # Assuming height=480
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

scatter1 = axes[0].scatter([], [], s=1, label="Original")
scatter2 = axes[1].scatter([], [], s=1, label="Filtered")

# Function to update each frame
def update(frame_idx):
    if frame_idx >= len(original_events) or frame_idx >= len(filtered_events):
        return scatter1, scatter2

    # Get event frames
    original_frame = original_events[frame_idx]
    filtered_frame = filtered_events[frame_idx]

    # Extract X, Y, and polarity for color mapping
    x1, y1, p1 = original_frame["x"], original_frame["y"], original_frame["p"]
    x2, y2, p2 = filtered_frame["x"], filtered_frame["y"], filtered_frame["p"]

    # Map polarity to color
    colors1 = np.where(p1 == 1, "deepskyblue", "darkblue")  # Light blue (ON) / Dark blue (OFF)
    colors2 = np.where(p2 == 1, "deepskyblue", "darkblue")

    # Update scatter plots
    scatter1.set_offsets(np.c_[x1, y1])
    scatter1.set_color(colors1)

    scatter2.set_offsets(np.c_[x2, y2])
    scatter2.set_color(colors2)

    return scatter1, scatter2


anim = FuncAnimation(fig, update, frames=min(len(original_events), len(filtered_events)), interval=200)

plt.tight_layout()
plt.show()

# Saving Animation
output_dir = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT/animations"
os.makedirs(output_dir, exist_ok=True)

output_mp4 = os.path.join(output_dir, "event_comparison_with_polarity.mp4")
output_gif = os.path.join(output_dir, "event_comparison_with_polarity.gif")
anim.save(output_mp4, writer="ffmpeg", fps=5)
anim.save(output_gif, writer="pillow", fps=5)

print(f"Saved animation as {output_mp4} and {output_gif}")
