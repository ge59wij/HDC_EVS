import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

file_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/train/paper_200211_140644_0_0.h5" #input("Enter HDF5 file path: ")
with h5py.File(file_path, "r") as f:
    event_data = f["data"][:]  # Shape: (T, 2, H, W)
    class_id = f["class_id"][()]
T, _, H, W = event_data.shape
print(f"Loaded HDF5 File: {file_path}")
print(f"Shape: {event_data.shape} (T={T}, Height={H}, Width={W})")
def extract_events_from_histogram(histogram_on, histogram_off):
    y_on, x_on = np.where(histogram_on > 0)
    y_off, x_off = np.where(histogram_off > 0)
    intensities_on = histogram_on[y_on, x_on]  # Get the intensity of ON events
    intensities_off = histogram_off[y_off, x_off]  # Get the intensity of OFF events
    return x_on, y_on, intensities_on, x_off, y_off, intensities_off
fig, ax = plt.subplots(figsize=(8, 6))

scatter_on = ax.scatter([], [], s=[], c=[], cmap="Blues", alpha=0.7, label="ON Events")
scatter_off = ax.scatter([], [], s=[], c=[], cmap="Reds", alpha=0.7, label="OFF Events")

plt.colorbar(scatter_on, label="ON Event Density", ax=ax)
plt.colorbar(scatter_off, label="OFF Event Density", ax=ax)

ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()
ax.set_xlabel("X Pixel")
ax.set_ylabel("Y Pixel")
ax.legend()
paused = False
current_frame = 0


def update(frame):
    global current_frame
    if not paused:
        current_frame = frame
    histogram_on = event_data[current_frame, 0, :, :]
    histogram_off = event_data[current_frame, 1, :, :]
    x_on, y_on, intensities_on, x_off, y_off, intensities_off = extract_events_from_histogram(histogram_on,
                                                                                              histogram_off)
    sizes_on = np.clip(intensities_on * 10, 10, 200)
    sizes_off = np.clip(intensities_off * 10, 10, 200)
    scatter_on.set_offsets(np.column_stack((x_on, y_on)))
    scatter_on.set_sizes(sizes_on)
    scatter_on.set_array(intensities_on)  # Color based on intensity
    scatter_off.set_offsets(np.column_stack((x_off, y_off)))
    scatter_off.set_sizes(sizes_off)
    scatter_off.set_array(intensities_off)
    ax.set_title(f"Gesture {class_id} | Bin {current_frame + 1}/{T} | {current_frame * 10} ms elapsed")
    return scatter_on, scatter_off
def toggle_pause(event):
    global paused
    if event.key == " ":
        paused = not paused
def step_frame(event):
    global current_frame
    if paused:
        if event.key == "right":
            current_frame = min(current_frame + 1, T - 1)
        elif event.key == "left":
            current_frame = max(current_frame - 1, 0)
        update(current_frame)
        fig.canvas.draw()
def exit_animation(event):
    if event.key == "escape":
        plt.close(fig)
fig.canvas.mpl_connect("key_press_event", toggle_pause)
fig.canvas.mpl_connect("key_press_event", step_frame)
fig.canvas.mpl_connect("key_press_event", exit_animation)

ani = animation.FuncAnimation(fig, update, frames=T, interval=50, blit=False)
# Manually create legend for ON and OFF events
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='ON Events', markersize=8, markerfacecolor='red', alpha=0.75),
    Line2D([0], [0], marker='o', color='w', label='OFF Events', markersize=8, markerfacecolor='blue', alpha=0.75)
]
ax.legend(handles=legend_elements, loc="upper right", title="Event Type")

plt.show()
