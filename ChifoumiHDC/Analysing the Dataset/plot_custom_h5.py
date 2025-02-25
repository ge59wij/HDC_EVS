import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/val/scissors_left_far_slow_standing_recording_017_2021-09-14_14-52-41.h5

file_path = input("Enter HDF5 file path: ")
with h5py.File(file_path, "r") as f:
    event_data = f["data"][:]  # Shape: (T, 2, 120, 160)
    class_id = f["class_id"][()]

T, _, H, W = event_data.shape
print(f"Loaded HDF5 File: {file_path}")
print(f"Shape: {event_data.shape} (T={T}, Height={H}, Width={W})")

def extract_events_from_histogram(histogram_on, histogram_off):
    y_on, x_on = np.where(histogram_on > 0)  #on
    y_off, x_off = np.where(histogram_off > 0)
    return x_on, y_on, x_off, y_off

fig, ax = plt.subplots(figsize=(8, 6))
scatter_on = ax.scatter([], [], s=8, color="blue", label="ON Events")
scatter_off = ax.scatter([], [], s=8, color="red", label="OFF Events")

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

    x_on, y_on, x_off, y_off = extract_events_from_histogram(histogram_on, histogram_off)

    scatter_on.set_offsets(np.column_stack((x_on, y_on)))
    scatter_off.set_offsets(np.column_stack((x_off, y_off)))

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

ani = animation.FuncAnimation(fig, update, frames=T, interval=10, blit=False)
plt.show()
