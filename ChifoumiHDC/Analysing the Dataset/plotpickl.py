import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
'''
plot pickle files, 10ms,
Check linechart.
/space/chair-nas/tosy/preprocessed_dat_chifoumi/picked_samples/
'''

file_path = input("file path: ")
with open(file_path, "rb") as f:
    data = pickle.load(f)

events = data[0]
label = data[1]

timestamps = events["t"]
x_coords = events["x"]
y_coords = events["y"]
polarities = events["p"]

bin_size = 10000
max_time = np.max(timestamps)  #in µs
gesture_duration_sec = max_time / 1e6  # to s
bins = np.arange(0, max_time, bin_size)

frames = []
for i in range(len(bins) - 1):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i + 1])
    frames.append((x_coords[mask], y_coords[mask], polarities[mask]))

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, np.max(x_coords))
ax.set_ylim(0, np.max(y_coords))
ax.invert_yaxis()
ax.set_xlabel("X Pixel")
ax.set_ylabel("Y Pixel")

scatter_on = ax.scatter([], [], s=1, color="blue", label="ON Events")
scatter_off = ax.scatter([], [], s=1, color="red", label="OFF Events")
ax.legend()

paused = False
current_frame = 0

def update(frame):
    global current_frame
    if not paused:
        current_frame = frame
    x, y, p = frames[current_frame]
    scatter_on.set_offsets(np.column_stack((x[p == 1], y[p == 1])))
    scatter_off.set_offsets(np.column_stack((x[p == 0], y[p == 0])))
    current_time_ms = (current_frame + 1) * (bin_size / 1000)  #µs to ms
    ax.set_title(
        f"Gesture {label} | Bin {current_frame+1}/{len(frames)} | {current_time_ms:.1f}ms elapsed\n"
        f"Bin Size: {bin_size / 1000}ms | Total Duration: {gesture_duration_sec:.2f}s"
    )
    return scatter_on, scatter_off
def toggle_pause(event):
    global paused
    if event.key == " ":
        paused = not paused
def step_frame(event):
    global current_frame, ani
    if paused:
        if event.key == "right":
            current_frame = min(current_frame + 1, len(frames) - 1)
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

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
plt.show()
