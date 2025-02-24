'''
Prints the top N files with:
The least number of events.
The most number of events.
Filters by gesture class (if specified)
'''


import os
import pickle
import numpy as np
from glob import glob


def find_top_and_bottom_event_files(dataset_path, top_n, gesture_class=None):
    event_counts = []
    pkl_files = glob(os.path.join(dataset_path, "**/*.pkl"), recursive=True)
    print(f"Found {len(pkl_files)} .pkl files. Processing...")
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                events, label = pickle.load(f)
            if gesture_class is not None and label != gesture_class:
                continue
            total_events = len(events["t"])
            last_timestamp_sec = events["t"].max() / 1e6  #µs to s
            event_counts.append((pkl_file, total_events, last_timestamp_sec, label))
        except Exception as e:
            print(f"Error processing {pkl_file}: {e}")
    if not event_counts:
        print("\nNo valid event data found.")
        return

    event_counts_sorted = sorted(event_counts, key=lambda x: x[1])
    least_events = event_counts_sorted[:top_n]
    most_events = event_counts_sorted[-top_n:]

    class_info = f"for class {gesture_class}" if gesture_class is not None else "for all classes"
    print(f"\n=== TOP {top_n} FILES WITH LEAST EVENTS {class_info} ===")
    for i, (file, count, last_t_sec, label) in enumerate(least_events, 1):
        print(f"{i}. {file} -> {count} events, Length: {last_t_sec:.3f} s, Label: {label}")

    print("\n" + "=" * 80 + "\n")

    print(f"\n=== TOP {top_n} FILES WITH MOST EVENTS {class_info} ===")
    for i, (file, count, last_t_sec, label) in enumerate(reversed(most_events), 1):  # highest first
        print(f"{i}. {file} -> {count} events, Length: {last_t_sec:.3f} s, Label: {label}")


DATASET_PATH = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/train"
GESTURE_CLASS = 0  #None for all
TOP_N = 30
find_top_and_bottom_event_files(DATASET_PATH, top_n=TOP_N, gesture_class=GESTURE_CLASS)
