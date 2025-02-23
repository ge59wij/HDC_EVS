from metavision_core.event_io import EventsIterator

# DAT FORMAT (x-coordinate, y-coordinate, polarity, timestamp)

'''
(  4, 472, 0,  31694)
( 95, 471, 0,  90095)
( 29, 404, 0, 138876)
(205, 206, 0, 190742)
Which means first 3 events are:
(4, 472, 0, 31694) → Event at (4,472), polarity OFF, time 31.694 ms
(95, 471, 0, 90095) → Event at (95,471), polarity OFF, time 90.095 ms
(29, 404, 1, 138876) → Event at (29,404), polarity ON, time 138.876 ms
'''

# Path to Prophesee .dat file
dat_file_path = r'/space/chair-nas/tosy/Gen3_Chifoumi_DAT/train/paper_recording_only_paper2__2021-09-14_13-22-54_cd.dat'
iterator = EventsIterator(input_path=dat_file_path, mode="n_events", n_events=999999999)
for events in iterator:
    print(f"processed {len(events)} events in one go hihi!")
    print(events[:10])
    break

print(events.dtype)
iterator = EventsIterator(input_path=dat_file_path, mode="delta_t", delta_t=50000)

#for events in iterator:
    #print(f" Processing {len(events)} events within 5000ms time window")
    #print(events[:1])  # Just print first 10

'''
import os
import pickle
import numpy as np

import os
import pickle
import numpy as np


def check_pickle(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    print(f"Checking: {file_path}")

    # Check if file is empty before loading
    if os.stat(file_path).st_size == 0:
        print("Error: Pickle file is empty!")
        return

    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Check structure
        if not isinstance(data, tuple) or len(data) != 2:
            print("Error: Unexpected pickle format")
            return

        events, class_id = data
        print(f"Class ID: {class_id}")
        print(
            f"Type of events: {type(events)} | Shape: {events.shape if hasattr(events, 'shape') else 'Unknown'} | Dtype: {events.dtype if hasattr(events, 'dtype') else 'Unknown'}")
        formatted_events = " ".join([f"({t:.1f}, {x}, {y}, {p})" for t, x, y, p in events[:3]])
        print(f"First 3 events: {formatted_events}")
    except EOFError:
        print("Error: Ran out of input - File might be corrupted.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    file_path = input("Enter the path to the pickle file: ").strip()
    check_pickle(file_path)
'''