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
dat_file_path = r'/space/chair-nas/tosy/Gen3_Chifoumi_DAT/train/paper_200210_144211_0_0_td.dat'
iterator = EventsIterator(input_path=dat_file_path, mode="n_events", n_events=999999999
                    )
for events in iterator:
    #print(f"processed {len(events)} events in one go hihi!")
    #print(events[:10])
    break
#print(events.dtype)
iterator = EventsIterator(input_path=dat_file_path, mode="delta_t", delta_t=50000)

for events in iterator:
    print(f" Processing {len(events)} events within 5000ms time window")
    print(events[:1])  # Just print first 10
