import torch
import torchhd
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.grasphdencoding_seedhvs import seedEncoder
import numpy as np


from collections import defaultdict

np.set_printoptions(suppress=True, precision=8)


class Raw_events_HDEncoder(seedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS):
        super().__init__(height, width, dims, k, time_subwindow , device, max_time, time_method,WINDOW_SIZE_MS, OVERLAP_MS )
        self.time_hv_cache = {}

    def encode_eventhd(self, events, class_id):
        """Encodes events using different time encoding methods:
        - **event_hd_timepermutation** → Applies permutation-based encoding (shifts spatial hypervector itself).
        - **event_hd_timeinterpolation** → Uses weighted sum per element for time interpolation.
        - **stem_hd** → Uses concatenation-based interpolation (one HV per bin).
        """
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        H_spatiotemporal = None  # Final encoding
        #temporal_dict = {}  # Stores spatial hypervectors per timestamp
        temporal_dict = defaultdict(list)
        TIME_BIN_SIZE = self.time_subwindow  # used directly for eventhd permutation


        # **Step 1: Compute Spatial Encoding Per Event and Accumulate per Time Bin**
        for t, x, y, polarity in events:
            P_xy = self.get_position_hv(x, y)  # Fetch position hypervector
            I_p = self.H_I_on if polarity == 1 else self.H_I_off  # Fetch polarity hypervector
            H_spatial = torchhd.bind(P_xy, I_p)  # Bind position and polarity
            #torchhd.normalize(H_spatial)
            time_bin = int((t // TIME_BIN_SIZE) * TIME_BIN_SIZE)  # Ensure integer bin key

            # Append spatial encodings for this bin (don't bundle yet)
            temporal_dict[time_bin].append(H_spatial)

        # **Step 2: Process Time Bins Based on Encoding Method**
        sorted_time_bins = sorted(temporal_dict.keys())  # Ensure bins are in order
        permuted_time_hvs = []
        bundled_time_hvs = []

        for index, time_bin in enumerate(sorted_time_bins):
            spatial_hvs = torch.stack(temporal_dict[time_bin])  # Stack all spatial hypervectors
            SE_t = torchhd.normalize(torchhd.multibundle(spatial_hvs))  # Bundle all spatial HVs for this bin

            if self.time_interpolation_method == "event_hd_timepermutation":
                # **Permutation Encoding: Shift spatial encoding by bin index**
                H_timebin = torchhd.permute(SE_t, shifts=index)
                permuted_time_hvs.append(H_timebin)

            elif self.time_interpolation_method == "event_hd_timeinterpolation":
                # **Interpolation Encoding: Weighted sum per element**
                T_t = self.get_time_hv(time_bin)  # Fetch time hypervector
                H_timebin = torchhd.bind(SE_t, T_t)
                bundled_time_hvs.append(H_timebin)

            elif self.time_interpolation_method == "stem_hd":
                # **STEMHD Encoding: Concatenation-based**
                T_t = self.get_time_hv(time_bin)  # Fetch time hypervector
                H_timebin = torch.cat((SE_t[:self.dims // 2], T_t[-self.dims // 2:]), dim=0)
                bundled_time_hvs.append(H_timebin)

        # **Step 3: Final Temporal Bundling**
        if self.time_interpolation_method == "event_hd_timepermutation":
            if permuted_time_hvs:
                H_spatiotemporal = torchhd.normalize(torchhd.multibundle(torch.stack(permuted_time_hvs)))
            else:
                H_spatiotemporal = torchhd.empty(1, self.dims).squeeze(0)

        elif self.time_interpolation_method in ["event_hd_timeinterpolation", "stem_hd"]:
            if bundled_time_hvs:
                H_spatiotemporal = torchhd.normalize(torchhd.multibundle(torch.stack(bundled_time_hvs)))
            else:
                H_spatiotemporal = torchhd.empty(1, self.dims).squeeze(0)

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal

    # 1 loop
    def process_windows(self, full_events, class_id):
        """Splits a full event sequence into sliding windows and encodes each separately."""
        event_hvs = []  # Stores HVs per window
        total_events = len(full_events)

        # Get first and last timestamp
        first_t = full_events[0][0] if total_events > 0 else None
        last_t = full_events[-1][0] if total_events > 0 else None
        total_duration = last_t - first_t if total_events > 0 else 0

        # Compute expected number of windows BEFORE processing
        if total_duration >= self.WINDOW_SIZE_MS:
            expected_windows = (total_duration - self.OVERLAP_MS) // (self.WINDOW_SIZE_MS - self.OVERLAP_MS) + 1
        else:
            expected_windows = 1 if total_events >= 5 else 0  # Only if enough events exist

        print(f"\n[INFO] Encoding Sample | Class: {class_id} | Total Events: {total_events}")
        print(f"      - First Timestamp: {first_t}, Last Timestamp: {last_t}")
        print(f"      - Total Duration: {total_duration} ms")
        print(f"      - Expected Windows: {expected_windows}")

        # Now process windows as before
        start_time = first_t if total_events > 0 else 0  # Ensure we start at the actual first event
        end_time = start_time + self.WINDOW_SIZE_MS
        window_index = 0
        window_events = []
        skipped_windows = 0

        for t, x, y, polarity in full_events:
            if t >= end_time:  # Window finished
                if window_events:
                    adjusted_events = [(t - start_time, x, y, p) for t, x, y, p in window_events]

                    if len(adjusted_events) < 5:  # Skip tiny windows
                        print(f"[DEBUG] Skipping small window {window_index} (Only {len(adjusted_events)} events)")
                        skipped_windows += 1
                    else:
                        first_ts, last_ts = adjusted_events[0][0], adjusted_events[-1][0]
                        print(f"[DEBUG] Window {window_index}: Start={start_time}, End={end_time}, "
                              f"Total Events={len(adjusted_events)}, First Timestamp={first_ts}, Last Timestamp={last_ts}")

                        window_hv = self.encode_eventhd(adjusted_events, class_id)
                        if window_hv is not None:
                            event_hvs.append(window_hv)

                    window_index += 1

                start_time += (self.WINDOW_SIZE_MS - self.OVERLAP_MS)
                end_time = start_time + self.WINDOW_SIZE_MS
                window_events = []

            window_events.append((t, x, y, polarity))  # Collect events for the current window

        # Final window processing
        if window_events:
            adjusted_events = [(t - start_time, x, y, p) for t, x, y, p in window_events]

            if len(adjusted_events) < 5:
                print(f"[DEBUG] Skipping small final window {window_index} (Only {len(adjusted_events)} events)")
                skipped_windows += 1
            else:
                first_ts, last_ts = adjusted_events[0][0], adjusted_events[-1][0]
                print(f"[DEBUG] Final Window {window_index}: Start={start_time}, End={end_time}, "
                      f"Events={len(adjusted_events)}, First Timestamp={first_ts}, Last Timestamp={last_ts}")

                if len(adjusted_events) < self.WINDOW_SIZE_MS:
                    print(f"[DEBUG] Final window is shorter than expected: {len(adjusted_events)} events.")

                window_hv = self.encode_eventhd(adjusted_events, class_id)
                if window_hv is not None:
                    event_hvs.append(window_hv)

        print(f"[INFO] Sample {class_id} - Created: {len(event_hvs)} windows | Skipped: {skipped_windows}\n")

        return event_hvs

    ####################graveyard##############

    '''
    def encode_temporalpermutation(self, events, class_id):
        """Encodes events using permutation-based temporal encoding.
        - Groups timestamps into bins of `bin_size`
        - Accumulates spatial encodings per bin
        - The accumulated encoding is permuted based on the event count before binding with time.
        """
        bin_size = 100  # Adjust as needed
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        H_spatiotemporal = None  # Final encoding
        current_bin = None  # Track the current time bin
        H_spatial_accumulated = None  # Accumulate spatial encoding per bin
        event_count = 0  # Count how many events are in the same bin

        for t, x, y, polarity in events:
            bin_index = t // bin_size  # Compute bin index
            P_xy = self.get_position_hv(x, y)
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)

            if current_bin is None:  # First event
                current_bin = bin_index
                H_spatial_accumulated = H_spatial
                event_count = 1

            elif bin_index == current_bin:  # Same time bin => accumulate
                H_spatial_accumulated = torchhd.bundle(H_spatial_accumulated, H_spatial)
                event_count += 1

            else:  # New bin => finalize previous bin and move on
                # Ensure accumulated spatial encoding has correct shape
                if H_spatial_accumulated.shape[0] != self.dims:
                    H_spatial_accumulated = H_spatial_accumulated.squeeze(0)

                T_perm = torchhd.permute(self.get_time_hv(current_bin * bin_size), shifts=event_count)
                permuted_H = torchhd.bind(H_spatial_accumulated, T_perm)

                if H_spatiotemporal is None:
                    H_spatiotemporal = permuted_H
                else:
                    if H_spatiotemporal.shape[0] != self.dims:
                        H_spatiotemporal = H_spatiotemporal.squeeze(0)
                    H_spatiotemporal = torchhd.bundle(H_spatiotemporal, permuted_H)

                # Reset for new bin
                current_bin = bin_index
                H_spatial_accumulated = H_spatial
                event_count = 1

        # Final bin processing
        if H_spatial_accumulated is not None:
            if H_spatial_accumulated.shape[0] != self.dims:
                H_spatial_accumulated = H_spatial_accumulated.squeeze(0)

            T_perm = torchhd.permute(self.get_time_hv(current_bin * bin_size), shifts=event_count)
            permuted_H = torchhd.bind(H_spatial_accumulated, T_perm)

            if H_spatiotemporal is None:
                H_spatiotemporal = permuted_H
            else:
                if H_spatiotemporal.shape[0] != self.dims:
                    H_spatiotemporal = H_spatiotemporal.squeeze(0)
                H_spatiotemporal = torchhd.bundle(H_spatiotemporal, permuted_H)

        print(f"\nEncoding Complete | Class: {class_id} | Output Shape: {H_spatiotemporal.shape}")
        return H_spatiotemporal

    def encode_accumulation(self, events, class_id):  # todo: include polarity. #todo: fix binning, more fine?
        """
        Encodes events using interval-based method: one spatial HV per time bin, bound with continuous time encoding
        Time Binning: Events are grouped into time bins of a fixed size.
        Spatial Encoding per Bin: The spatial encodings of the events in the same time bin are accumulated.
        ignores polarity for now.
        Binding with Time: Each bin’s accumulated spatial encoding is bound with a continuous time encoding, thermometer or .
        """
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")
        H_gesture = None
        # 1: process events in time bins
        for t in range(0, int(self.max_time), int(self.time_subwindow)):
            # Find all unique (x, y) positions in this bin
            active_positions = {(x, y) for e in events if t <= e[0] < t + self.time_subwindow for x, y in
                                [(e[1], e[2])]}
            # e[0]:  timestamp
            # e[1], e[2]:  (x, y)
            if not active_positions:  # If no events happened in this bin, skip
                continue
                # 2 Encode spatial HV for this bin
            H_spatial_bin = None
            for x, y in active_positions:
                P_xy = self.get_position_hv(x, y)  # Spatial encoding ########change back
                H_spatial_bin = P_xy if H_spatial_bin is None else torchhd.bind(H_spatial_bin, P_xy)
            # 3: Bind with continuous time encoding (thermometer
            T_t = self.time_continious(t)  # `time_continious()` instead of `get_time_hv()`
            H_temporal_bin = torchhd.bind(H_spatial_bin, T_t)
            # bundling all
            H_gesture = H_temporal_bin if H_gesture is None else torchhd.multiset(
                torch.stack([H_gesture, H_temporal_bin]))
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_gesture.shape}")
        return H_gesture

    def encode_accumulation_weight(self, events, class_id):
        """Encodes events using interval-based method: one spatial HV per time bin, bound with continuous time encoding.
        - Events are grouped into time bins of a fixed size (self.time_subwindow)
        - Each (x, y) position tracks how many times it appears in a bin
        - More frequent events contribute more strongly by bundling spatial hvs multiple times
        - The final spatial encoding for each bin is bound with a continuous time encoding (thermometer/permutation)
        """
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")
        H_gesture = None
        time_bins = {}  # Dictionary to store event counts per bin
        # Step 1: Count occurrences of each (x, y) position in each time bin
        for t, x, y, _ in events:
            bin_index = t // self.time_subwindow  # Find which bin this timestamp belongs to
            if bin_index not in time_bins:
                time_bins[bin_index] = {}
            if (x, y) not in time_bins[bin_index]:
                time_bins[bin_index][(x, y)] = 0
            time_bins[bin_index][(x, y)] += 1  # Increase count for this (x, y) in the bin
        # Step 2: Encode each bin separately
        for bin_index, active_positions in time_bins.items():
            H_spatial_bin = None
            for (x, y), count in active_positions.items():  # Now count-aware
                P_xy = self.get_position_hv(x, y)  # Get spatial encoding

                # Reinforce contribution by bundling multiple times based on count
                for _ in range(count):
                    H_spatial_bin = P_xy if H_spatial_bin is None else torchhd.bundle(H_spatial_bin, P_xy)
            if H_spatial_bin is None:
                continue  # Skip empty bins (shouldn't happen)
            # Step 3: Bind with continuous time encoding (thermometer/permutation)
            T_t = self.time_continious(bin_index * self.time_subwindow)  # Use bin center time
            H_temporal_bin = torchhd.bind(H_spatial_bin, T_t)
            # Step 4: Accumulate all bins
            H_gesture = H_temporal_bin if H_gesture is None else torchhd.bundle(H_gesture, H_temporal_bin)

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_gesture.shape}")
        return H_gesture


    def encode_grasphd(self, events, class_id):   ## 2 for loops!
        """Encodes events using GraspHD method, added subwindow-based accumulation following the paper method: #1 interpolation between anchor TimeHVS"""  
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}") 

        subwindow_dict = {}     # for accumulated spatial vectors

        # **Step 1: Compute Spatial Encoding Per Event then GROUP per subwindow.**
        for t, x, y, polarity in events:   #try to make this and next one, one loop, since timestamps are in order,
            # we will encode all spatial ones in a window, when first t belonds to another window, we can maybe
            #already bind with the corresponding T and move, at the end of the loop, multiset all of them. implement laetr.
            subwindow_index = t // self.time_subwindow

            P_xy = self.get_position_hv(x, y)  # Fetch position HV (computed on demand)
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)  # XOR operation!

            # **Accumulate within the same subwindow**
            if subwindow_index not in subwindow_dict:
                subwindow_dict[subwindow_index] = H_spatial
            else:
                subwindow_dict[subwindow_index] = torchhd.multiset(
                    torch.stack([subwindow_dict[subwindow_index], H_spatial])
                )

        # **Step 2: Bind Subwindow Accumulated HVs to Time HVs**
        H_spatiotemporal = None
        for subwindow_index, H_spatial_accumulated in subwindow_dict.items():
            T_t = self.get_time_hv(subwindow_index * self.time_subwindow)  # Fetch time HV for this subwindow
            H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)

            # **Step 3: Bundle Across Subwindows**
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))  #element wise sum , vectors are -1,1, sum might be 0, issue? idk

        # **Normalize Final Hypervector**
        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"\n\nSample Encoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal
    '''
    ''' 
    #2 loops: 94.90s/it : 30 samples, around 55 mins, try it tho.
    def encode_eventhd(self, events, class_id):  # weighted linear time interpolation EVENTHD, same as grasp mostly, but each t gets T.
        """Encodes events using GraspHD/EVenthd method, no subwindow grouping, each t interpolated between anchor TimeHVS"""
        temporal_dict = {}
        # Spatial: per event & accumulated per t**
        for t, x, y, polarity in events:
            P_xy = self.get_position_hv(x, y)
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)

            if t not in temporal_dict:
                temporal_dict[t] = H_spatial
            else:
                temporal_dict[t] = torchhd.multiset(
                    torch.stack([temporal_dict[t], H_spatial]))  # Bundle events in same timestamp

        H_spatiotemporal = None #init
        for t, H_spatial_accumulated in temporal_dict.items():
            T_t = self.get_time_hv(t)
            H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)
            #Bundle all timestamp-bound HVs together**
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

        print(f"Hsample: {H_spatiotemporal}")
        print(f"Norm before normalization: {torch.norm(H_spatiotemporal)}")
        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"Norm after normalization: {torch.norm(H_spatiotemporal)}")
        print(f"Encoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}\n")
        return H_spatiotemporal

   def encode_grasphd(self, events, class_id):
        """Encodes events using GraspHD method, added subwindow-based accumulation following the paper method: #1 interpolation between anchor TimeHVS"""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        H_spatiotemporal = None  # Final encoding
        current_subwindow = None  # Track the current subwindow
        H_spatial_accumulated = None  # Running sum of spatial encodings per subwindow

        for t, x, y, polarity in events:
            subwindow_index = t // self.time_subwindow

            P_xy = self.get_position_hv(x, y)  # Fetch position HV
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)

            if current_subwindow is None:  # First event
                current_subwindow = subwindow_index
                H_spatial_accumulated = H_spatial

            elif subwindow_index == current_subwindow:  # Same subwindow → accumulate
                H_spatial_accumulated = torchhd.multiset(torch.stack([H_spatial_accumulated, H_spatial]))
            else:  # New subwindow detected → finalize previous subwindow
                T_t = self.get_time_hv(current_subwindow * self.time_subwindow)  # Get time HV for previous subwindow
                H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)  # Bind spatial to time
                H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                    torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))
                # Reset for new subwindow
                current_subwindow = subwindow_index
                H_spatial_accumulated = H_spatial  # Start new accumulation
        # Finalize last accumulated subwindow
        if H_spatial_accumulated is not None:
            T_t = self.get_time_hv(current_subwindow * self.time_subwindow)
            H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))
        # **Normalize before returning**
        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")

        return H_spatiotemporal '''