import torch
import torchhd
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.grasphdencoding_seedhvs import *
import numpy as np

np.set_printoptions(suppress=True, precision=8)


class Raw_events_HDEncoder(seedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time, time_method):
        super().__init__(height, width, dims, time_subwindow, k, device, max_time, time_method)
        self.time_hv_cache = {}



    # 1 loop
    def encode_eventhd(self, events, class_id):
        """Encodes events using EventHD or STEMHD method (controlled via `time_interpolation_method`).
        - **Spatial encoding**: Interpolates position HVs.
        - **Temporal encoding**:
        - `event_hd_timepermutation`: Uses permutation-based encoding (shifts base HV).
        - `event_hd_timeinterpolation`: Uses weighted sum per element for time interpolation.
        - `stem_hd`: Uses concatenation-based interpolation (one HV per bin).
        """
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        H_spatiotemporal = None  #  Final encoding
        temporal_dict = {}  # Dictionary to accumulate spatial encodings per timestamp

        # **Step 1: Compute Spatial Encoding Per Event and Accumulate per Time**
        for t, x, y, polarity in events:
            P_xy = self.get_position_hv(x, y)  # Fetch position hypervector
            I_p = self.H_I_on if polarity == 1 else self.H_I_off  # Fetch polarity hypervector
            H_spatial = torchhd.bind(P_xy, I_p)  # Bind position and polarity

            if t not in temporal_dict:
                temporal_dict[t] = H_spatial  # First event in this timestamp
            else:
                temporal_dict[t] = torchhd.bundle(temporal_dict[t],
                                                  H_spatial)  # Bundle new event into accumulated spatial encoding

        # **Step 2: Bind Each Timestamp’s Accumulated Spatial Encoding with its Time Hypervector**
        for t, H_spatial_accumulated in temporal_dict.items():
            T_t = self.get_time_hv(t)  # Get time hypervector for this timestamp

            if self.time_interpolation_method == "stem_hd":
                # STEMHD uses concatenation-based interpolation for temp
                H_timebin = torch.cat((H_spatial_accumulated[:self.dims // 2], T_t[-self.dims // 2:]), dim=0)
            else:
                # EventHD & GraspHD use weighted sum per element
                H_timebin = torchhd.bind(H_spatial_accumulated, T_t)

            # **Step 3: Accumulate Time-Encoded Hypervectors Across All Time Bins**
            H_spatiotemporal = H_timebin if H_spatiotemporal is None else torchhd.bundle(H_spatiotemporal, H_timebin)

        if H_spatiotemporal is not None:
            H_spatiotemporal = torchhd.normalize(H_spatiotemporal)

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal











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
