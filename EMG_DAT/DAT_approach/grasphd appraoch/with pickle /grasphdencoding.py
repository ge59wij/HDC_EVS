import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
import numpy as np

np.set_printoptions(suppress=True, precision=8)


class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time):
        super().__init__(height, width, dims, time_subwindow, k, device, max_time)
        self.time_hv_cache = {}

    '''
    def encode_grasphd(self, events, class_id):
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

    #'''ONE loop appraoch grasp:
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
                H_spatial_accumulated = H_spatial  # Start fresh accumulation

        # Finalize last accumulated subwindow
        if H_spatial_accumulated is not None:
            T_t = self.get_time_hv(current_subwindow * self.time_subwindow)
            H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

        # **Normalize before returning**
        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")

        return H_spatiotemporal
    #'''

    #''' #2 loops: 94.90s/it : 30 samples, around 55 mins, try it tho.
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
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal



    #'''
    '''#test previous, 2 loops time wiese, then this:
    def encode_eventhd(self, events, class_id):
        """Encodes events using EventHD method: binding all spatial encodings per timestamp before binding with time."""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")
    
        H_spatiotemporal = None  # Final encoding
        current_t = None  # Track the current timestamp
        H_spatial_accumulated = None  # Accumulate spatial encoding per timestamp
    
        for t, x, y, polarity in events:
            P_xy = self.get_position_hv(x, y)  # Fetch position HV
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)  # Bind position and illumination
    
            if current_t is None:  # First event => Initialize
                current_t = t
                H_spatial_accumulated = H_spatial
    
            elif t == current_t:  # Same timestamp => Accumulate spatial encoding
                H_spatial_accumulated = torchhd.multiset(torch.stack([H_spatial_accumulated, H_spatial]))
    
            else:  # New timestamp  Finalize previous timestamp and move on
                T_t = self.get_time_hv(current_t)  # Get time HV for previous timestamp
                H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)  # Bind accumulated spatial to time HV
    
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
            torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))
            current_t = t
            H_spatial_accumulated = H_spatial  # Start fresh accumulation for the new timestamp
    
        # Processing the last accumulated timestamp**
        if H_spatial_accumulated is not None:
            T_t = self.get_time_hv(current_t)
            H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))
    
        print(f"Hsample: {H_spatiotemporal}")
        #print(f"Norm before normalization: {torch.norm(H_spatiotemporal)}")
        #H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        #print(f"Norm after normalization: {torch.norm(H_spatiotemporal)}")
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
    
        return H_spatiotemporal

    '''
    def encode_temporalpermutation(self, events, class_id):
        """Encodes events using EventHD method with permutation-based temporal encoding.
        Spatial vector for each event, events are grouped by timestamp t, and spatial encodings are accumulated.
        accumulated spatial encodings permuted based on the number of events in that timestamp, then bound binding with time.
              """
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        H_spatiotemporal = None
        current_t = None  # Track the current timestamp
        H_spatial_accumulated = None  # Accumulate spatial encoding per timestamp
        event_count = 0  # Count how many events are in the same timestamp

        for t, x, y, polarity in events:
            P_xy = self.get_position_hv(x, y)  # Fetch position HV
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)  # Bind position and polarity

            if current_t is None:  # First event
                current_t = t
                H_spatial_accumulated = H_spatial
                event_count = 1

            elif t == current_t:  # Same timestamp => accumulate spatial encoding
                H_spatial_accumulated = torchhd.multiset(torch.stack([H_spatial_accumulated, H_spatial]))
                event_count += 1

            else:  # New timestamp => finalize the previous timestamp and move on
                permuted_H = torchhd.permute(H_spatial_accumulated, event_count)  # identity permutation shift
                H_spatiotemporal = permuted_H if H_spatiotemporal is None else torchhd.bind(H_spatiotemporal,
                                                                                            permuted_H)

                # Reset for the new timestamp
                current_t = t
                H_spatial_accumulated = H_spatial
                event_count = 1  # Reset count

        # last accumulated timestamp
        if H_spatial_accumulated is not None:
            T_perm = torchhd.permute(torchhd.identity(self.dims), event_count)  # Unique shift for each timestamp # by count, or by bin number?
            permuted_H = torchhd.bind(H_spatial_accumulated, T_perm)
            H_spatiotemporal = permuted_H if H_spatiotemporal is None else torchhd.multiset(
                torch.stack([H_spatiotemporal, permuted_H])
            )

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal


    def encode_accumulation(self, events, class_id):
        """Encodes events using interval-based method: one spatial HV per time bin, bound with continuous time encoding.
        Time Binning: Events are grouped into time bins of a fixed size.
        Spatial Encoding per Bin: The spatial encodings of the events in the same time bin are accumulated.
        ignores polarity for now.
        Binding with Time: Each bin’s accumulated spatial encoding is bound with a continuous time encoding, thermometer or .
        """
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        H_gesture = None

        # 1: rocess events in time bins
        for t in range(0, int(self.max_time), int(self.time_subwindow)):
            # Find all unique (x, y) positions in this bin
            active_positions = {(x, y) for e in events if t <= e[0] < t + self.time_subwindow for x, y in
                                [(e[1], e[2])]}
            #e[0]:  timestamp
            #e[1], e[2]:  (x, y)

            if not active_positions:  # If no events happened in this bin, skip
                continue

                # 2 Encode spatial HV for this bin
            H_spatial_bin = None
            for x, y in active_positions:
                P_xy = self.get_position_hv(x, y)  # Spatial encoding ########change back
                H_spatial_bin = P_xy if H_spatial_bin is None else torchhd.multiset(torch.stack([H_spatial_bin, P_xy]))

            # 3: Bind with continuous time encoding (thermometer
            T_t = self.time_continious(t)  # `time_continious()` instead of `get_time_hv()`
            H_temporal_bin = torchhd.bind(H_spatial_bin, T_t)

            # bundling all
            H_gesture = H_temporal_bin if H_gesture is None else torchhd.multiset(
                torch.stack([H_gesture, H_temporal_bin]))

        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_gesture.shape}")
        return H_gesture













    def linear_interpolation(self, events, class_id) : ####doesnt make sense, all interpolated vectors will be identical.
        """Encodes events using EventHD method: binding all spatial encodings per timestamp before binding with time.
        interpolated spatial and temp, linear."""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")
        H_spatiotemporal = None  # Final encoding
        current_t = None  # Track the current timestamp
        H_spatial_accumulated = None  # Accumulate spatial encoding per timestamp

        for t, x, y, polarity in events:
            P_xy = self.weighted_position_hv(x, y)  # Fetch position HV
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)  # Bind position and illumination

            if current_t is None:  # First event => Initialize
                current_t = t
                H_spatial_accumulated = H_spatial

            elif t == current_t:  # Same timestamp => Accumulate spatial encoding
                H_spatial_accumulated = torchhd.multiset(torch.stack([H_spatial_accumulated, H_spatial]))

            else:  # New timestamp =Y Finalize previous timestamp and move on
                T_t = self.get_time_hv(current_t)  # Get time HV for previous timestamp
                H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)  # Bind accumulated spatial to time HV

                # **Accumulate all timestamp-based encodings**
                H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                    torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

                # Reset for new timestamp
                current_t = t
                H_spatial_accumulated = H_spatial  # Start fresh accumulation for the new timestamp

        # Processing the last accumulated timestamp**
        if H_spatial_accumulated is not None:
            T_t = self.get_time_hv(current_t)
            H_temporal_t = torchhd.bind(H_spatial_accumulated, T_t)
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

        print(
            f"[DEBUG] Encoded Hypervector Sample | Class {class_id} | Norm: {torch.norm(H_spatiotemporal.clone().float()):.4f}")
        # print(f"Norm before normalization: {torch.norm(H_spatiotemporal)}")
        # H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        # print(f"Norm after normalization: {torch.norm(H_spatiotemporal)}")
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")

        return H_spatiotemporal  ###
