import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
import numpy as np
np.set_printoptions(suppress=True, precision=8)


class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time):
        super().__init__(height, width, dims, time_subwindow, k, device, max_time)

        self.time_hv_cache = {}

    def encode_grasphd(self, events, class_id):
        """Encodes events using GraspHD method following the correct paper method."""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")
        time_dict = {}
        # **Step 1: Compute Spatial Encoding Per Event**
        for t, x, y, polarity in events:
            P_xy = self.get_position_hv(x, y)  # Fetch position HV (computed on demand)
            I_p = self.H_I_on if polarity == 1 else self.H_I_off
            H_spatial = torchhd.bind(P_xy, I_p)  # XOR operation

            # **Step 2: Group by Timestamp**
            if t not in time_dict:
                time_dict[t] = H_spatial
            else:
                time_dict[t] = torchhd.multiset(torch.stack([time_dict[t], H_spatial]))

        # **Step 3: Bind Time Hypervectors**
        H_spatiotemporal = None
        for t, H_spatial_t in time_dict.items():
            T_t = self.get_time_hv(t)  # Fetch precomputed time HV
            H_temporal_t = torchhd.bind(H_spatial_t, T_t)

            # **Step 4: Bundle Across Time**
            H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else \
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

        # **Normalize Final Hypervector**
        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"Encoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal
