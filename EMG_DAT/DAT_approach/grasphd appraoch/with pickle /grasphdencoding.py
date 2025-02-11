import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
import numpy as np
np.set_printoptions(suppress=True, precision=8)


class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device):
        super().__init__(height, width, dims, time_subwindow, k=k, device=device)
        self.time_hv_cache = {}

    def encode_grasphd(self, events, class_id):
        """Encodes events using GraspHD method (spatial first, then temporal binding)."""


        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        # **Find the latest timestamp **
        last_timestamp = max(event[0] for event in events)
        self._update_time_hvs(last_timestamp)  # Ensure we have enough time hypervectors

        H_spatiotemporal = None
        time_dict = {}

        # **Step 1: Compute Spatial Encoding per Timestamp**
        for event in events:
            #print(event)  # Debugging

            t, x, y, polarity = event
            P_xy = self.get_position_hv(x, y)
            I_p = self.H_I_plus if polarity == 1 else self.H_I_minus

            H_spatial = torchhd.bind(P_xy, I_p)

            if t not in time_dict:
                time_dict[t] = H_spatial
            else:
                time_dict[t] = torchhd.multiset(torch.stack([time_dict[t], H_spatial]))

        # **Step 2: Bind with Time & Aggregate**
        for t, H_spatial_t in time_dict.items():
            T_t = self.get_time_hv(t)
            H_temporal_t = torchhd.bind(H_spatial_t, T_t)

            if H_spatiotemporal is None:
                H_spatiotemporal = H_temporal_t
            else:
                H_spatiotemporal = torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"Encoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal
