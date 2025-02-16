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
        """Encodes events using GraspHD method, added subwindow-based accumulation following the correct paper method."""
        print(f"Encoding {len(events)} events | Class: {class_id} | Device: {self.device}")

        subwindow_dict = {}     # for accumulated spatial vectors
        # **Step 1: Compute Spatial Encoding Per Event then GROUP per subwindow.**
        for t, x, y, polarity in events:
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
                torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

        # **Normalize Final Hypervector**
        H_spatiotemporal = torchhd.normalize(H_spatiotemporal)
        print(f"\nEncoding Complete | Class: {class_id} | Output: {H_spatiotemporal.shape}")
        return H_spatiotemporal
