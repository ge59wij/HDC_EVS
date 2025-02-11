import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
import numpy as np
np.set_printoptions(suppress=True, precision=8)


class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device):
        device = torch.device(device) if isinstance(device, str) else device
        print(f"Initializing encoder")
        super().__init__(height, width, dims, time_subwindow, k=k, device=device)
        self.time_hv_cache = {}  # **CACHE for precomputed timestamp hypervectors!**

    def encode_grasphd(self, events, class_id):
        """Performs spatial encoding first, then binds with time to obtain final spatiotemporal HV."""
        if not events:
            raise ValueError("No events provided for encoding.")
        print(f"Temporal Encoding: Processing {len(events)} events on {self.device}...")

        last_timestamp = events[-1][0]
        self._generate_time_hvs(last_timestamp)

        H_spatiotemporal = None
        time_dict = {}  # Stores per-timestamp aggregated spatial encoding

        # ----1. Compute Spatial Encoding & Group by Timestamp ----
        for event in events:
            t, (x, y), polarity = event
            P_xy = self.get_position_hv(x, y)  # Position HV
            I_hv = self.H_I_plus if polarity == 1 else self.H_I_minus  # Polarity HV (blue)
            H_spatial = torchhd.bind(P_xy, I_hv)  # Spatial encoding per event

            if t not in time_dict:
                time_dict[t] = H_spatial
            else:
                time_dict[t] = torchhd.bundle(time_dict[t], H_spatial)  # Aggregate events at same timestamp/green, per timestamp)

        # ---- 2.Bind Time HV with Spatial Encoding Per Timestamp ----
        for t, H_spatial_t in time_dict.items():
            # **CACHE TIME HVs TO AVOID DUPLICATE COMPUTATION**
            if t not in self.time_hv_cache:
                self.time_hv_cache[t] = self.get_time_hv(t)  # Save once

            T_t = self.time_hv_cache[t]  # Fetch from cache
            H_temporal_t = torchhd.bind(H_spatial_t, T_t)  # # Temporal encoding (purple, per timestamp)

            # ---- 3. Aggregate Over Time Windows ----
            if H_spatiotemporal is None:
                H_spatiotemporal = H_temporal_t
            else:
                H_spatiotemporal = torchhd.bundle(H_spatiotemporal, H_temporal_t)

        print(f"\nSpatiotemporal Encoding Complete | Class ID: {class_id} | Output Shape: {H_spatiotemporal.shape} | Device: {H_spatiotemporal.device}")
        print("spatio HV: ", H_spatiotemporal[:10])
        return H_spatiotemporal  # Final HV representing the entire sample (purple)
