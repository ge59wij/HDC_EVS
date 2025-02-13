import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
import numpy as np
from collections import defaultdict

class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time):
        super().__init__(height, width, dims, time_subwindow, k, device, max_time)
        self.time_hv_cache = {}
    def encode_grasphd(self, events_batch, mask, class_ids):
        batch_size = events_batch.shape[0]
        print(f"Encoding {batch_size} samples | Device: {self.device}")
        batch_encoded = []

        for batch_idx in range(batch_size):
            events = events_batch[batch_idx]  # Take individual sample from batch
            valid_mask = mask[batch_idx]  # Get the mask for valid events
            if class_ids.dim() == 0:
                class_id = class_ids.item()  # Convert scalar tensor to number
            else:
                class_id = class_ids[batch_idx]
            H_spatiotemporal = None
            time_dict = {}

            # Step 1: Spatial Encoding (Group Events by Timestamp)
            for i in range(events.shape[0]):
                if not valid_mask[i]:
                    continue  # Ignore padded values

                t, x, y, polarity = events[i]
                P_xy = self.get_position_hv(int(x.item()), int(y.item()))
                I_p = self.H_I_plus if polarity.item() == 1 else self.H_I_minus
                H_spatial = torchhd.bind(P_xy, I_p)

                t_key = t.item()
                if t_key not in time_dict:
                    time_dict[t_key] = H_spatial
                else:
                    time_dict[t_key] = torchhd.multiset(torch.stack([time_dict[t_key], H_spatial]))

            # Step 2: Temporal Binding & Aggregation
            for t, H_spatial_t in time_dict.items():
                T_t = self.get_time_hv(t)
                H_temporal_t = torchhd.bind(H_spatial_t, T_t)

                if H_spatiotemporal is None:
                    H_spatiotemporal = H_temporal_t
                else:
                    H_spatiotemporal = torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

            # Normalize and store
            batch_encoded.append(torchhd.normalize(H_spatiotemporal))

        return torch.stack(batch_encoded)
    def encode_grasp_n_summing(self, events_batch, mask, class_ids):
        """ Encoding method with n-summing inside time windows, while ignoring padding. """
        batch_size = events_batch.shape[0]
        print(f"Encoding {batch_size} samples with n-summing | Device: {self.device}")
        batch_encoded = []

        for batch_idx in range(batch_size):
            events = events_batch[batch_idx]  # Take individual sample from batch
            valid_mask = mask[batch_idx]  # Get the mask for valid events
            if class_ids.dim() == 0:
                class_id = class_ids.item()  # Convert scalar tensor to number
            else:
                class_id = class_ids[batch_idx]
            H_spatiotemporal = None
            window_dict = {}

            # Step 1: Group Events by Time Window
            for i in range(events.shape[0]):
                if not valid_mask[i]:
                    continue  # Ignore padded values

                t, x, y, polarity = events[i]
                P_xy = self.get_position_hv(int(x.item()), int(y.item()))
                I_p = self.H_I_plus if polarity.item() == 1 else self.H_I_minus
                H_spatial = torchhd.bind(P_xy, I_p)

                window_idx = int(t.item() // self.time_subwindow)

                if window_idx not in window_dict:
                    window_dict[window_idx] = {
                        "H_spatial_sum": H_spatial,
                        "first_t": t.item()
                    }
                else:
                    window_dict[window_idx]["H_spatial_sum"] = torchhd.multiset(
                        torch.stack([window_dict[window_idx]["H_spatial_sum"], H_spatial])
                    )

            # Step 2: Temporal Binding & Aggregation
            for window_idx, window_data in window_dict.items():
                T_n = self.get_time_hv(window_data["first_t"])
                H_temporal_t = torchhd.bind(window_data["H_spatial_sum"], T_n)

                if H_spatiotemporal is None:
                    H_spatiotemporal = H_temporal_t
                else:
                    H_spatiotemporal = torchhd.multiset(torch.stack([H_spatiotemporal, H_temporal_t]))

            # Normalize and store
            batch_encoded.append(torchhd.normalize(H_spatiotemporal))

        return torch.stack(batch_encoded)
