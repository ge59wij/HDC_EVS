import torch
import torchhd
from grasphdencoding_seedhvs import GraspHDseedEncoder
import numpy as np
np.set_printoptions(suppress=True, precision=8)

class GraspHDEventEncoder(GraspHDseedEncoder):
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time):
        super().__init__(height, width, dims, time_subwindow, k=k, device=device, max_time=max_time)
        self.time_hv_cache = {}

    def encode_grasphd(self, events_batch, class_ids):
        """Encodes a batch of events using GraspHD."""
        batch_size = events_batch.shape[0]
        print(f"Encoding {batch_size} samples | Device: {self.device}")

        batch_encoded = []
        for batch_idx in range(batch_size):
            events = events_batch[batch_idx]  # Take individual sample from batch
            class_id = class_ids[batch_idx]

            H_spatiotemporal = None
            time_dict = {}

            for i in range(events.shape[0]):
                t, x, y, polarity = events[i]
                P_xy = self.get_position_hv(int(x.item()), int(y.item()))
                I_p = self.H_I_plus if polarity.item() == 1 else self.H_I_minus
                H_spatial = torchhd.bind(P_xy, I_p)

                if t.item() not in time_dict:
                    time_dict[t.item()] = H_spatial
                else:
                    time_dict[t.item()] = torchhd.multiset(torch.stack([time_dict[t.item()], H_spatial]))

            for t, H_spatial_t in time_dict.items():
                T_t = self.get_time_hv(t)
                H_temporal_t = torchhd.bind(H_spatial_t, T_t)
                H_spatiotemporal = H_temporal_t if H_spatiotemporal is None else torchhd.multiset(
                    torch.stack([H_spatiotemporal, H_temporal_t]))
            batch_encoded.append(torchhd.normalize(H_spatiotemporal))

        return torch.stack(batch_encoded)
