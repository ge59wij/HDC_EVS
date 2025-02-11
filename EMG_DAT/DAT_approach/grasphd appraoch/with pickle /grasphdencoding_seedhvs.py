import torch
import torchhd
import numpy as np

np.set_printoptions(suppress=True, precision=8)


class GraspHDseedEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device):
        print("Initializing Seed Encoder...")
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow  # Defines the interval for time encoding
        self.k = k
        self.device = torch.device(device) if isinstance(device, str) else device

        # **Polarity Hypervectors**
        print("Generating Polarity seed Hypervectors...")
        self.H_I_plus = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_minus = -self.H_I_plus

        # **Position Hypervectors (Corner-Based)**
        print("Generating Position seed Hypervectors...")
        num_corners = ((self.width // self.k) + 1) * ((self.height // self.k) + 1)
        self.corner_hvs = torchhd.embeddings.Random(num_corners, dims, "MAP", device=self.device)

        # **Time Hypervectors (Lazy Initialization)**
        self.time_hvs = None
        self.max_time = None  # To be determined dynamically

        # **Precompute position hypervectors (Concatenation-based)**
        self.precomputed_positions = self._precompute_position_hvs()

    def _update_time_hvs(self, last_timestamp):
        """Ensure enough time hypervectors exist for the dataset."""
        num_time_bins = int((last_timestamp // self.time_subwindow) + 1)

        if self.time_hvs is None:
            # Initialize time hypervectors properly as a tensor, not an embedding object
            self.time_hvs = torchhd.random(num_time_bins, self.dims, "MAP", device=self.device)
            print(f"[DEBUG] Initialized Time Hypervectors up to {last_timestamp}µs (Bins: {num_time_bins})")

        elif num_time_bins > self.time_hvs.shape[0]:  # Expand if needed
            missing_hvs = torchhd.random(num_time_bins - self.time_hvs.shape[0], self.dims, "MAP", device=self.device)
            self.time_hvs = torch.cat([self.time_hvs, missing_hvs], dim=0)  # Concatenate new hypervectors
            print(f"[DEBUG] Expanded Time Hypervectors up to {last_timestamp}µs (Bins: {num_time_bins})")

    def get_time_hv(self, time):
        """Retrieve a time hypervector from embeddings"""
        i = int(time // self.time_subwindow)

        # Ensure `i` is within the range of generated hypervectors
        max_time_bin = (self.max_time // self.time_subwindow) if hasattr(self, 'max_time') else None
        if max_time_bin is not None and i >= max_time_bin:
            print(f"[WARNING] Clamping time index {i} to max available {max_time_bin - 1}")
            i = max_time_bin - 1  # Avoid out-of-bounds error

        i_next = min(i + 1, max_time_bin - 1) if max_time_bin is not None else i + 1

        # Debugging prints
        print(f"[DEBUG] Time Query: time={time}, i={i}, i_next={i_next}, max_time_bin={max_time_bin}")

        # Retrieve time hypervectors
        T_i = self.time_hvs(torch.tensor(i, dtype=torch.long, device=self.device))
        T_next = self.time_hvs(torch.tensor(i_next, dtype=torch.long, device=self.device))

        alpha_t = (time % self.time_subwindow) / self.time_subwindow if self.time_subwindow > 1 else 0.5

        # Interpolation
        num_from_T_i = int((1 - alpha_t) * self.dims)
        num_from_T_next = self.dims - num_from_T_i
        interpolated_hv = torch.cat((T_i[:num_from_T_i], T_next[:num_from_T_next]), dim=0)

        return interpolated_hv

    def _precompute_position_hvs(self):
        """Precompute position interpolation using concatenation instead of weighted sum."""
        precomputed = {}

        num_rows = self.height // self.k + 1
        num_cols = self.width // self.k + 1

        for x in range(self.width):
            for y in range(self.height):
                i = min(x // self.k, num_rows - 1)
                j = min(y // self.k, num_cols - 1)
                i_next = min(i + 1, num_rows - 1)
                j_next = min(j + 1, num_cols - 1)

                idx_00 = i * num_cols + j
                idx_01 = i * num_cols + j_next
                idx_10 = i_next * num_cols + j
                idx_11 = i_next * num_cols + j_next

                # Retrieve corner hypervectors
                P00 = self.corner_hvs(torch.tensor(idx_00, dtype=torch.long, device=self.device))
                P01 = self.corner_hvs(torch.tensor(idx_01, dtype=torch.long, device=self.device))
                P10 = self.corner_hvs(torch.tensor(idx_10, dtype=torch.long, device=self.device))
                P11 = self.corner_hvs(torch.tensor(idx_11, dtype=torch.long, device=self.device))

                # Compute the proportions for each quarter
                if x % self.k == 0 and y % self.k == 0:
                    precomputed[(x, y)] = P00
                elif x % self.k == 0:
                    precomputed[(x, y)] = torch.cat([P00[:self.dims // 2], P01[self.dims // 2:]])
                elif y % self.k == 0:
                    precomputed[(x, y)] = torch.cat([P00[:self.dims // 2], P10[self.dims // 2:]])
                else:
                    precomputed[(x, y)] = torch.cat([
                        P00[:self.dims // 4],
                        P10[self.dims // 4:self.dims // 2],
                        P01[self.dims // 2:3 * self.dims // 4],
                        P11[3 * self.dims // 4:]
                    ])

        print("[DEBUG] Successfully Precomputed All Position Hypervectors (Concatenation-Based)!\n")
        return precomputed

    def get_position_hv(self, x, y):
        return self.precomputed_positions[(x, y)]
