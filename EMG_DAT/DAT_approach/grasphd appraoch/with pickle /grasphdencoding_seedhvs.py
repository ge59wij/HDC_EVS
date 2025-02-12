import torch
import torchhd
import numpy as np

np.set_printoptions(suppress=True, precision=8)

class GraspHDseedEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time):
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

        # **Time Hypervectors - Now Fully Precomputed!**
        self.max_time = max_time
        self.num_time_bins = int(self.max_time // self.time_subwindow) + 1
        print(f"[INFO] Precomputing {self.num_time_bins} time hypervectors...")
        self.time_hvs = torchhd.random(self.num_time_bins, self.dims, "MAP", device=self.device)

        # **Precompute position hypervectors (Concatenation-based)**
        self.precomputed_positions = self._precompute_position_hvs()

    def get_time_hv(self, time):
        """Retrieve a time hypervector from embeddings"""
        i = int(time // self.time_subwindow)
        i_next = min(i + 1, self.num_time_bins - 1)  # Ensure index is within valid range

        # Debugging prints
        #print(f"[DEBUG] Time Query: time={time}, i={i}, i_next={i_next}, max_time_bin={self.num_time_bins}")

        # Retrieve time hypervectors
        T_i = self.time_hvs[i]
        T_next = self.time_hvs[i_next]

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
