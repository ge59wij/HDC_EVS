import torch
import torchhd
import numpy as np
import warnings


np.set_printoptions(suppress=True, precision=8)

class GraspHDseedEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time):
        print(f"SEED ENCODER: Received params = {height, width, dims, time_subwindow, k, device, max_time}")
        print("Initializing Seed Encoder...")
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.k = int(k)
        self.device = torch.device(device) if isinstance(device, str) else device
        # **Polarity Hypervectors**
        print("Generating 2 Polarity seed Hypervectors...")
        self.H_I_plus = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_minus = -self.H_I_plus

        # **Position Hypervectors (Corner-Based)**
        print("Generating Position seed Hypervectors...")
        num_rows = self.height // self.k + 1
        num_cols = self.width // self.k + 1
        num_corners = num_rows * num_cols
        num_interpolation = (self.height * self.width) - num_corners
        print(f"number of Corners: {num_corners}, Number Interpolated HVS: {num_interpolation}. [ number rows {num_rows}, number cols {num_cols}]")
        self.corner_hvs = torchhd.embeddings.Random(num_corners, dims, "MAP", device=self.device)
        print(f"Generated {num_corners} Corner HVS")
        self.precomputed_positions = self._precompute_position_hvs(num_cols, num_rows)        # **Precompute position hypervectors (Concatenation-based)**
        print(f"Generated {num_interpolation} Position HVS")

        # **Time Hypervectors**
        self.max_time = max_time
        print(f"Max Time: {self.max_time} µs | Time Window: {self.time_subwindow} µs")
        self.num_time_bins = int(np.ceil(self.max_time / self.time_subwindow)) + 1
        print(f"[INFO] Precomputing {self.num_time_bins} border-time hypervectors...")
        self.time_hvs = torchhd.random(self.num_time_bins, self.dims, "MAP", device=self.device)
        # **Precompute Interpolated Time HVs**
        self.interpolated_hvs = self._precompute_interpolated_time_hvs()
        #self.time_hv_cache = {}  # Caching to avoid redundant interpolations #(Concatenation-based)

    def get_time_hv(self, time):
        """Retrieve a time hypervector efficiently using precomputed values."""

        # Identify which time bin we are in
        i = int(time // self.time_subwindow)
        i = min(i, self.num_time_bins - 1)  # Clip to last bin if needed

        # **If exactly at the border, return precomputed border hypervector**
        if time % self.time_subwindow == 0:
            return self.time_hvs[i]

        # **Otherwise, fetch precomputed interpolated hypervector**
        interpolated_index = min(i, self.num_time_bins - 2)  # Ensure valid index
        return self.interpolated_hvs[interpolated_index]
    def _precompute_interpolated_time_hvs(self):
        """Precompute all interpolated time hypervectors and store them in a tensor for fast lookup."""
        print(f"[INFO] Precomputing {self.num_time_bins - 1} interpolated time hypervectors...")

        interpolated_hvs = torch.zeros((self.num_time_bins - 1, self.dims), device=self.device)

        for i in range(self.num_time_bins - 1):
            T_i = self.time_hvs[i]
            T_next = self.time_hvs[i + 1]

            # Generate interpolated hypervectors for different fractions within the time bin
            for j in range(1, self.time_subwindow):  # Assuming we want interpolation within each bin
                alpha_t = j / self.time_subwindow  # Fraction within the window
                split_index = int((1 - alpha_t) * self.dims)  # Elements taken from T_i

                interpolated_hvs[i] = torch.cat((T_i[:split_index], T_next[split_index:]), dim=0)

        print("[DEBUG] Successfully Precomputed All Interpolated Time Hypervectors.\n")
        return interpolated_hvs

    def _precompute_position_hvs(self, num_cols, num_rows):
        """Precompute position interpolation using concatenation instead of weighted sum."""
        precomputed = {}
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
                        P11[3 * self.dims // 4:] ])

        print("[DEBUG] Successfully Precomputed All Position Hypervectors.\n")
        return precomputed
    def get_position_hv(self, x, y): #Retrieve a precomputed position hypervector
        return self.precomputed_positions[(x, y)]
