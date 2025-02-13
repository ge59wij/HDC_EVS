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
        self.precomputed_positions = self._precompute_position_hvs(num_cols, num_rows)        # **Precompute position hypervectors (Concatenation-based)** #307,200 iterations!!!!!!!!!!
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
        """Vectorized approach to precompute position hypervectors."""
        x_indices = torch.arange(self.width, device=self.device).unsqueeze(1)  # (640, 1)
        y_indices = torch.arange(self.height, device=self.device).unsqueeze(0)  # (1, 480)

        i = torch.clamp(x_indices // self.k, max=num_rows - 1)
        j = torch.clamp(y_indices // self.k, max=num_cols - 1)
        i_next = torch.clamp(i + 1, max=num_rows - 1)
        j_next = torch.clamp(j + 1, max=num_cols - 1)

        idx_00 = (i * num_cols + j).flatten()
        idx_01 = (i * num_cols + j_next).flatten()
        idx_10 = (i_next * num_cols + j).flatten()
        idx_11 = (i_next * num_cols + j_next).flatten()

        P00 = self.corner_hvs(idx_00.clone().detach())
        P01 = self.corner_hvs(idx_01.clone().detach())
        P10 = self.corner_hvs(idx_10.clone().detach())
        P11 = self.corner_hvs(idx_11.clone().detach())

        precomputed = torch.cat([P00[:, :self.dims // 4],
                                 P10[:, self.dims // 4:self.dims // 2],
                                 P01[:, self.dims // 2:3 * self.dims // 4],
                                 P11[:, 3 * self.dims // 4:]], dim=1)

        return precomputed

    def get_position_hv(self, x, y): #Retrieve a precomputed position hypervector
        return self.precomputed_positions[(x, y)]
