import torch
import torchhd
import numpy as np
from collections import defaultdict
import bisect

np.set_printoptions(suppress=True, precision=8)

class seedEncoder:
    def __init__(self, height, width, dims, k, time_subwindow, device, max_time, time_interpolation_method, WINDOW_SIZE_MS, OVERLAP_MS):
        """Initialize seed encoder with spatial and temporal encoding parameters."""
        print("Initializing Seed Encoder:")
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.k = k
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_time = max_time
        self.time_interpolation_method = time_interpolation_method
        self.WINDOW_SIZE_MS = WINDOW_SIZE_MS
        self.OVERLAP_MS = OVERLAP_MS
        self.time_hv_cache = {}
        # Generate base polarity hypervectors
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on

        self._precompute_corners()  # Initialize position-based hypervectors
        print("| Generated Polarity and Position HVs |")
        if self.time_interpolation_method in ["stem_hd", "event_hd_timeinterpolation"]:
                self._generate_time_hvs()  # Initialize time hypervectors



    def _generate_time_hvs(self):
        """Precompute time hypervectors for efficient lookup."""
        self.time_hvs = {}
        num_bins = int(self.max_time // self.time_subwindow) + 2
        if self.time_interpolation_method in ["stem_hd", "event_hd_timeinterpolation"]:
            # Generate anchor hypervectors at time bin edges
            for i in range(num_bins):
                time_key = int(i * self.time_subwindow)
                self.time_hvs[time_key] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)

            # Interpolation between time hypervectors
            for i in range(0, (num_bins - 1) * 10_000, 10_000):

                T_iK = self.time_hvs[i * self.time_subwindow]
                T_next = self.time_hvs[(i + 1) * self.time_subwindow]
                for t in range(1, self.time_subwindow):
                    alpha = t / self.time_subwindow  # Interpolation factor
                    if self.time_interpolation_method == "stem_hd":
                        interpolated_hv = T_iK.clone()
                        interpolated_hv[self.dims // 2:] = T_next[-self.dims // 2:]
                    else:
                        interpolated_hv = (1 - alpha) * T_iK + alpha * T_next
                    self.time_hvs[(i * self.time_subwindow) + t] = interpolated_hv
            print(f"| Precomputed {len(self.time_hvs)} total time hypervectors (anchors + interpolations).")
        elif self.time_interpolation_method == "event_hd_timepermutation":
            print("| Using Temporal Permutation Encoding EVENTHD |")

    def get_time_hv(self, t):
        """Retrieve time hypervector using binary search for efficiency."""
        if not (0 <= t < self.WINDOW_SIZE_MS):
            raise ValueError(f"[ERROR] Event timestamp {t} out of range (0-{self.WINDOW_SIZE_MS})")

        time_keys = list(self.time_hvs.keys())
        idx = bisect.bisect_left(time_keys, t)
        if idx == len(time_keys):
            return self.time_hvs[time_keys[-1]]
        return self.time_hvs[time_keys[idx]]

    # ----------------- Spatial Encoding -------------------

    def _precompute_corners(self):
        """Precompute position hypervectors at fixed grid locations."""
        self.corner_x_positions = list(range(0, self.width, self.k)) + [self.width]
        self.corner_y_positions = list(range(0, self.height, self.k)) + [self.height]

        self.corner_grid = torch.empty(
            (len(self.corner_x_positions), len(self.corner_y_positions), self.dims),
            device=self.device
        )
        for i, x in enumerate(self.corner_x_positions):
            for j, y in enumerate(self.corner_y_positions):
                self.corner_grid[i, j] = torchhd.random(1, self.dims, "MAP", device=self.device)

        self.x_to_index = {x: i for i, x in enumerate(self.corner_x_positions)}
        self.y_to_index = {y: j for j, y in enumerate(self.corner_y_positions)}

        # **Dictionary-based caching for position HVs**
        self.position_hvs_cache = {}

    def get_position_hv(self, x, y):
        """Retrieve or generate Position HV using dictionary-based caching (efficient for large-scale spatial encoding)."""
        key = (int(x), int(y))

        if key not in self.position_hvs_cache:
            self.position_hvs_cache[key] = self._interpolate_hv(x, y)

        return self.position_hvs_cache[key]

    def _interpolate_hv(self, x, y):
        """Interpolates position hypervectors for continuous spatial encoding."""
        x = min(max(x, 0), self.width)
        y = min(max(y, 0), self.height)

        i = max(0, np.searchsorted(self.corner_x_positions, x) - 1)
        j = max(0, np.searchsorted(self.corner_y_positions, y) - 1)

        i_next = min(i + 1, len(self.corner_x_positions) - 1)
        j_next = min(j + 1, len(self.corner_y_positions) - 1)

        P00 = self.corner_grid[self.x_to_index[self.corner_x_positions[i]], self.y_to_index[self.corner_y_positions[j]]]
        P10 = self.corner_grid[self.x_to_index[self.corner_x_positions[i_next]], self.y_to_index[self.corner_y_positions[j]]]
        P01 = self.corner_grid[self.x_to_index[self.corner_x_positions[i]], self.y_to_index[self.corner_y_positions[j_next]]]
        P11 = self.corner_grid[self.x_to_index[self.corner_x_positions[i_next]], self.y_to_index[self.corner_y_positions[j_next]]]

        dx = max(self.corner_x_positions[i_next] - self.corner_x_positions[i], 1e-9)
        dy = max(self.corner_y_positions[j_next] - self.corner_y_positions[j], 1e-9)
        lambda_x = (x - self.corner_x_positions[i]) / dx
        lambda_y = (y - self.corner_y_positions[j]) / dy

        return (
            (1 - lambda_x) * (1 - lambda_y) * P00 +
            lambda_x * (1 - lambda_y) * P10 +
            (1 - lambda_x) * lambda_y * P01 +
            lambda_x * lambda_y * P11
        )



######################Graveyard###############






'''
           elif self.time_interpolation_method in [ "encode_temporalpermutation"]:
               """Shifts an identity HV based on time (no caching)"""
               base_hv = self.time_hvs[0]  # Get identity HV
               return torchhd.permute(base_hv, shifts= int(time % self.time_subwindow))  # Shift based on time
           elif self.time_interpolation_method in [ "thermometer" , "permutation"]:
               return self.time_continious(time)

       def time_continious(self, time):
           """Continuous time encoding for thermometer or permutation."""
           print(f"[DEBUG] time_continious({time}) called with method: {self.time_interpolation_method}")
           if self.time_interpolation_method == "thermometer":
               """The level increases with time."""
               num_bins = len(self.time_hvs.keys())
               scale = time / self.max_time  # Normalize to [0, 1]
               level = int(scale * num_bins)  # Map to thermometer levels
               thermometer_hv = torch.ones(self.dims, device=self.device) * -1
               thermometer_hv[:level] = 1  # Activate increasing levels
               print(f"[DEBUG] Returning Thermometer HV with shape: {thermometer_hv.shape}")
               return thermometer_hv
           elif self.time_interpolation_method == "permutation":
               """Shifts the identity time vector based on time."""
               base_hv = self.time_hvs.get(0, None)  # Get identity HV safely
               if base_hv is None:
                   print("[ERROR] time_hvs[0] is missing!")
                   return torch.zeros(self.dims, device=self.device)  # Fallback
               permuted_hv = torchhd.permute(base_hv, shifts=int(time % self.time_subwindow))  # Shift based on time
               print(f"[DEBUG] Returning Permuted HV with shape: {permuted_hv.shape}")
               return permuted_hv
           else:  # Fallback to original
               return self.get_time_hv(time)
           '''

'''wrong.. concatination spatial.
    def _interpolate_hv(self, x, y):
        """Core interpolation logic."""
        # Clamp coordinates to grid bounds
        x_clamped = min(max(x, 0), self.width)
        y_clamped = min(max(y, 0), self.height)

        # Find surrounding corners
        i = max(0, np.searchsorted(self.corner_x_positions, x_clamped) - 1)
        j = max(0, np.searchsorted(self.corner_y_positions, y_clamped) - 1)

        i_next = min(i + 1, len(self.corner_x_positions) - 1)
        j_next = min(j + 1, len(self.corner_y_positions) - 1)

        # Get corner coordinates
        x0, x1 = self.corner_x_positions[i], self.corner_x_positions[i_next]
        y0, y1 = self.corner_y_positions[j], self.corner_y_positions[j_next]

        # Retrieve HVs from corner_grid (not a local dict)
        i_x0 = self.x_to_index[x0]
        j_y0 = self.y_to_index[y0]
        P00 = self.corner_grid[i_x0, j_y0]

        i_x1 = self.x_to_index[x1]
        P10 = self.corner_grid[i_x1, j_y0]

        j_y1 = self.y_to_index[y1]
        P01 = self.corner_grid[i_x0, j_y1]
        P11 = self.corner_grid[i_x1, j_y1]

        # interpolation weights
        dx = max(x1 - x0, 1e-9)
        dy = max(y1 - y0, 1e-9)
        lambda_x = min(max((x_clamped - x0) / dx, 0.0), 1.0)  # Clamp between 0 and 1
        lambda_y = min(max((y_clamped - y0) / dy, 0.0), 1.0)  # Clamp between 0 and 1

        #print(f"  Weights: λ_x={lambda_x:.2f}, λ_y={lambda_y:.2f}")
        print(f"\nInterpolating at ({x}, {y})")
        print(f"  Clamped: x_clamped={x_clamped}, y_clamped={y_clamped}")
        print(f"  Nearest Corners: P00=({x0}, {y0}), P10=({x1}, {y0}), P01=({x0}, {y1}), P11=({x1}, {y1})")
        print(f"  λ_x={lambda_x:.4f}, λ_y={lambda_y:.4f}")

        # Compute segment sizes (global indices)
        #split_00 = round((1 - lambda_x) * (1 - lambda_y) * self.dims)
        #split_10 = round(lambda_x * (1 - lambda_y) * self.dims)
        #split_01 = round((1 - lambda_x) * lambda_y * self.dims)
        #split_11 = self.dims - (split_00 + split_10 + split_01)
        split_00 = int((1 - lambda_x) * (1 - lambda_y) * self.dims)
        split_10 = int(lambda_x * (1 - lambda_y) * self.dims)
        split_01 = int((1 - lambda_x) * lambda_y * self.dims)
        split_11 = self.dims - (split_00 + split_10 + split_01)  # Force sum to self.dims

        # Ensure the sum of splits equals self.dims
        assert split_00 + split_10 + split_01 + split_11 == self.dims, "Split sizes do not sum to self.dims!"

        # Global indices for each segment (same across all interpolations!!)
        idx_00 = slice(0, split_00)
        idx_10 = slice(split_00, split_00 + split_10)
        idx_01 = slice(split_00 + split_10, split_00 + split_10 + split_01)
        idx_11 = slice(split_00 + split_10 + split_01, self.dims)



        print(f"  Split Sizes -> 00: {split_00}, 10: {split_10}, 01: {split_01}, 11: {split_11}")
        print(f"  Global Indices:")
        print(f"    P00: {idx_00.start}-{idx_00.stop}")
        print(f"    P10: {idx_10.start}-{idx_10.stop}")
        print(f"    P01: {idx_01.start}-{idx_01.stop}")
        print(f"    P11: {idx_11.start}-{idx_11.stop}")

        # Concatenate using global indices (same for all HVs)
        position_hv = torch.cat([
            P00[idx_00],
            P10[idx_10],
            P01[idx_01],
            P11[idx_11]
        ])

        assert position_hv.shape[0] == self.dims, f"Size mismatch! Expected {self.dims}, got {position_hv.shape[0]}"
        #print(f" HV size = {position_hv.shape[0]}")
        return position_hv
'''