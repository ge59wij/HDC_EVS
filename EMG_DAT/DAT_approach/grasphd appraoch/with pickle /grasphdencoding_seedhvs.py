import torch
import torchhd
import numpy as np

class GraspHDseedEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device):
        print("Initializing seed Encoder...")
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.k = k
        self.device = torch.device(device) if isinstance(device, str) else device

        print("Generating Polarity seed Hypervectors...")
        self.H_I_plus = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_minus = -self.H_I_plus

        print("Generating 2D Position seed Hypervectors...")
        self.corner_hvs = self._generate_corner_hvs()

        print("Initializing seed Timestamp Hypervectors...")
        self.time_hvs = None

        #  Precompute position interpolation factors
        self.precomputed_positions = self._precompute_position_hvs()

    def _generate_corner_hvs(self):
        """Generates corner hypervectors ONCE and stores them."""
        num_rows = self.height // self.k + 1
        num_cols = self.width // self.k + 1
        return torchhd.random(num_rows * num_cols, self.dims, "MAP",
                              device=self.device).reshape(num_rows, num_cols, self.dims)

    def _precompute_position_hvs(self):
        """Precompute position interpolation values"""
        precomputed = {}
        for x in range(self.width):
            for y in range(self.height):
                i = min(x // self.k, self.corner_hvs.shape[0] - 1)
                j = min(y // self.k, self.corner_hvs.shape[1] - 1)
                i_next = min(i + 1, self.corner_hvs.shape[0] - 1)
                j_next = min(j + 1, self.corner_hvs.shape[1] - 1)

                # Retrieve corner hypervectors
                P00 = self.corner_hvs[i, j]
                P01 = self.corner_hvs[i, j_next]
                P10 = self.corner_hvs[i_next, j]
                P11 = self.corner_hvs[i_next, j_next]

                # Compute interpolation factors
                alpha_x = (x % self.k) / float(self.k - 1) if self.k > 1 else 0.5
                alpha_y = (y % self.k) / float(self.k - 1) if self.k > 1 else 0.5

                # Precompute the interpolated hypervector
                precomputed[(x, y)] = (
                    (1 - alpha_x) * (1 - alpha_y) * P00 +
                    alpha_x * (1 - alpha_y) * P10 +
                    (1 - alpha_x) * alpha_y * P01 +
                    alpha_x * alpha_y * P11
                )

        print("Precomputed all position hypervectors!")
        return precomputed

    def get_position_hv(self, x, y):
        return self.precomputed_positions[(x, y)]

    def _generate_time_hvs(self, last_timestamp):
        """Generate and store time hypervectors."""
        num_time_bins = int((last_timestamp // self.time_subwindow) + 1)

        if self.time_hvs is not None and self.time_hvs.shape[0] >= num_time_bins:
            return

        missing_hvs = torchhd.random(num_time_bins - (self.time_hvs.shape[0] if self.time_hvs is not None else 0),
                                     self.dims, "MAP", device=self.device)
        self.time_hvs = torch.cat([self.time_hvs, missing_hvs], dim=0) if self.time_hvs is not None else missing_hvs

    def get_time_hv(self, time):
        """Retrieve or generate a time hypervector for a timestamp"""
        i = int(time // self.time_subwindow)
        i_next = min(i + 1, self.time_hvs.shape[0] - 1)  # Ensure index is within range

        T_i = self.time_hvs[i]
        T_next = self.time_hvs[i_next]
        alpha_t = (time % self.time_subwindow) / self.time_subwindow if self.time_subwindow > 1 else 0.5

        #dimension-specific interpolation
        num_from_T_i = int((1 - alpha_t) * self.dims)
        num_from_T_next = self.dims - num_from_T_i
        interpolated_hv = torch.cat((T_i[:num_from_T_i], T_next[:num_from_T_next]), dim=0)
        return interpolated_hv
