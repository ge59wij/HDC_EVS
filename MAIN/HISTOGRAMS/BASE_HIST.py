import torch
import torchhd
import numpy as np

np.set_printoptions(suppress=True, precision=8)


class HDHypervectorGenerators:
    def __init__(self, height, width, dims, device, threshold, window_size, n_gram):
        """
        Generates base hypervectors for HDC encoding:
        - Position HVs (thermometer encoding)
        - Polarity HVs for ON/OFF events
        - Temporal HVs for different time steps
        """
        self.BACKGROUND_LABEL = 404
        print("Initializing Hypervector Generator...")

        self.height = height
        self.width = width
        self.dims = dims
        self.window_size = window_size  # Number of bins per window
        self.device = torch.device(device)
        self.threshold = threshold  # Event count threshold for noise filtering
        self.n_gram = n_gram

        # **Generate Hypervectors**
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on  # OFF polarity is the inverse of ON

        # **Spatial Hypervectors using Thermometer Encoding**
        self.HV_x = self._generate_axis_hvs(self.width, levels=width)
        self.HV_y = self._generate_axis_hvs(self.height, levels=height)
        print(f"[DEBUG] Generated Polarity HVs: ON shape {self.H_I_on.shape}, OFF shape {self.H_I_off.shape}")
        print(f"[DEBUG] Generated {len(self.HV_x)} X-position HVs, {len(self.HV_y)} Y-position HVs")

        # **Temporal Hypervectors**
        self.time_hvs = self._generate_time_hvs(self.window_size)
        print(f"[DEBUG] Generated {len(self.time_hvs)} Time HVs")

    def _generate_axis_hvs(self, size, levels):
        """
        Creates continuous position HVs using **thermometer encoding**.
        - Uses multiple **permutation levels** for gradual variation.
        - Nearby positions share more permutation levels → higher similarity.
        """
        base = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        hvs = [base]

        for lvl in range(1, levels):
            permuted = torchhd.permute(hvs[-1], shifts=lvl)
            hvs.append(torchhd.bundle(hvs[-1], permuted))  # Combine increasing shifts

        return [hvs[min(int(pos / size * levels), levels - 1)] for pos in range(size)]

    def _generate_time_hvs(self, n_bins):
        """
        Generates **N** hypervectors for the time bins inside a window.
        - Each time step is **a permutation** of a base hypervector.
        - Ensures temporal structure is **preserved**.
        """
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        return torch.stack([torchhd.permute(base_hv, shifts=i) for i in range(n_bins)])

    def get_pos_hv(self, x: int, y: int) -> torch.Tensor:
        """
        Retrieves **spatial hypervector** for a given (x, y) position.
        - Uses **thermometer-encoded** hypervectors.
        """
        return torchhd.bind(self.HV_x[x], self.HV_y[y])

    def get_time_hv(self, bin_id: int) -> torch.Tensor:
        """
        Retrieves **temporal hypervector** for a given time bin.
        - Ensures valid indexing.
        """
        return self.time_hvs[min(bin_id, self.window_size - 1)]
