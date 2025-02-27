import torch
import torchhd
import numpy as np
np.set_printoptions(suppress=True, precision=8)

class HDHypervectorGenerators:
    def __init__(self, height, width, dims, device, threshold, window_size, stride):
        self.BACKGROUND_LABEL = 404
        print("Initializing Hypervector Generator:")
        self.height = height
        self.width = width
        self.dims = dims
        self.window_size = window_size
        self.stride = stride
        self.device = torch.device(device)
        self.threshold = threshold  # Event count threshold for noise filtering later

        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on

        self.HV_x = self._generate_continuous_hvs(self.width)
        self.HV_y = self._generate_continuous_hvs(self.height)

        self.time_hvs = self._generate_time_hvs(n_bins=self.window_size)  # Only for 10 bins

    def _generate_continuous_hvs(self, size):
        """Generates continuous hypervectors for X and Y positions."""
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        hvs = [base_hv]

        for _ in range(1, size):
            new_hv = torchhd.permute(hvs[-1])  # Permutation-based encoding
            hvs.append(new_hv)

        return torch.stack(hvs)

    def _generate_time_hvs(self, n_bins):
        """Generates N hypervectors for the time bins inside a window."""
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        return torch.stack([torchhd.permute(base_hv, shifts=i) for i in range(n_bins)])

    def get_pos_hv(self, x: int, y: int) -> torch.Tensor:
        return torchhd.bind(self.HV_x[x], self.HV_y[y])  # Bind X and Y together

    def get_time_hv(self, bin_id: int) -> torch.Tensor:
        return self.time_hvs[min(bin_id, self.window_size - 1)]  # Safe indexing

