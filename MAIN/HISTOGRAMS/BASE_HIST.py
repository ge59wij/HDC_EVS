import torch
import torchhd
import numpy as np

np.set_printoptions(suppress=True, precision=8)
class HDHypervectorGenerators:
    def __init__(self, height, width, dims, device, threshold, window_size, n_gram,
                 spatial_encoding="thermometer", levels=4, debug=True):
        """
        Generates base hypervectors for HDC encoding with selectable encoding methods.

        Args:
            height (int): Image height
            width (int): Image width
            dims (int): Hypervector dimensionality
            device (str): Computation device ('cpu' or 'cuda')
            threshold (float): Event count threshold for noise filtering
            window_size (int): Number of bins per window
            n_gram (int): Size of temporal n-grams
            spatial_encoding (str): Encoding method - "thermometer" or "linear"
            levels (int): Number of levels for thermometer encoding
            debug (bool): Enable debug prints
        """
        self.BACKGROUND_LABEL = 404
        if debug:
            print(f"Initializing Hypervector Generator with {spatial_encoding} encoding...")

        self.height = height
        self.width = width
        self.dims = dims
        self.window_size = window_size
        self.device = torch.device(device)
        self.threshold = threshold
        self.n_gram = n_gram
        self.spatial_encoding = spatial_encoding
        self.debug = debug

        # Generate Polarity Hypervectors (same for both encoding methods)
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on  # OFF polarity is the inverse of ON

        # Generate Spatial Hypervectors based on selected encoding method
        if spatial_encoding == "linear":
            # Linear mapping encoding
            self.HV_x = self._generate_linear_axis_hvs(self.width)
            self.HV_y = self._generate_linear_axis_hvs(self.height)
            # Cache pixel hypervectors for faster access
            self.pixel_hvs = torch.zeros((height, width, dims), device=self.device)
            self._generate_pixel_hvs()
        else:  # thermometer encoding
            # Thermometer encoding
            self.HV_x = self._generate_thermometer_axis_hvs(self.width, levels)
            self.HV_y = self._generate_thermometer_axis_hvs(self.height, levels)

        # Generate Temporal Hypervectors (same for both methods)
        self.time_hvs = self._generate_time_hvs(self.window_size)

        if debug:
            print(f"[DEBUG] Generated Polarity HVs: ON shape {self.H_I_on.shape}, OFF shape {self.H_I_off.shape}")
            print(f"[DEBUG] Generated {len(self.HV_x)} X-position HVs, {len(self.HV_y)} Y-position HVs")
            print(f"[DEBUG] Generated {len(self.time_hvs)} Time HVs")
            if spatial_encoding == "linear":
                print(f"[DEBUG] Cached {self.pixel_hvs.shape[0] * self.pixel_hvs.shape[1]} Pixel HVs")

    def _generate_linear_axis_hvs(self, size):
        """
        Generates position HVs using linear mapping, ensuring similarity between nearby positions.
        Each position flips a small number of bits from the previous position's HV.
        """
        flip_bits = self.dims // (4 * (size - 1)) #flip_bits = self.dims // (4 * (size - 1))
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        hvs = [base_hv.clone()]

        for i in range(1, size):
            new_hv = hvs[-1].clone()
            flip_indices = torch.randperm(self.dims)[:flip_bits]
            new_hv[flip_indices] = -new_hv[flip_indices]
            hvs.append(new_hv)

        return torch.stack(hvs)

    def _generate_thermometer_axis_hvs(self, size, levels=4):
        """
        Creates continuous position HVs using thermometer encoding.
        Uses multiple permutation levels for gradual variation.
        Nearby positions share more permutation levels -> higher similarity.
        """
        base = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        hvs = [base]

        for lvl in range(1, levels):
            permuted = torchhd.permute(hvs[-1], shifts=lvl)
            hvs.append(torchhd.bundle(hvs[-1], permuted))  # Combine increasing shifts

        return [hvs[min(int(pos / size * levels), levels - 1)] for pos in range(size)]

    def _generate_pixel_hvs(self):
        """Generates and caches hypervectors for each pixel (x, y) for linear encoding."""
        for y in range(self.height):
            for x in range(self.width):
                self.pixel_hvs[y, x] = torchhd.bind(self.HV_x[x], self.HV_y[y])

    def _generate_time_hvs(self, n_bins):
        """
        Generates N hypervectors for the time bins inside a window.
        Each time step is a permutation of a base hypervector.
        """
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        return torch.stack([torchhd.permute(base_hv, shifts=i) for i in range(n_bins)])

    def get_pos_hv(self, x, y):
        """
        Retrieves spatial hypervector for a given (x, y) position.
        Uses cached values for linear encoding, computes on-the-fly for thermometer.
        """
        if self.spatial_encoding == "linear":
            return self.pixel_hvs[y, x]
        else:  # thermometer
            return torchhd.bind(self.HV_x[x], self.HV_y[y])

    def get_time_hv(self, bin_id):
        """
        Retrieves temporal hypervector for a given time bin.
        Ensures valid indexing.
        """
        return self.time_hvs[min(bin_id, self.window_size - 1)]