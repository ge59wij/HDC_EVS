import torch
import torchhd
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "4"
torch.set_num_threads(4)

np.set_printoptions(suppress=True, precision=8)


class HDHypervectorGenerators:
    def __init__(self, height, width, dims, device, threshold, window_size, n_gram,
                 method_encoding, K, debug):
        """
        Generates base hypervectors for HDC encoding with selectable encoding methods.

        Args:
            height, width (int): Image size (downsampled).
            dims (int): Hypervector dimensionality.
            device (str): Computation device ('cpu' or 'cuda').
            threshold (float): Event count threshold for filtering.
            window_size (int): Number of bins per window.
            n_gram (int): Size of temporal n-grams.
            method_encoding (str): "thermometer", "linear", "eventhd_timepermutation", "eventhd_timeinterpolation".
            K (int): Grid size for EventHD-based encoding.
            debug (bool): Enable debug prints.
        """
        self.BACKGROUND_LABEL = 404
        self.device = torch.device(device)
        self.height = height
        self.width = width
        self.dims = dims
        self.window_size = window_size
        self.threshold = threshold
        self.n_gram = n_gram
        self.method_encoding = method_encoding
        self.K = K
        self.debug = debug
        self.step_size  = self.window_size // 9 #10  # Anchor steps for EventHD time interpolation.########


        # **1. Generate Polarity Hypervectors**
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on  # OFF polarity is inverse of ON

        # **2. Generate Spatial Hypervectors**
        if method_encoding in ["linear", "thermometer"]:
            self._generate_spatial_hvs()
        elif method_encoding in ["eventhd_timepermutation", "eventhd_timeinterpolation", "stem_hd", "kxk_ngram"]:
            self._precompute_eventhd_positions()

        # **3. Generate Temporal Hypervectors**
        if method_encoding in ["eventhd_timeinterpolation", "stem_hd"]:
            self.time_hvs = self._generate_time_hvs()

        if debug:
            print(f"[DEBUG] Initialized HDHypervectorGenerators with {method_encoding} encoding.")
            print(f"[DEBUG] Polarity HVs: ON={self.H_I_on.shape}, OFF={self.H_I_off.shape}")
            if method_encoding in [ "thermometer","linear"]:
                print(f"[DEBUG] Cached {self.pixel_hvs.shape[0] * self.pixel_hvs.shape[1]} Pixel HVs.")
                print("No Seed For Time. Using Ngrams.")
            else: print(f"[DEBUG] Time HVs generated: { len(self.time_hvs)}")


    # ----------------- **Spatial Encoding** -------------------

    def _generate_spatial_hvs(self):
        """Generates spatial hypervectors for 'linear' and 'thermometer' encodings."""
        self.HV_x = torchhd.thermometer(self.width, self.dims,
                                        "MAP") if self.method_encoding == "thermometer" else self._generate_linear_axis_hvs(
            self.width)
        self.HV_y = torchhd.thermometer(self.height, self.dims,
                                        "MAP") if self.method_encoding == "thermometer" else self._generate_linear_axis_hvs(
            self.height)

        self.pixel_hvs = torch.stack([
            torchhd.bind(self.HV_x[x], self.HV_y[y])
            for y in range(self.height)
            for x in range(self.width)
        ]).reshape(self.height, self.width, self.dims)

    def _generate_linear_axis_hvs(self, size):
        """Generates position HVs using level hypervectors for gradual similarity transitions."""
        return torchhd.level(size, self.dims, "MAP", device=self.device)

    def _precompute_eventhd_positions(self):
        """Precomputes EventHD-style spatial hypervectors at grid points."""
        self.corner_x_positions = list(range(0, self.width, self.K)) + ([self.width] if self.width % self.K != 0 else [])
        self.corner_y_positions = list(range(0, self.height, self.K)) + ([self.height] if self.height % self.K != 0 else [])
        # Ensure we generate the correct number of unique hypervectors
        self.corner_grid = torch.empty((len(self.corner_x_positions), len(self.corner_y_positions), self.dims), device=self.device)
        for i, x in enumerate(self.corner_x_positions):
            for j, y in enumerate(self.corner_y_positions):
                self.corner_grid[i, j] = torchhd.random(1, self.dims, "MAP", device=self.device)
        # Dictionary-based caching for position HVs
        self.x_to_index = {x: i for i, x in enumerate(self.corner_x_positions)}
        self.y_to_index = {y: j for j, y in enumerate(self.corner_y_positions)}
        self.position_hvs_cache = {}

    def get_pos_hv(self, x, y):
        """
        Retrieves spatial hypervectors for a batch of (x, y) coordinates.

        Args:
            x (torch.Tensor): Tensor of x-coordinates.
            y (torch.Tensor): Tensor of y-coordinates.

        Returns:
            torch.Tensor: A batch of spatial hypervectors.
        """
        # Ensure x and y are 1D tensors
        x, y = x.flatten(), y.flatten()

        # **Handle Linear and Thermometer encoding directly**
        if self.method_encoding in ["linear", "thermometer"]:
            return torchhd.bind(self.HV_x[x], self.HV_y[y])  # Batch lookup

        # **Handle EventHD (KxK grid interpolation)**
        if self.method_encoding in ["eventhd_timepermutation", "eventhd_timeinterpolation", "stem_hd", "kxk_ngram"]:
            pos_hvs = torch.zeros((len(x), self.dims), device=self.device)  # Preallocate tensor

            for i in range(len(x)):  # Process each coordinate
                key = (int(x[i].item()), int(y[i].item()))  # Convert tensor to scalar
                if key in self.position_hvs_cache:
                    pos_hvs[i] = self.position_hvs_cache[key]  # Cached value
                else:
                    pos_hvs[i] = self._interpolate_eventhd(x[i].item(), y[i].item())  # Compute new value
                    self.position_hvs_cache[key] = pos_hvs[i]  # Cache it

            return pos_hvs

        raise ValueError(f"[ERROR] Unknown spatial encoding method: {self.method_encoding}")

    def _interpolate_eventhd(self, x, y):
        """Interpolates spatial hypervectors using weighted sums between grid points."""
        x = min(max(x, 0), self.width)
        y = min(max(y, 0), self.height)

        i = max(0, np.searchsorted(self.corner_x_positions, x) - 1)
        j = max(0, np.searchsorted(self.corner_y_positions, y) - 1)
        i_next = min(i + 1, len(self.corner_x_positions) - 1)
        j_next = min(j + 1, len(self.corner_y_positions) - 1)

        # **✅ Fix: Use Precomputed Hypervectors from Lookup Dictionary**
        P00 = self.corner_grid[self.x_to_index[self.corner_x_positions[i]], self.y_to_index[self.corner_y_positions[j]]]
        P10 = self.corner_grid[
            self.x_to_index[self.corner_x_positions[i_next]], self.y_to_index[self.corner_y_positions[j]]]
        P01 = self.corner_grid[
            self.x_to_index[self.corner_x_positions[i]], self.y_to_index[self.corner_y_positions[j_next]]]
        P11 = self.corner_grid[
            self.x_to_index[self.corner_x_positions[i_next]], self.y_to_index[self.corner_y_positions[j_next]]]

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

    # ----------------- **Time Encoding** -------------------


    def _generate_time_hvs(self):
        """Generates time hypervectors for interpolation-based methods."""

        # Step 1: Determine anchor points
        self.time_hvs = {}
        anchor_bins = list(range(0, self.window_size + 1, self.step_size))  # Ensure last bin included

        for bin_id in anchor_bins:
            self.time_hvs[bin_id] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)

        # Step 3: Interpolate for missing bins
        for i in range(self.window_size):
            if i not in self.time_hvs:
                # Find nearest anchors
                prev_t = max([t for t in anchor_bins if t <= i])
                next_t = min([t for t in anchor_bins if t >= i])

                # Compute interpolation factor
                alpha = (i - prev_t) / (next_t - prev_t + 1e-9)
                # Perform interpolation
                if self.method_encoding == "eventhd_timeinterpolation":
                    self.time_hvs[i] = (1 - alpha) * self.time_hvs[prev_t] + alpha * self.time_hvs[next_t]

                elif self.method_encoding =="stemd_hd":
                    torch.cat((self.time_hvs[prev_t][:self.dims // 2], self.time_hvs[next_t][-self.dims // 2:]), dim=0)

        return self.time_hvs

    def get_time_hv(self, bin_ids):
        """
        Retrieves time hypervectors for a batch of bin indices.

        Args:
            bin_ids (torch.Tensor): Tensor containing multiple time indices.

        Returns:
            torch.Tensor: A batch of time hypervectors.
        """
        if isinstance(bin_ids, int):  # ✅ If a single scalar, return directly
            return self.time_hvs.get(bin_ids, self.time_hvs[min(self.time_hvs.keys(), key=lambda x: abs(x - bin_ids))])

        # **Ensure bin_ids is a 1D tensor**
        bin_ids = bin_ids.flatten()

        # **Look up precomputed time hypervectors**
        time_hvs = torch.zeros((len(bin_ids), self.dims), device=self.device)  # Preallocate tensor
        for i, bin_id in enumerate(bin_ids):
            bin_id = int(bin_id.item())  # Convert tensor element to integer
            time_hvs[i] = self.time_hvs.get(bin_id,
                                            self.time_hvs[min(self.time_hvs.keys(), key=lambda x: abs(x - bin_id))])

        return time_hvs
