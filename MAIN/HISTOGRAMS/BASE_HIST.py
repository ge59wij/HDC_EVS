import torch
import torchhd
import numpy as np

np.set_printoptions(suppress=True, precision=8)
class HDHypervectorGenerators:
    def __init__(self, height, width, dims, device, threshold, window_size, n_gram,
                 method_encoding="thermometer", levels=4, K = 6, debug=True):
        """
        Generates base hypervectors for HDC encoding with selectable encoding methods.
        thermometer, linear, eventhd.

        Args:
            height, width (int): downsampled.
            dims (int): Hypervector dimensionality
            device (str): Computation device ('cpu' or 'cuda')
            threshold (float): Event count threshold for empty bin filtering
            window_size (int): Number of bins per window
            n_gram (int): Size of temporal n-grams
            method_encoding (str): Encoding method - "thermometer" or "linear" or "eventhd": permuation or interpolation
            time_encoding (str): Time encoding method - "thermometer", "linear", or "eventhd_timepermutation" or eventhd_interpolation
            levels (int): Number of levels for thermometer encoding
            k (int): Grid size for EventHD's spatial encoding
            debug (bool): Enable debug prints
        """
        self.BACKGROUND_LABEL = 404
        self.step_size = 5    ##### anchor steps for eventhdtimeinterpolation and stemhd.
        if debug:
            print(f"Initializing Hypervector Generator with {method_encoding} encoding...")

        self.height = height
        self.width = width
        self.dims = dims
        self.window_size = window_size
        self.device = torch.device(device)
        self.threshold = threshold
        self.n_gram = n_gram
        self.method_encoding = method_encoding
        self.K = K
        self.debug = debug

        # Generate Polarity Hypervectors (same for all encoding methods)
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on  # OFF polarity is the inverse of ON

        # Generate Spatial Hypervectors based on selected encoding method
        if method_encoding == "linear":
            # Linear mapping encoding
            self.HV_x = self._generate_linear_axis_hvs(self.width)
            self.HV_y = self._generate_linear_axis_hvs(self.height)
            # Cache pixel hypervectors for faster access
            self.pixel_hvs = torch.zeros((height, width, dims), device=self.device)
            self._generate_pixel_hvs()
        elif method_encoding == "thermometer":
            self.HV_x = self._generate_thermometer_axis_hvs(self.width, levels)
            self.HV_y = self._generate_thermometer_axis_hvs(self.height, levels)
        elif method_encoding in [ "eventhd_timepermutation", "eventhd_timetinterpolation" ]:
            self._precompute_eventhd_positions()

        ##time:
        if method_encoding == "eventhd_timepermutation":
            self.base_time_HV = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
            self.time_hvs = None  # Still define it, but we won't use it
        else:
            # Generate Temporal Hypervectors (same for thermometer and linear methods)
            self.time_hvs = self._generate_time_hvs(self.window_size)

        if debug:
            print(f"[DEBUG] Generated Polarity HVs: ON shape {self.H_I_on.shape}, OFF shape {self.H_I_off.shape}")
            print(f"[DEBUG] Generated {len(self.HV_x)} X-position HVs, {len(self.HV_y)} Y-position HVs")
            print(f"[DEBUG] Generated {len(self.time_hvs)} Time HVs")
            if method_encoding == "linear":
                print(f"[DEBUG] Cached {self.pixel_hvs.shape[0] * self.pixel_hvs.shape[1]} Pixel HVs")


########################spatial#######################
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

    def _precompute_eventhd_positions(self):
        """Precomputes position HVs using EventHD's k×k interpolation method and caches results."""
        self.corner_x_positions = list(range(0, self.width, self.K))
        self.corner_y_positions = list(range(0, self.height, self.K))

        # Extend grid to cover the full image, even if not perfectly divisible by K
        if self.width % self.K != 0:
            self.corner_x_positions.append(self.width)
        if self.height % self.K != 0:
            self.corner_y_positions.append(self.height)

        # Store corner hypervectors in a structured grid
        self.corner_grid = torch.empty(
            (len(self.corner_x_positions), len(self.corner_y_positions), self.dims),
            device=self.device
        )

        # Precompute and cache corners
        self.x_to_index = {x: i for i, x in enumerate(self.corner_x_positions)}
        self.y_to_index = {y: j for j, y in enumerate(self.corner_y_positions)}
        self.position_hvs_cache = {}

        for i, x in enumerate(self.corner_x_positions):
            for j, y in enumerate(self.corner_y_positions):
                self.corner_grid[i, j] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
                self.position_hvs_cache[(x, y)] = self.corner_grid[i, j]  # Cache corners directly

        print(f"| Cached {len(self.corner_x_positions) * len(self.corner_y_positions)} corner hypervectors.")


    def get_pos_hv(self, x, y):
        """
        Retrieves spatial hypervector for a given (x, y) position.
        Uses cached values for linear encoding, computes on-the-fly for thermometer.
        for eventhd: If position exists, fetch it. Otherwise, interpolate once, cache it, and return.
        """
        if self.method_encoding == "linear":
            return self.pixel_hvs[y, x]
        elif self.method_encoding == "thermometer":
            return torchhd.bind(self.HV_x[x], self.HV_y[y])
        elif self.method_encoding == "eventhd_timepermutation":
            key = (x, y)
            if key in self.position_hvs_cache:
                return self.position_hvs_cache[key]
            interpolated_hv = self._interpolate_eventhd(x, y)
            self.position_hvs_cache[key] = interpolated_hv  # Store in cache
            return interpolated_hv

    def _interpolate_eventhd(self, x, y):
        """
        Interpolates hypervectors using EventHD's weighted sum method.
        - Uses `np.searchsorted()` to find the correct corners.
        - Fetches from `self.corner_grid` for efficiency.
        - Ensures numerical stability in weight computation.
        - Caches interpolated hypervectors.
        """

        # Clamp to valid range
        x_clamped = min(max(x, 0), self.width)
        y_clamped = min(max(y, 0), self.height)

        # Find nearest grid positions
        i = max(0, np.searchsorted(self.corner_x_positions, x_clamped) - 1)
        j = max(0, np.searchsorted(self.corner_y_positions, y_clamped) - 1)

        i_next = min(i + 1, len(self.corner_x_positions) - 1)
        j_next = min(j + 1, len(self.corner_y_positions) - 1)

        # Get corner positions
        x0, x1 = self.corner_x_positions[i], self.corner_x_positions[i_next]
        y0, y1 = self.corner_y_positions[j], self.corner_y_positions[j_next]

        # Retrieve precomputed hypervectors from grid
        P00 = self.corner_grid[self.x_to_index[x0], self.y_to_index[y0]]
        P10 = self.corner_grid[self.x_to_index[x1], self.y_to_index[y0]]
        P01 = self.corner_grid[self.x_to_index[x0], self.y_to_index[y1]]
        P11 = self.corner_grid[self.x_to_index[x1], self.y_to_index[y1]]

        # Compute interpolation weights
        dx = max(x1 - x0, 1e-9)
        dy = max(y1 - y0, 1e-9)
        lambda_x = (x_clamped - x0) / dx
        lambda_y = (y_clamped - y0) / dy

        # Compute weighted sum
        interpolated_hv = (
                (1 - lambda_x) * (1 - lambda_y) * P00 +
                lambda_x * (1 - lambda_y) * P10 +
                (1 - lambda_x) * lambda_y * P01 +
                lambda_x * lambda_y * P11
        )

        # Cache and return the computed hypervector
        self.position_hvs_cache[(x, y)] = interpolated_hv
        return interpolated_hv
###########################################################time#####
    def _generate_time_hvs(self, n_bins):
        """
        Generates N hypervectors for the time bins inside a window.
        Each time step is a permutation of a base hypervector.
        """
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        return torch.stack([torchhd.permute(base_hv, shifts=i) for i in range(n_bins)])

    def get_time_hv(self, bin_id):
        """
        Retrieves time hypervector based on selected interpolation method.
        Supports:
        - **EventHD time interpolation** (weighted sum) per bin
        - **STEMHD time interpolation** (concatenation-based), one vector per anchor bin
        """
        if self.method_encoding == "eventhd_timepermutation":
            return torchhd.permute(self.base_time_HV, shifts=bin_id)

        elif self.method_encoding in ["eventhd_timeinterpolation", "stem_hd"]:
            # If the time HV already exists, return it
            if bin_id in self.time_hvs:
                return self.time_hvs[bin_id]
            if self.time_hvs is None:
                raise ValueError("Time HVs were not initialized properly. Check method_encoding settings.")

            # Step-size based anchors
            step_size = max(self.window_size // self.step_size, 1)  # Default 5 anchors, adjust as needed ####################
            prev_anchor = (bin_id // step_size) * step_size
            next_anchor = min(prev_anchor + step_size, self.window_size - 1)

            # Get anchor hypervectors
            T_prev = self.time_hvs[prev_anchor]
            T_next = self.time_hvs[next_anchor]

            alpha = (bin_id - prev_anchor) / (next_anchor - prev_anchor + 1e-9)

            if self.method_encoding == "stem_hd":
                # STEMHD: Concatenation-based interpolation temporally, spatial same.(one interpolated vector per anchor pair)
                num_from_T_prev = int((1 - alpha) * self.dims)
                num_from_T_next = self.dims - num_from_T_prev
                interpolated_hv = torch.cat((T_prev[:num_from_T_prev], T_next[-num_from_T_next:]), dim=0)
            else:
                # EventHD Time Interpolation: Weighted sum
                interpolated_hv = (1 - alpha) * T_prev + alpha * T_next

            # Store the interpolated hypervector
            self.time_hvs[bin_id] = interpolated_hv
            return interpolated_hv


