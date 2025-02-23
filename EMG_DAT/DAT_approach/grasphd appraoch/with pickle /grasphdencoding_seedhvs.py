import torch
import torchhd
import numpy as np
np.set_printoptions(suppress=True, precision=8)


class GraspHDseedEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time):
        print("Initializing Seed Encoder:")
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.k = k
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_time = max_time

        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on

        # Remove manual corner grid initialization
        self._precompute_corners()  # Moved all corner logic here

        print(f"| Generated 2 Polarity hvs | Generated Random Corner hvs.")
        self._generate_time_hvs()

    def _generate_time_hvs(self):
        """Generate and store all time hypervectors, including border and bin interpolated ones."""
        self.time_hvs = {}  # Dictionary storing all hypervectors
        num_bins = int(self.max_time // self.time_subwindow) + 2

        # print(f"| Interpolated {num_bins} border seed hypervectors.|")
        if self.time_interpolation_method == "grasp_hd":
            """: Interpolates time vectors between anchor bins and stores both anchors and interpolated vectors."""
            for i in range(num_bins):
                self.time_hvs[i] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)

            for i in range(num_bins - 1):
                T_iK = self.time_hvs[i]  # Start bin hypervector
                T_next = self.time_hvs[i + 1]  # Next bin hypervector
                # **Ensure correct slicing sum to DIMS!!!!!!!!!!!!!**
                alpha_t = 1 / num_bins  # Uniform bin spacing
                num_from_T_i = int((1 - alpha_t) * self.dims)
                num_from_T_next = self.dims - num_from_T_i  # Ensure total = self.dims
                interpolated_hv = torch.cat((T_iK[:num_from_T_i], T_next[-num_from_T_next:]), dim=0)  #Every interpolated hypervector seems  extremely close to its neighbors!
                #  Ensure correct dimension**
                assert interpolated_hv.shape[
                           0] == self.dims, f"Incorrect dimension {interpolated_hv.shape[0]}, expected {self.dims}"

                self.time_hvs[i + 0.5] = interpolated_hv  # Store under 0.5step index

        elif self.time_interpolation_method == "event_hd":
            """Dynamically caches time vectors based on the time subwindow."""
            for i in range(num_bins):
                time_key = int(i * self.time_subwindow)  # Ensure integer keys
                self.time_hvs[time_key] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
            print(
                f"| Using {self.time_interpolation_method}, caching dynamically | Initial anchors: {len(self.time_hvs)} Timevectors.")


        elif self.time_interpolation_method == "encode_temporalpermutation":
            """Uses identity vectors and shifts them based on time"""
            # We don't store interpolated time HVs, but instead a base identity HV
            self.time_hvs[0] = torchhd.identity(1, self.dims, device=self.device)  # Base identity HV
            print(f"| Using Temporal Permutation Encoding | Base Identity Vector Initialized")





    def get_time_hv(self, time):
        """Retrieves time hypervector based on selected interpolation method."""

        if self.time_interpolation_method == "grasp_hd":
            """Interpolates between the two closest bins (i.e., T[t] and T[t+1])"""
            bin_index = int((time // self.time_subwindow))  # Ensure integer key
            interpolated_key = bin_index + 0.5

            bin_fraction = (time % self.time_subwindow) / self.time_subwindow

            if interpolated_key in self.time_hvs and bin_fraction >= 0.5:
                return self.time_hvs[interpolated_key]
            elif bin_index in self.time_hvs:
                return self.time_hvs[bin_index]
            else:
                print(f"[ERROR] Missing Time HV for time={time} (Bin: {bin_index}, Interpolated: {interpolated_key})")
                return None  # debug

        elif self.time_interpolation_method == "event_hd":
            """
             Interpolating per actual timestamp
             Dynamically generates a new time HV and caches it"""
            if time in self.time_hvs:
                return self.time_hvs[time]
            bin_index = (time // self.time_subwindow) * self.time_subwindow
            next_bin = min(bin_index + self.time_subwindow, self.max_time)
            if bin_index in self.time_hvs and next_bin in self.time_hvs:
                proportion_1 = (next_bin - time) / self.time_subwindow  #   % from T[jt] left anchor
                num_from_T_i = int(proportion_1 * self.dims)
                num_from_T_next = self.dims - num_from_T_i  # Ensure total = self.dims
                interpolated_hv = torch.cat((
                    self.time_hvs[bin_index][:num_from_T_i], # Take from start of Tjt
                    self.time_hvs[next_bin][-num_from_T_next:]  # Take from end of Tj+1 t
                ), dim=0)
                self.time_hvs[time] = interpolated_hv  # Cache for later use
                return interpolated_hv
            if bin_index in self.time_hvs:
                return self.time_hvs[bin_index]
            elif next_bin in self.time_hvs:
                return self.time_hvs[next_bin]
            else:
                closest_key = min(self.time_hvs.keys(), key=lambda k: abs(k - time))
                print(f"[WARNING] Requested time {time} not found! Using closest available: {closest_key}")
                return self.time_hvs[closest_key]

        elif self.time_interpolation_method == "encode_temporalpermutation":
            """Shifts an identity HV based on time (no caching)"""
            base_hv = self.time_hvs[0]  # Get identity HV
            return torchhd.permute(base_hv, shifts= int(time % self.time_subwindow))  # Shift based on time

        elif self.time_interpolation_method in [ "thermometer" , "permutation"]:
            return self.time_continious(time)
    def time_continious(self, time):
        """Continuous time encoding for thermometer or permutation."""
        if self.time_interpolation_method == "thermometer":
            """the level increases with time."""
            num_bins = len(self.time_hvs.keys())
            scale = time / self.max_time  # Normalize to [0, 1]
            level = int(scale * num_bins)  # Map to thermometer levels

            thermometer_hv = torch.ones(self.dims, device=self.device) * -1
            thermometer_hv[:level] = 1  # activate increasing levels
            return thermometer_hv

        elif self.time_interpolation_method == "permutation":
            """Shifts the identity time vector based on time"""
            base_hv = self.time_hvs[0]  # Get identity HV
            return torchhd.permute(base_hv, shifts=int(time % self.time_subwindow))  # Shift based on time

        else:  # Fallback to original
            return super().get_time_hv(time)





    #-----------------Spatial-----------------------------------

    def _precompute_corners(self):
        """Precompute corners, virtual corners, and populate cache."""
        # Generate corner positions with virtual extension
        self.corner_x_positions = list(range(0, self.width, self.k))
        self.corner_y_positions = list(range(0, self.height, self.k))

        # Extend to virtual corners if needed
        if self.width % self.k != 0:
            self.corner_x_positions.append(self.width)
        if self.height % self.k != 0:
            self.corner_y_positions.append(self.height)

        # Generate corner grid
        self.corner_grid = torch.empty(
            (len(self.corner_x_positions), len(self.corner_y_positions), self.dims),
            device=self.device
        )
        for i, x in enumerate(self.corner_x_positions):
            for j, y in enumerate(self.corner_y_positions):
                self.corner_grid[i, j] = torchhd.random(1, self.dims, "MAP", device=self.device)

        # Preload corners into cache
        self.x_to_index = {x: i for i, x in enumerate(self.corner_x_positions)}
        self.y_to_index = {y: j for j, y in enumerate(self.corner_y_positions)}
        self.position_hvs_cache = {
            (x, y): self.corner_grid[i, j]
            for i, x in enumerate(self.corner_x_positions)
            for j, y in enumerate(self.corner_y_positions)
        }

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

        # Compute segment sizes (global indices)
        split_00 = round((1 - lambda_x) * (1 - lambda_y) * self.dims)
        split_10 = round(lambda_x * (1 - lambda_y) * self.dims)
        split_01 = round((1 - lambda_x) * lambda_y * self.dims)
        split_11 = self.dims - (split_00 + split_10 + split_01)

        # Ensure the sum of splits equals self.dims
        assert split_00 + split_10 + split_01 + split_11 == self.dims, "Split sizes do not sum to self.dims!"

        # Global indices for each segment (same across all interpolations!!)
        idx_00 = slice(0, split_00)
        idx_10 = slice(split_00, split_00 + split_10)
        idx_01 = slice(split_00 + split_10, split_00 + split_10 + split_01)
        idx_11 = slice(split_00 + split_10 + split_01, self.dims)
        '''
        print(f"  Global Indices:")
        print(f"    P00: {idx_00.start}-{idx_00.stop}")
        print(f"    P10: {idx_10.start}-{idx_10.stop}")
        print(f"    P01: {idx_01.start}-{idx_01.stop}")
        print(f"    P11: {idx_11.start}-{idx_11.stop}")
        '''
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