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

        num_corners = ((self.width // self.k) + 1) * ((self.height // self.k) + 1)
        self.corner_hvs = torchhd.embeddings.Random(num_corners, dims, "MAP", device=self.device)
        print(f"| Generated 2 Polarity hvs | Generated {num_corners} Random Corner hvs.")
        self.position_hvs_cache = {}


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
                f"| Using {self.time_interpolation_methode}, caching dynamically | Initial anchors: {len(self.time_hvs)} Timevectors.")

        elif self.time_interpolation_method == "linear_interpolation":
            # Generate anchor time vectors at bin intervals
            for i in range(num_bins):
                time_key = int(i * self.time_subwindow)  # Ensure integer keys
                self.time_hvs[time_key] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)

            print(f"| Using Linear Interpolation for Time HVs | Initial anchors: {len(self.time_hvs)} Timevectors.")

        elif self.time_interpolation_method == "encode_temporalpermutation":
            """Uses identity vectors and shifts them based on time."""
            # We don't store interpolated time HVs, but instead a base identity HV
            self.time_hvs[0] = torchhd.identity(self.dims, device=self.device)  # Base identity HV
            print(f"| Using Temporal Permutation Encoding | Base Identity Vector Initialized")

    def get_time_hv(self, time):
        """Retrieves time hypervector based on selected interpolation method."""

        if self.time_interpolation_method == "grasp_hd":
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

        elif self.time_interpolation_method == "event_hd":  ## Interpolating per actual timestamp
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
            base_hv = self.time_hvs[0]  # Get identity HV
            return torchhd.permute(base_hv, time % self.time_subwindow)  # Shift based on time

        elif self.time_interpolation_method == "linear_interpolation":
            if len(self.time_hvs) == 0:
                raise ValueError(
                    "[ERROR] No time hypervectors found! Ensure _generate_time_hvs() is called before encoding.")

            if time in self.time_hvs:
                return self.time_hvs[time]

            bin_index = (time // self.time_subwindow) * self.time_subwindow
            next_bin = min(bin_index + self.time_subwindow, self.max_time)

            if bin_index in self.time_hvs and next_bin in self.time_hvs:
                proportion = ((time - bin_index) / self.time_subwindow) ** 1.5  # Exponential weighting
                interpolated_hv = (1 - proportion) * self.time_hvs[bin_index] + proportion * self.time_hvs[next_bin]
                self.time_hvs[time] = interpolated_hv  # Cache for future
                return interpolated_hv

            if bin_index in self.time_hvs:
                return self.time_hvs[bin_index]
            elif next_bin in self.time_hvs:
                return self.time_hvs[next_bin]
            else:
                if len(self.time_hvs) == 0:
                    raise ValueError("[ERROR] No time hypervectors initialized!")

                closest_key = min(self.time_hvs.keys(), key=lambda k: abs(k - time))
                print(f"[WARNING] Requested time {time} not found! Using closest available: {closest_key}")
                return self.time_hvs[closest_key]
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
            return torchhd.permute(base_hv, time % self.time_subwindow)  # Shift based on time

        else:  # Fallback to original
            return super().get_time_hv(time)





    #-----------------Spatial-----------------------------------
    def get_position_hv(self, x, y):
        """Compute position hypervector only when active, then cache it, with association"""
        if (x, y) in self.position_hvs_cache:
            return self.position_hvs_cache[(x, y)]

        num_rows = self.height // self.k + 1
        num_cols = self.width // self.k + 1

        i = min(x // self.k, num_rows - 1)
        j = min(y // self.k, num_cols - 1)
        i_next = min(i + 1, num_rows - 1)
        j_next = min(j + 1, num_cols - 1)

        idx_00 = i * num_cols + j
        idx_01 = i * num_cols + j_next
        idx_10 = i_next * num_cols + j
        idx_11 = i_next * num_cols + j_next

        P00 = self.corner_hvs.weight[int(idx_00)]
        P01 = self.corner_hvs.weight[int(idx_01)]
        P10 = self.corner_hvs.weight[int(idx_10)]
        P11 = self.corner_hvs.weight[int(idx_11)]


        # Compute the proportions for each quarter:
        #this: most likely: 2 windows next to each other will get different corners! weird approach but follows paper.
        if x % self.k == 0 and y % self.k == 0:
            position_hv = P00
        elif x % self.k == 0:
            position_hv = torch.cat([P00[:self.dims // 2], P01[self.dims // 2:]])
        elif y % self.k == 0:
            position_hv = torch.cat([P00[:self.dims // 2], P10[self.dims // 2:]])
        else:
            position_hv = torch.cat([
                P00[:self.dims // 4],
                P10[self.dims // 4:self.dims // 2],
                P01[self.dims // 2:3 * self.dims // 4],
                P11[3 * self.dims // 4:]
            ])

        self.position_hvs_cache[(x, y)] = position_hv
        return position_hv

    def weighted_position_hv(self, x, y):
        """Compute position hypervector using weighted linear interpolation instead of concatenation.""" #wrong!
        if (x, y) in self.position_hvs_cache:
            return self.position_hvs_cache[(x, y)]

        num_rows = self.height // self.k + 1
        num_cols = self.width // self.k + 1

        i = min(x // self.k, num_rows - 1)
        j = min(y // self.k, num_cols - 1)
        i_next = min(i + 1, num_rows - 1)
        j_next = min(j + 1, num_cols - 1)

        idx_00 = i * num_cols + j
        idx_01 = i * num_cols + j_next
        idx_10 = i_next * num_cols + j
        idx_11 = i_next * num_cols + j_next

        P00 = self.corner_hvs.weight[int(idx_00)]
        P01 = self.corner_hvs.weight[int(idx_01)]
        P10 = self.corner_hvs.weight[int(idx_10)]
        P11 = self.corner_hvs.weight[int(idx_11)]

        dx = (x % self.k) / self.k
        dy = (y % self.k) / self.k

        # **Linear Interpolation**
        P_x0 = (1 - dx) * P00 + dx * P10  # x-axis (top row)
        P_x1 = (1 - dx) * P01 + dx * P11  # x-axis (bottom row)
        P_final = (1 - dy) * P_x0 + dy * P_x1  #  y-axis

        self.position_hvs_cache[(x, y)] = P_final
        return P_final