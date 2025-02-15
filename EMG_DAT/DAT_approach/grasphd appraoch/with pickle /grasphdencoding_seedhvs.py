import torch
import torchhd
import numpy as np
np.set_printoptions(suppress=True, precision=8)


class GraspHDseedEncoder:
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time):
        print("Initializing Seed Encoder...")
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
        print(f"Generated 2 Polarity hvs. Generated {num_corners} Random Corner hvs.")

        # **Position Interpolated Hypervectors - Lazy Initialization**
        self.position_hvs_cache = {}  # Store computed hypervectors on demand
        # **Time Hypervector Initialization**
        self._generate_time_hvs()

    def _generate_time_hvs(self):
        """Generate and store all time hypervectors, including border and bin interpolated ones."""
        self.time_hvs = {}  # Dictionary storing all hypervectors

        # **Step 1: Generate Time Borders**
        num_bins = int(self.max_time // self.time_subwindow) + 1
        print(f"Generating {num_bins} border seed hypervectors...")

        for i in range(num_bins):
            self.time_hvs[i] = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)

        # **Step 2: Generate Interpolated Bin Hypervectors**
        print(f"Interpolating and caching **{num_bins - 1}** bin time hypervectors...")

        for i in range(num_bins - 1):
            T_iK = self.time_hvs[i]  # Start bin hypervector
            T_next = self.time_hvs[i + 1]  # Next bin hypervector

            # **Ensure correct slicing sum to DIMS!!!!!!!!!!!!!**
            alpha_t = 1 / num_bins  # Uniform bin spacing
            num_from_T_i = int((1 - alpha_t) * self.dims)
            num_from_T_next = self.dims - num_from_T_i  # Ensure total = self.dims

            interpolated_hv = torch.cat((T_iK[:num_from_T_i], T_next[-num_from_T_next:]), dim=0)

            #  Ensure correct dimension**
            assert interpolated_hv.shape[
                       0] == self.dims, f"Incorrect dimension {interpolated_hv.shape[0]}, expected {self.dims}"

            self.time_hvs[i + 0.5] = interpolated_hv  # Store under 0.5 step index

        print(f"Time hypervectors initialized. {len(self.time_hvs)} total vectors stored.")

    def get_time_hv(self, time):
        """Retrieve precomputed time hypervector based on event timestamp."""
        bin_index = time // self.time_subwindow  # Get the bin index
        bin_fraction = (time % self.time_subwindow) / self.time_subwindow

        # **Fix: Only fetch `bin_index + 0.5` if it exists**
        if bin_index + 0.5 in self.time_hvs and bin_fraction >= 0.5:
            T_t = self.time_hvs[bin_index + 0.5]
        else:
            T_t = self.time_hvs[bin_index]  # Default to the bin index

        return T_t

    def get_position_hv(self, x, y):
        """Compute position hypervector only when needed, then cache it."""
        if (x, y) in self.position_hvs_cache:
            return self.position_hvs_cache[(x, y)]

        # Compute indices
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

        # Retrieve corner hypervectors
        P00 = self.corner_hvs(torch.tensor(idx_00, dtype=torch.long, device=self.device))
        P01 = self.corner_hvs(torch.tensor(idx_01, dtype=torch.long, device=self.device))
        P10 = self.corner_hvs(torch.tensor(idx_10, dtype=torch.long, device=self.device))
        P11 = self.corner_hvs(torch.tensor(idx_11, dtype=torch.long, device=self.device))

        # Compute the proportions for each quarter
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

        # Cache the computed position hypervector
        self.position_hvs_cache[(x, y)] = position_hv
        return position_hv