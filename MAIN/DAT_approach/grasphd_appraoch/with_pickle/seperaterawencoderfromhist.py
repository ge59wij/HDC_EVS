import torch
import torchhd
import numpy as np
from collections import defaultdict

np.set_printoptions(suppress=True, precision=8)


class Raw_events_HDEncoder_Enhanced:
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS):
        self.n_gram = 5
        self.height = height
        self.width = width
        self.dims = dims
        self.time_subwindow = time_subwindow
        self.device = torch.device(device)
        self.time_method = time_method
        self.WINDOW_SIZE_MS = WINDOW_SIZE_MS
        self.OVERLAP_MS = OVERLAP_MS

        # Polarity Hypervectors
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on
        self.pixel_hvs = torch.zeros((height, width, dims), device=self.device)

        # Generate Spatial Encoding Hypervectors
        if time_method == "linear":
            self.HV_x = self._generate_linear_axis_hvs(self.width)
            self.HV_y = self._generate_linear_axis_hvs(self.height)
            self._generate_pixel_hvs()
        elif time_method == "thermometer":
            self.HV_x = torchhd.thermometer(self.width, self.dims, "MAP")
            self.HV_y = torchhd.thermometer(self.height, self.dims, "MAP")
            self._generate_pixel_hvs()

    def get_pos_hv(self, x, y):
        x_idx = int(x)
        y_idx = int(y)
        return self.pixel_hvs[y_idx, x_idx]

    def _generate_linear_axis_hvs(self, size):
        flip_bits = self.dims // (4 * (size - 1))
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        hvs = [base_hv.clone()]

        for i in range(1, size):
            new_hv = hvs[-1].clone()
            flip_indices = torch.randperm(self.dims)[:flip_bits]
            new_hv[flip_indices] = -new_hv[flip_indices]
            hvs.append(new_hv)

        return torch.stack(hvs)

    def _generate_pixel_hvs(self):
        for y in range(self.height):
            for x in range(self.width):
                self.pixel_hvs[y, x] = torchhd.bind(self.HV_x[x], self.HV_y[y])

    def encode_bin(self, bin_data, time_idx):
        """Encodes a single bin by bundling spatial hypervectors and binding with N-gram ordering."""
        if len(bin_data) == 0:
            print(f"[WARNING] Skipping empty bin at index {time_idx}")
            return None

        # Compute spatial encoding for all events in the bin
        spatial_hvs = torch.stack([
            torchhd.bind(self.get_pos_hv(x, y), self.H_I_on if polarity == 1 else self.H_I_off)
            for _, x, y, polarity in bin_data
        ])
        bin_hv = torchhd.normalize(torchhd.multibundle(spatial_hvs))

        return bin_hv  # No time HV applied here, it's handled in N-grams

    def encode_window(self, window_data):
        """Encodes an entire window by forming and binding N-grams from bins."""
        bins = [(bin_data, idx) for idx, bin_data in enumerate(window_data) if len(bin_data) > 0]

        if len(bins) < self.n_gram:
            print(f"[ERROR] Not enough bins for N-gram encoding ({len(bins)}/{self.n_gram}). Skipping window...")
            return None

        return self._process_ngrams(bins)

    def _process_ngrams(self, bins):
        """Creates N-grams and bundles them together for final encoding."""
        window_hv = torch.zeros(self.dims, device=self.device)

        for i in range(len(bins) - self.n_gram + 1):
            gram_hvs = [self.encode_bin(bin_data, idx) for bin_data, idx in bins[i:i + self.n_gram]]
            gram_hvs = [hv for hv in gram_hvs if hv is not None]

            if len(gram_hvs) == 0:
                print(f"[WARNING] Skipping empty N-gram at index {i}")
                continue

            # Bind the N-gram sequence together to encode temporal order
            gram_hv = torchhd.multibind(torch.stack(gram_hvs))
            window_hv = torchhd.bundle(window_hv, gram_hv)

        return torchhd.normalize(window_hv) if window_hv.norm() > 0 else None

    def process_windows(self, full_events, class_id):
        """Splits the event sequence into windows, encodes them, and returns a list of encoded hypervectors."""
        event_hvs = []
        total_events = len(full_events)
        first_t = full_events[0][0] if total_events > 0 else 0
        last_t = full_events[-1][0] if total_events > 0 else 0
        total_duration = last_t - first_t if total_events > 0 else 0

        expected_windows = max(1, (total_duration - self.OVERLAP_MS) // (self.WINDOW_SIZE_MS - self.OVERLAP_MS) + 1)

        print(f"\n[INFO] Encoding Sample | Class: {class_id} | Total Events: {total_events}")
        print(f"      - First Timestamp: {first_t}, Last Timestamp: {last_t}")
        print(f"      - Total Duration: {total_duration} ms")
        print(f"      - Expected Windows: {expected_windows}")

        # Sequentially number bins to prevent gaps
        sorted_time_stamps = sorted(set([t // self.time_subwindow for t, _, _, _ in full_events]))
        time_bin_map = {t: i + 1 for i, t in enumerate(sorted_time_stamps)}

        # Assign events to time bins
        temporal_dict = defaultdict(list)
        for t, x, y, polarity in full_events:
            time_bin = time_bin_map[t // self.time_subwindow]
            temporal_dict[time_bin].append((t, x, y, polarity))

        sorted_bins = sorted(temporal_dict.keys())
        window_events = [temporal_dict[b] for b in sorted_bins]

        for start_idx in range(0, len(window_events) - self.n_gram + 1, self.n_gram):
            window_hv = self.encode_window(window_events[start_idx:start_idx + self.n_gram])
            if window_hv is not None:
                event_hvs.append(window_hv)

        print(f"[INFO] Sample {class_id} - Created: {len(event_hvs)} windows\n")
        return event_hvs
