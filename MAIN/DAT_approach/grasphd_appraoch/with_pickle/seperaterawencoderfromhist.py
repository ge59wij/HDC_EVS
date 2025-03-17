import torch
import torchhd
import numpy as np
from collections import defaultdict
import gc
from grasphdencoding_seedhvs import seedEncoder  # Import the existing seedEncoder


np.set_printoptions(suppress=True, precision=8)

class Raw_events_HDEncoder_Enhanced:
    def __init__(self, height, width, dims, time_subwindow, k, device, max_time, time_method, WINDOW_SIZE_MS, OVERLAP_MS):
        """Initialize encoder with spatial and temporal encoding parameters."""
        print("Initializing Raw Events HD Encoder...")
        self.n_gram = 5
        self.height = height
        self.width = width
        self.dims = dims
        self.k = k
        self.time_subwindow = time_subwindow
        self.device = torch.device(device)
        self.time_method = time_method
        self.WINDOW_SIZE_MS = WINDOW_SIZE_MS
        self.OVERLAP_MS = OVERLAP_MS
        self.debug = True

        # **Polarity Hypervectors (Seed HVs)**
        self.H_I_on = torchhd.random(1, dims, "MAP", device=self.device).squeeze(0)
        self.H_I_off = -self.H_I_on

        # **Caches (Only Seed HVs Stay)**
        self.pixel_hvs = {}
        self.time_hv_cache = {}  # Stores precomputed time hypervectors
        if time_method == "kxk_ngram":
            self.seed_encoder = seedEncoder(height, width, dims, k, time_subwindow, device, max_time, time_method,
                                            WINDOW_SIZE_MS, OVERLAP_MS)

        # **Generate Spatial Encoding Hypervectors**
        if time_method == "linear":
            self.HV_x = self._generate_linear_axis_hvs(self.width)
            self.HV_y = self._generate_linear_axis_hvs(self.height)
            self._generate_pixel_hvs()
        elif time_method == "thermometer":
            self.HV_x = torchhd.thermometer(self.width, self.dims, "MAP")
            self.HV_y = torchhd.thermometer(self.height, self.dims, "MAP")
            self._generate_pixel_hvs()

    def get_pos_hv(self, x, y):
        """Retrieve spatial hypervector for a given position with batch processing."""
        key = list(zip(x.tolist(), y.tolist()))
        hvs = []
        for k in key:
            if k not in self.pixel_hvs:
                self.pixel_hvs[k] = torchhd.bind(self.HV_x[k[0]], self.HV_y[k[1]])
            hvs.append(self.pixel_hvs[k])
        return torch.stack(hvs)

    def _generate_linear_axis_hvs(self, size):
        """Generate structured hypervectors for spatial encoding using linear interpolation."""
        flip_bits = self.dims // (4 * (size - 1))
        base_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)
        hvs = [base_hv.clone()]

        for _ in range(1, size):
            new_hv = hvs[-1].clone()
            flip_indices = torch.randperm(self.dims)[:flip_bits]
            new_hv[flip_indices] = -new_hv[flip_indices]
            hvs.append(new_hv)

        return torch.stack(hvs)

    def _generate_pixel_hvs(self):
        """Precompute spatial encoding hypervectors for each pixel (cached in dictionary)."""
        for y in range(self.height):
            for x in range(self.width):
                self.pixel_hvs[(x, y)] = torchhd.bind(self.HV_x[x], self.HV_y[y])

    def encode_bin(self, bin_data, time_idx):
        """Encodes a single bin by bundling spatial hypervectors."""
        if len(bin_data) == 0:
            return None  # Skip empty bins

        bin_data = torch.tensor(np.array(bin_data, dtype=np.int32), device=self.device)
        x, y, polarity = bin_data[:, 1], bin_data[:, 2], bin_data[:, 3]

        if self.time_method == "kxk_ngram":
            #  Use KxK position encoding from seedEncoder
            pos_hvs = torch.stack([self.seed_encoder.get_position_hv(int(x_i), int(y_i)) for x_i, y_i in zip(x, y)])
        else:
            pos_hvs = self.get_pos_hv(x, y)

        H_I_on_expanded = self.H_I_on.expand(len(polarity), self.dims)
        H_I_off_expanded = self.H_I_off.expand(len(polarity), self.dims)

        pol_hvs = torch.where(polarity.unsqueeze(-1) == 1, H_I_on_expanded, H_I_off_expanded)

        spatial_hvs = torchhd.bind(pos_hvs, pol_hvs)

        return (torchhd.multibundle(spatial_hvs))

    def encode_window(self, window_data):
        """Encodes an entire window using N-grams."""
        bins = [(bin_data, idx) for idx, bin_data in enumerate(window_data) if len(bin_data) > 0]
        if len(bins) < self.n_gram:
            return None  # Skip if not enough bins for an N-gram
        return self._process_ngrams(bins)

    def _process_ngrams(self, bins):
        """
        Applies n-gram encoding to temporal bins using torchhd.ngrams() and bundles results for the entire window.

        Args:
            bins (list): List of (bin_data, time_idx) tuples

        Returns:
            torch.Tensor: Encoded hypervector for the n-grams, or None if invalid
        """
        if len(bins) < self.n_gram:
            if self.debug:
                print(f"[WARNING] Not enough bins for full n-gram (Need {self.n_gram}, got {len(bins)})")
            return None  # Skip if not enough bins

        # Encode all bins first
        bin_hvs = []
        for bin_data, idx in bins:
            bin_hv = self.encode_bin(bin_data, idx)
            if bin_hv is not None:
                bin_hvs.append(bin_hv)
            else:
                if self.debug:
                    print(f"[DEBUG NGRAM] Skipping empty bin at time {idx}")

        if len(bin_hvs) < self.n_gram:
            if self.debug:
                print(f"[WARNING] Skipping window: Only {len(bin_hvs)} valid bins, needs at least {self.n_gram}")
            return None  # Skip incomplete N-grams

        #  Ensure `bin_hvs` is a tensor before passing to `ngrams()`
        stacked_hvs = torch.stack(bin_hvs)  # Shape: (T, dims)

        # Apply N-grams encoding
        ngram_hvs = torchhd.ngrams(stacked_hvs, n=self.n_gram)  # Expected shape: (T - n + 1, dims)

        # Fix: Handle case where only a single N-gram is returned
        if ngram_hvs.dim() == 1:  # If there’s only one hypervector
            window_hv = ngram_hvs  # No need to bundle
        else:
            window_hv = torchhd.multiset(ngram_hvs)  # Ensures proper bundling

        if self.debug:
            print(f"[DEBUG NGRAM] Processed {len(bin_hvs)} valid bins into {self.n_gram}-grams")
            print(f"[DEBUG NGRAM] ngram_hvs shape: {ngram_hvs.shape}")

        return torchhd.normalize(window_hv)

    def process_windows(self, full_events, class_id):
        """Splits event sequence into sliding windows, encodes them, and clears cache after each sample."""
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

        # Assign events to time bins
        temporal_dict = defaultdict(list)
        for t, x, y, polarity in full_events:
            time_bin = t // self.time_subwindow
            temporal_dict[time_bin].append((t, x, y, polarity))

        sorted_bins = sorted(temporal_dict.keys())
        window_events = [temporal_dict[b] for b in sorted_bins]

        for start_idx in range(0, len(window_events) - self.n_gram + 1, self.n_gram):
            window_hv = self.encode_window(window_events[start_idx:start_idx + self.n_gram])
            if window_hv is not None:
                event_hvs.append(window_hv)

            # **Clear caches after each window**
            self.clear_temporary_cache()

        print(f"[INFO] Sample {class_id} - Created: {len(event_hvs)} windows\n")
        return event_hvs

    def clear_temporary_cache(self):
        """Clears all temporary computed hypervectors while keeping essential seed hypervectors."""
        self.time_hv_cache.clear()  # Clear stored time hypervectors
        self.pixel_hvs.clear()  # Clear spatial hypervectors (except seed ones)
        gc.collect()

        print("[DEBUG] Cleared temporary caches after processing window.")
