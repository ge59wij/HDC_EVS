import torch
import torchhd
import numpy as np
from collections import Counter
from BASE_HIST import HDHypervectorGenerators


class HISTEncoder:
    def __init__(self, height, width, dims, device, window_size, n_gram,
                 threshold=1 / 16, method_encoding="thermometer", levels=4, K=6 ,debug=True):
        """
        Encodes event data into hypervectors for HDC-based gesture recognition.

        Args:
            height (int): Image height
            width (int): Image width
            dims (int): Hypervector dimensionality
            device (str): Computation device ('cpu' or 'cuda')
            window_size (int): Number of bins per window
            n_gram (int): Size of temporal n-grams
            threshold (float): Event count threshold for noise filtering
            spatial_encoding (str): Encoding method - "thermometer" or "linear"
            levels (int): Number of levels for thermometer encoding
            debug (bool): Enable debug prints
        """
        self.height = height
        self.width = width
        self.dims = dims
        self.threshold = threshold
        self.device = torch.device(device)
        self.window_size = window_size
        self.n_gram = n_gram
        self.debug = debug
        self.K = K
        self.BACKGROUND_LABEL = 404

        # Create hypervector generator with selected encoding method
        self.hv_gen = HDHypervectorGenerators(
            height, width, dims, device, threshold,
            window_size=window_size, n_gram=n_gram,
            method_encoding=method_encoding, levels=levels, debug=debug
        )
        self.time_hvs = self.hv_gen.time_hvs

        if debug:
            print(f"Initialized HISTEncoder with {method_encoding} encoding")
            print(f"Parameters: dims={dims}, window_size={window_size}, n_gram={n_gram}, threshold={threshold}")

    def encode_bin(self, bin_data, time_idx):
        """
        Encodes a single time bin with bundled ON/OFF events before polarity binding.

        Args:
            bin_data (torch.Tensor): Event data for a single bin [2, H, W]
            time_idx (int): Time index within the window

        Returns:
            torch.Tensor: Normalized hypervector for the bin
        """
        on_events = bin_data[0]  # ON polarity events
        off_events = bin_data[1]  # OFF polarity events

        # Find active ON event positions and bundle their hypervectors
        #x_on = torch.where(on_events >= self.threshold)
        y_on, x_on = torch.where(on_events >= self.threshold)

        if len(x_on) > 0:
            on_pos_hv = self.hv_gen.get_pos_hv(x_on[0], y_on[0])  # Start with first position HV
            for x, y in zip(x_on[1:], y_on[1:]):  # Skip first, bundle rest
                on_pos_hv = torchhd.bundle(on_pos_hv, self.hv_gen.get_pos_hv(x, y))
        #else:
        #    on_pos_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)  # Random HV if no ON events

        # Find active OFF event positions and bundle their hypervectors
        y_off, x_off = torch.where(off_events >= self.threshold)
        if len(x_off) > 0:
            off_pos_hv = self.hv_gen.get_pos_hv(x_off[0], y_off[0])  # Start with first position HV
            for x, y in zip(x_off[1:], y_off[1:]):  # Skip first, bundle rest
                off_pos_hv = torchhd.bundle(off_pos_hv, self.hv_gen.get_pos_hv(x, y))
        #else:
        #    off_pos_hv = torchhd.random(1, self.dims, "MAP", device=self.device).squeeze(0)  # Random HV if no OFF events

        # Create event polarity hypervectors
        on_hv = torchhd.bind(on_pos_hv, self.hv_gen.H_I_on)
        off_hv = torchhd.bind(off_pos_hv, self.hv_gen.H_I_off)

        # Bundle ON and OFF polarity hypervectors
        bin_hv = torchhd.bundle(on_hv, off_hv)

        # Bind with time hypervector
        #encoded_bin = torchhd.bind(bin_hv, self.time_hvs[time_idx % self.n_gram])
        encoded_bin = torchhd.bind(bin_hv, torchhd.permute(self.time_hvs[0], shifts=time_idx))

        if self.debug and (len(x_on) > 0 or len(x_off) > 0):
            print(f"  [DEBUG BIN] Time {time_idx}: ON events: {len(x_on)}, OFF events: {len(x_off)}")

        return torchhd.normalize(encoded_bin)

    def encode_window(self, window_data, window_labels):
        """
        Encodes a sliding window of event data.
        Skips background bins (label=404)
            - Uses n-grams only for linear and thermometer.
        - Uses direct permutation for eventhd_timepermutation.
        Args:
            window_data (torch.Tensor): Event data for window [T, 2, H, W]
            window_labels (torch.Tensor): Labels for each bin in window
        Returns:
            torch.Tensor: Encoded hv for the window, or None if no valid gesture
        """
        gesture_bins = []         # Extract valid gesture bins
        valid_labels = []         # Track valid labels for debugging

        for idx, (bin_data, label) in enumerate(zip(window_data, window_labels)):
            if label == self.BACKGROUND_LABEL:
                continue  # Skip background bins
            gesture_bins.append((bin_data, idx))
            valid_labels.append(label.item())

        if self.debug:
            if len(gesture_bins) > 0:
                label_counts = Counter(valid_labels)
                print(f"[DEBUG WINDOW] Found {len(gesture_bins)} valid bins with labels: {dict(label_counts)}")
            else:
                print("[DEBUG WINDOW] No valid gesture bins found in window")

        if len(gesture_bins) == 0:
            return None  # No valid gesture bins in this window

        if self.hv_gen.method_encoding == "eventhd_timepermutation":   #skip n-grams if EventHD is selected
            window_hv = torch.zeros(self.dims, device=self.device)
            for bin_data, time_idx in gesture_bins:
                bin_hv = self.encode_bin(bin_data, time_idx)
                permuted_hv = torchhd.permute(bin_hv, shifts=time_idx)  # Apply permutation for time
                window_hv = torchhd.bundle(window_hv, permuted_hv)
            return torchhd.normalize(window_hv) if window_hv.norm() > 0 else None


        # Process valid gesture bins with n-grams  **Linear & Thermometer
        gesture_hv = self._process_ngrams(gesture_bins)
        return gesture_hv

    def _process_ngrams(self, bins):
        """
        Applies n-gram encoding to temporal bins.

        Args:
            bins (list): List of (bin_data, time_idx) tuples

        Returns:
            torch.Tensor: Encoded hypervector for the ngrams, or None if invalid
        """
        if len(bins) == 0:
            return None  # No valid gesture bins in window

        # Check if we have enough bins for n-gram
        if len(bins) < self.n_gram:
            if self.debug:
                print(f"[WARNING] Not enough bins for full n-gram (Need {self.n_gram}, got {len(bins)})")
            return None  # Skip this window

        window_hv = torch.zeros(self.dims, device=self.device)

        # Apply n-gram encoding
        for i in range(len(bins) - self.n_gram + 1):
            gram = bins[i:i + self.n_gram]
            # Encode each bin in the n-gram and combine with multibind
            gram_hv = torchhd.multibind(torch.stack([self.encode_bin(bin, idx) for bin, idx in gram]))
            # Bundle the n-gram hypervector with the window hypervector
            window_hv = torchhd.bundle(window_hv, gram_hv)

            if self.debug:
                print(f"  [DEBUG NGRAM] Processing n-gram {i + 1}/{len(bins) - self.n_gram + 1}")

        # Normalize the window hypervector if non-zero
        return torchhd.normalize(window_hv) if window_hv.norm() > 0 else None