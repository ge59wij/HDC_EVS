
import torch
import torchhd
from collections import Counter
from BASE_HIST import HDHypervectorGenerators


# Encode training dataset
from collections import defaultdict

class HISTEncoder:
    def __init__(self, height, width, dims, device, window_size, n_gram,
                 threshold, method_encoding, K, debug):
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
        self.method_encoding = method_encoding
        self.BACKGROUND_LABEL = 404

        # Create hypervector generator with selected encoding method
        self.hv_gen = HDHypervectorGenerators(
            height, width, dims, device, threshold,
            window_size=window_size, n_gram=n_gram,
            method_encoding=method_encoding, debug=debug
        )
        self.time_hvs = self.hv_gen.time_hvs

        if debug:
            print(f"Initialized HISTEncoder with {method_encoding} encoding")
            print(f"Parameters: dims={dims}, window_size={window_size}, n_gram={n_gram}, threshold={threshold}")


    def encode_bin(self, bin_data, time_idx):
        """
        Encodes a single time bin with optimized spatial bundling before polarity binding.

        Args:
            bin_data (torch.Tensor): Event data for a single bin [2, H, W]
            time_idx (int): Time index within the window

        Returns:
            torch.Tensor: Normalized hypervector for the bin, or None if empty.
        """
        temporal_dict = defaultdict(list)  # Store spatial hypervectors per bin **without bundling yet**

        # **Step 1 & 2: Gather all spatial hypervectors (ON & OFF in one loop)**
        for polarity, hv_polarity in zip([1, 0], [self.hv_gen.H_I_on, self.hv_gen.H_I_off]):
            event_map = bin_data[polarity]  # Select ON or OFF event grid
            condition = event_map >= self.threshold  # Filter active pixels

            if torch.any(condition):  # Check if any events exist
                y_idx, x_idx = torch.where(condition)
                for x, y in zip(x_idx, y_idx):
                    P_xy = self.hv_gen.get_pos_hv(x, y)  # Get spatial HV
                    H_spatial = torchhd.bind(P_xy, hv_polarity)  # Bind with polarity HV
                    temporal_dict[time_idx].append(H_spatial)  # Store without bundling yet

        # **Step 3: Ensure there's something to encode**
        if time_idx not in temporal_dict or len(temporal_dict[time_idx]) == 0:
            return None  # If no events, return None

        # **Step 4: Bundle all stored spatial hypervectors ONCE per bin**
        spatial_hvs = torch.stack(temporal_dict[time_idx])  # Stack them
        bin_hv = torchhd.normalize(torchhd.multibundle(spatial_hvs))  # Bundle once

        # **Step 5: Apply Permutation to the Spatial Encoding (FIXED)**
        if self.method_encoding == "eventhd_timepermutation":
            encoded_bin = torchhd.permute(bin_hv, shifts=time_idx)  # Shift spatial HV directly
        elif self.method_encoding == "eventhd_timeinterpolation":
            encoded_bin = torchhd.bind(bin_hv, self.hv_gen.get_time_hv(time_idx))
        else:
            raise ValueError(f"Unsupported time encoding method: {self.method_encoding}")

        # **Step 6: Debugging Info**
        if self.debug and len(temporal_dict[time_idx]) > 0:
            print(f"  [DEBUG BIN] Time {time_idx}: Events={len(temporal_dict[time_idx])}")

        return torchhd.normalize(encoded_bin)  # Return final encoded vector

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
        gesture_bins = []  # Extract valid gesture bins
        valid_labels = []  # Track valid labels for debugging

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

        if self.hv_gen.method_encoding in ["eventhd_timepermutation",
                                           "eventhd_timeinterpolation"]:  # skip n-grams if EventHD is selected

            window_hv = torch.zeros(self.dims, device=self.device)
            for bin_data, time_idx in gesture_bins:
                bin_hv = self.encode_bin(bin_data, time_idx)
                if self.method_encoding == "eventhd_timepermutation":
                    permuted_hv = torchhd.permute(bin_hv, shifts=time_idx)  # Apply permutation for time
                    window_hv = torchhd.bundle(window_hv, permuted_hv)
                elif self.method_encoding == "eventhd_timeinterpolation":
                    window_hv = torchhd.bundle(window_hv, bin_hv)

            return torchhd.normalize(window_hv)  # if window_hv.norm() > 0 else None

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

            # Encode each bin in the n-gram, skipping None values
            gram_hvs = []
            for bin, idx in gram:
                bin_hv = self.encode_bin(bin, idx)
                if bin_hv is not None:  # Skip empty bins
                    gram_hvs.append(bin_hv)

            # If no valid bins in the n-gram, skip it
            if len(gram_hvs) == 0:
                if self.debug:
                    print(f"  [DEBUG NGRAM] Skipping n-gram {i + 1} (no valid bins)")
                continue

            # Combine valid bins with multibind
            gram_hv = torchhd.multibind(torch.stack(gram_hvs))

            # Bundle the n-gram hypervector with the window hypervector
            window_hv = torchhd.bundle(window_hv, gram_hv)

            if self.debug:
                print(f"  [DEBUG NGRAM] Processing n-gram {i + 1}/{len(bins) - self.n_gram + 1}")

        # Normalize the window hypervector if it has non-zero norm
        if window_hv.norm() > 0:
            return torchhd.normalize(window_hv)
        else:
            return None  # Return None if no valid bins were processed

    """
    def encode_bin(self, bin_data, time_idx):

        '''Encodes a single time bin with bundled ON/OFF events before polarity binding.

        Args:
            bin_data (torch.Tensor): Event data for a single bin [2, H, W]
            time_idx (int): Time index within the window

        Returns:
            torch.Tensor: Normalized hypervector for the bin, or None if empty.'''
        on_events = bin_data[1]  # ON polarity events
        off_events = bin_data[0]  # OFF polarity events

        # Initialize ON and OFF position hypervectors as None
        on_pos_hv = None
        off_pos_hv = None

        # Corrected ON events handling
        on_condition = on_events >= self.threshold
        if torch.any(on_condition):
            y_on, x_on = torch.where(on_condition)
        else:
            y_on = torch.tensor([], dtype=torch.long, device=self.device)
            x_on = torch.tensor([], dtype=torch.long, device=self.device)

        # Corrected OFF events handling
        off_condition = off_events >= self.threshold
        if torch.any(off_condition):
            y_off, x_off = torch.where(off_condition)
        else:
            y_off = torch.tensor([], dtype=torch.long, device=self.device)
            x_off = torch.tensor([], dtype=torch.long, device=self.device)

        # If no events are present, return None
        if len(x_on) == 0 and len(x_off) == 0:
            return None

        if self.method_encoding in ["thermometer", "linear"]:
            if len(x_on) > 0:
                on_pos_hv = self.hv_gen.get_pos_hv(x_on[0], y_on[0])  # Start with first position HV
                for x, y in zip(x_on[1:], y_on[1:]):  # Skip first, bundle rest
                    on_pos_hv = torchhd.bundle(on_pos_hv, self.hv_gen.get_pos_hv(x, y))

            if len(x_off) > 0:
                off_pos_hv = self.hv_gen.get_pos_hv(x_off[0], y_off[0])
                for x, y in zip(x_off[1:], y_off[1:]):  # Skip first, bundle rest
                    off_pos_hv = torchhd.bundle(off_pos_hv, self.hv_gen.get_pos_hv(x, y))

            # Ensure HVs are initialized, else use zero HV
            on_pos_hv = on_pos_hv if on_pos_hv is not None else torch.zeros(self.dims, device=self.device)
            off_pos_hv = off_pos_hv if off_pos_hv is not None else torch.zeros(self.dims, device=self.device)

            # Bind with polarity HVs
            on_hv = torchhd.bind(on_pos_hv, self.hv_gen.H_I_on)
            off_hv = torchhd.bind(off_pos_hv, self.hv_gen.H_I_off)

            # Bundle ON and OFF hypervectors
            encoded_bin = torchhd.bundle(on_hv, off_hv)

        elif self.method_encoding in ["eventhd_timepermutation", "eventhd_timeinterpolation"]:
            temporal_dict = {}

            for x, y in zip(x_on, y_on):
                P_xy = self.hv_gen.get_pos_hv(x, y)
                H_spatial = torchhd.bind(P_xy, self.hv_gen.H_I_on)
                temporal_dict[time_idx] = torchhd.bundle(
                    temporal_dict.get(time_idx, torch.zeros(self.dims, device=self.device)), H_spatial)

            for x, y in zip(x_off, y_off):
                P_xy = self.hv_gen.get_pos_hv(x, y)
                H_spatial = torchhd.bind(P_xy, self.hv_gen.H_I_off)
                temporal_dict[time_idx] = torchhd.bundle(
                    temporal_dict.get(time_idx, torch.zeros(self.dims, device=self.device)), H_spatial)

            # Ensure we have at least one valid encoded HV
            if time_idx not in temporal_dict:
                return None

            bin_hv = temporal_dict[time_idx]

            if self.method_encoding == "eventhd_timepermutation":
                encoded_bin = torchhd.bind(bin_hv, torchhd.permute(self.hv_gen.base_time_HV, shifts=time_idx))
            elif self.method_encoding == "eventhd_timeinterpolation":
                encoded_bin = torchhd.bind(bin_hv, self.hv_gen.get_time_hv(time_idx))
            else:
                raise ValueError(f"Unsupported time encoding method: {self.method_encoding}")

        if self.debug and (len(x_on) > 0 or len(x_off) > 0):
            print(f"  [DEBUG BIN] Time {time_idx}: ON events: {len(x_on)}, OFF events: {len(x_off)}")

        return torchhd.normalize(encoded_bin) 
        """
