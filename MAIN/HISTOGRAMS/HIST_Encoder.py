import torch
import torchhd
from collections import Counter, defaultdict
from BASE_HIST import HDHypervectorGenerators

class HISTEncoder:
    def __init__(self, height, width, dims, device, window_size, n_gram,
                 threshold, method_encoding, K, debug, weighting):
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
        self.weighting = weighting  # Enable weighted encoding

        self.hv_gen = HDHypervectorGenerators(
            height, width, dims, device, threshold,
            window_size=window_size, n_gram=n_gram,
            method_encoding=method_encoding, K=K, debug=debug
        )

        if method_encoding in ["eventhd_timeinterpolation", "stem_hd"]:
            self.time_hvs = self.hv_gen.time_hvs

        if debug:
            print(f"Initialized HISTEncoder with {method_encoding} encoding")
            print(f"Parameters: dims={dims}, window_size={window_size}, n_gram={n_gram}, threshold={threshold}, weighting={self.weighting}")

    def encode_bin(self, bin_data, time_idx):
        """
        Encodes a single time bin with spatial encoding before applying time transformation.

        Args:
            bin_data (torch.Tensor): Event data for a single bin [2, H, W]
            time_idx (int): Time index within the window

        Returns:
            torch.Tensor: Encoded spatial hypervector for the bin, or zero vector if empty.
        """
        spatial_hv_per_polarity = {}  # Store aggregated spatial HVs for ON/OFF

        for polarity, hv_polarity in zip([1, 0], [self.hv_gen.H_I_on, self.hv_gen.H_I_off]):
            event_map = bin_data[polarity]  # Select ON or OFF event grid
            indices = torch.nonzero(event_map, as_tuple=True)  # Get active event pixel indices

            if indices[0].numel() == 0:  # Skip if no events exist for this polarity
                continue

            max_events = event_map.max().item()  # Normalize event count

            #  **Step 1: Get all spatial hypervectors for active pixels**
            pos_hvs = self.hv_gen.get_pos_hv(indices[1], indices[0])  # Get hypervectors for active pixels

            # **Step 2: Apply weighting if enabled**
            if self.weighting and max_events > 0:
                weights = event_map[indices] / max_events  # Normalize event count weight
                pos_hvs *= weights.unsqueeze(-1)  # Apply weights

            #  **Step 3: Bundle all positions within this polarity into one spatial HV**
            spatial_hv = torchhd.multiset(pos_hvs)

            #  **Step 4: Bind spatial HV with the polarity hypervector**
            spatial_hv_per_polarity[polarity] = torchhd.bind(spatial_hv, hv_polarity)

        #  **Step 5: Handle empty bins**
        if not spatial_hv_per_polarity:
            if self.debug:
                print(f"[ERROR] No valid events for bin {time_idx}!")
            return torch.zeros(self.dims, device=self.device)

        # ✅ **Step 6: Bundle ON and OFF polarities together to get a single hypervector per bin**
        bin_hv = torchhd.bundle(
            spatial_hv_per_polarity.get(1, torch.zeros(self.dims, device=self.device)),  # ON events
            spatial_hv_per_polarity.get(0, torch.zeros(self.dims, device=self.device))   # OFF events
        )

        return bin_hv  # Return the final per-bin hypervector

    def encode_window(self, window_data, window_labels):
        """
        Encodes a sliding window of event data.

        Args:
            window_data (torch.Tensor): Event data for window [T, 2, H, W]
            window_labels (torch.Tensor): Labels for each bin in window

        Returns:
            torch.Tensor: Encoded hypervector for the window, or None if no valid gesture.
        """
        gesture_bins = [(bin_data, idx) for idx, (bin_data, label) in enumerate(zip(window_data, window_labels))
                        if label != self.BACKGROUND_LABEL]

        if not gesture_bins:
            if self.debug:
                print("[DEBUG WINDOW] No valid gesture bins found in window")
            return None  # No valid gesture bins

        if self.debug:
            print(f"[DEBUG WINDOW] Found {len(gesture_bins)} valid bins")

        #  **Step 1: Encode all bins first (Spatial Encoding)**
        bin_hvs = [(self.encode_bin(bin_data, time_idx), time_idx) for bin_data, time_idx in gesture_bins]
        bin_hvs = [(hv, idx) for hv, idx in bin_hvs if hv is not None]

        if not bin_hvs:
            return None  # No valid bins to encode

        # **Step 2: Apply Time Encoding Based on Method**
        if self.method_encoding == "eventhd_timepermutation":
            # **Sort bins by time index** for consistent ordering
            bin_hvs.sort(key=lambda x: x[1])

            permuted_time_hvs = []
            for index, (hv, _) in enumerate(bin_hvs):
                permuted_hv = torchhd.permute(hv, shifts=index)  # **Permute each spatial hypervector individually**
                permuted_time_hvs.append(permuted_hv.detach())

            #  **Step 3: Bundle all permuted hypervectors together**
            window_hv = torchhd.multibundle(torch.stack(permuted_time_hvs))

        else:
            # Convert list of tuples to tensors
            bin_hvs_tensors = torch.stack([hv for hv, _ in bin_hvs])
            time_indices = torch.tensor([idx for _, idx in bin_hvs], device=bin_hvs_tensors.device)

            if self.method_encoding in ["eventhd_timeinterpolation", "stem_hd"]:
                time_hvs = self.hv_gen.get_time_hv(time_indices)
                interpolated_hvs = torchhd.bind(bin_hvs_tensors, time_hvs)
                window_hv = torchhd.multiset(interpolated_hvs)

            elif self.method_encoding in ["thermometer", "linear", "kxk_ngram"]:
                window_hv = self._process_ngrams(bin_hvs)

            else:
                raise ValueError(f"Unsupported encoding method: {self.method_encoding}")

        return torchhd.normalize(window_hv)

    def _process_ngrams(self, bins):
        """
        Applies n-gram encoding to temporal bins using torchhd.ngrams() and bundles results.

        Args:
            bins (list): List of (bin_hv, time_idx) tuples

        Returns:
            torch.Tensor: Encoded hypervector for the n-grams, or None if invalid.
        """
        if len(bins) < self.n_gram:
            if self.debug:
                print(f"[WARNING] Not enough bins for full n-gram (Need {self.n_gram}, got {len(bins)})")
            return None  # Skip if not enough bins

        bin_hvs_tensors = torch.stack([hv for hv, _ in bins])

        # Apply N-grams encoding
        ngram_hvs = torchhd.ngrams(bin_hvs_tensors, n=self.n_gram)

        # Handle case where only a single N-gram is returned
        window_hv = ngram_hvs if ngram_hvs.dim() == 1 else torchhd.multiset(ngram_hvs)

        return window_hv
