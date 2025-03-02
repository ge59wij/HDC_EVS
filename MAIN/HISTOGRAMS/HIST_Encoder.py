from TrainValTest import *
import torchhd.utils
from BASE_HIST import HDHypervectorGenerators
import numpy as np
####qwe probbaly have to ignore event count 0, maybe unlike other all pixels have a value in the tensor.
import torch
import torchhd
import numpy as np

'''Spatial Encoder (Per Bin):
   ON/OFF events → Position-bound HVs
   Temporal Encoder (Per Window):
   n-Grams within window
   Skip 404 bins
   Sliding Window Generator:
   window_size + overlap parameters'''

class HISTEncoder:
    def __init__(self, height, width, dims, device, threshold, window_size, n_gram):
        self.height = height
        self.width = width
        self.dims = dims
        self.threshold = threshold
        self.device = torch.device(device)
        self.window_size = window_size  # N-bins per encoding window
        self.n_gram = n_gram  # temporal permutation
        self.hv_gen = HDHypervectorGenerators(height, width, dims, device, threshold, window_size=window_size, n_gram=n_gram)
        self.time_hvs = self.hv_gen.time_hvs
        self.BACKGROUND_LABEL = 404

    def encode_bin(self, bin_data, time_idx):
        """Encodes bin with bundled ON/OFF before polarity binding"""
        on_events = bin_data[0]
        off_events = bin_data[1]

        # Bundle all ON/OFF positions before binding
        on_pos_hv = torch.zeros(self.dims, device=self.device)
        off_pos_hv = torch.zeros(self.dims, device=self.device)
        #print(f"[DEBUG] Encoding bin at time index {time_idx}")
        #print(
        #    f"  - ON events count: {on_events.count_nonzero().item()}, OFF events count: {off_events.count_nonzero().item()}")

        y_on, x_on = torch.where(on_events >= self.threshold)
        for x, y in zip(x_on, y_on):
            on_pos_hv = torchhd.bundle(on_pos_hv, self.hv_gen.get_pos_hv(x, y))

        y_off, x_off = torch.where(off_events >= self.threshold)
        for x, y in zip(x_off, y_off):
            off_pos_hv = torchhd.bundle(off_pos_hv, self.hv_gen.get_pos_hv(x, y))

        # Bind once per polarity
        on_hv = torchhd.bind(on_pos_hv, self.hv_gen.H_I_on)
        off_hv = torchhd.bind(off_pos_hv, self.hv_gen.H_I_off)
        #print(f"  - ON HV Bundled: {on_pos_hv.shape}, OFF HV Bundled: {off_pos_hv.shape}")

        # Bundle polarities instead of binding
        bin_hv = torchhd.bundle(on_hv, off_hv)
        encoded_bin = torchhd.bind(bin_hv, self.time_hvs[time_idx % self.n_gram])
        #print(f"  - Final Encoded Bin HV: {bin_hv.shape}")
        return torchhd.normalize(encoded_bin)

    def encode_window(self, window_data, window_labels):
        """Encodes a sliding window with separate gesture/background vectors"""
        gesture_bins = []
        #background_bins = []

        for idx, (bin_data, label) in enumerate(zip(window_data, window_labels)):
            if label == self.BACKGROUND_LABEL:
                continue

            gesture_bins.append((bin_data, idx))
        #print(f"[DEBUG] Encoding Window: {len(gesture_bins)} valid gesture bins found")

        gesture_hv = self._process_ngrams(gesture_bins)
        #bg_hv = self._process_ngrams(background_bins)
        #print(f"[DEBUG] Encoded Window HV: {gesture_hv.shape if gesture_hv is not None else 'None'}")

        return gesture_hv  #, bg_hv

    def _process_ngrams(self, bins):
        """Applies n-gram encoding, ensuring valid gesture bins are encoded."""

        if len(bins) == 0:
            return None  # **No valid gesture bins in window**

        window_hv = torch.zeros(self.dims, device=self.device)

        # **Apply n-gram encoding, handling edge cases**
        if len(bins) < self.n_gram:
            print(
                f"[WARNING] Not enough bins for full n-gram (Needed {self.n_gram}, got {len(bins)}). Skipping these bins.")
            return None  # Skip this window

        for i in range(len(bins) - self.n_gram + 1):
            gram = bins[i:i + self.n_gram]
            gram_hv = torchhd.multibind(torch.stack([self.encode_bin(bin, idx) for bin, idx in gram]))
            window_hv = torchhd.bundle(window_hv, gram_hv)

        return torchhd.normalize(window_hv) if window_hv.norm() > 0 else None
