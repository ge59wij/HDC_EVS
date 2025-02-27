from TrainValTest import *
import torchhd.utils
from BASE_HIST import HDHypervectorGenerators
import numpy as np
####qwe probbaly have to ignore event count 0, maybe unlike other all pixels have a value in the tensor.
import torch
import torchhd
import numpy as np

class HISTEncoder:
    def __init__(self, height, width, dims, device, threshold, window_size, stride):
        self.height = height
        self.width = width
        self.dims = dims
        self.threshhold = threshold
        self.device = torch.device(device)
        self.window_size = window_size  # N-bins per encoding window
        self.stride = stride  # Overlap
        self.hv_gen = HDHypervectorGenerators(height, width, dims, device, threshold, window_size, stride)

    def encode_bin(self, bin_data, bin_idx):
        """Encodes a single bin"""
        print(f"  [DEBUG] Encoding BIN {bin_idx}")

        bin_hv = torch.zeros(self.dims, device=self.device)
        on_events = bin_data[0]  # ON-event matrix
        off_events = bin_data[1]  # OFF-event matrix

        for y in range(self.height):
            for x in range(self.width):
                if on_events[y, x] >= self.hv_gen.threshold:
                    #print(f"    ➤ ON Event at ({x},{y})")
                    pos_hv = self.hv_gen.get_pos_hv(x, y)
                    pixel_hv = torchhd.bind(pos_hv, self.hv_gen.H_I_on)
                    bin_hv = torchhd.bundle(bin_hv, pixel_hv)

                if off_events[y, x] >= self.hv_gen.threshold:
                    #print(f"    ➤ OFF Event at ({x},{y})")
                    pos_hv = self.hv_gen.get_pos_hv(x, y)
                    pixel_hv = torchhd.bind(pos_hv, self.hv_gen.H_I_off)
                    bin_hv = torchhd.bundle(bin_hv, pixel_hv)

        # Bind with time hypervector
        bin_hv = torchhd.bind(bin_hv, self.hv_gen.get_time_hv(bin_idx))
        return torchhd.normalize(bin_hv)

    def encode_window(self, event_data, start_bin, bundle_size):
        """Encodes a sliding window of N-bins into one hypervector using multiset instead of bundling."""
        print(f"[DEBUG] Encoding WINDOW {start_bin}-{start_bin + self.window_size}")

        end_bin = min(start_bin + self.window_size, event_data.shape[0])
        bin_hvs = []  # Collect bin vectors instead of initializing to zero

        for i, bin_idx in enumerate(range(start_bin, end_bin)):
            bin_hv = self.encode_bin(event_data[bin_idx], bin_idx)
            bin_hvs.append(bin_hv)  # Store instead of bundling

        # Stack and multiset instead of incremental bundling
        if len(bin_hvs) > 0:
            stacked_bins = torch.stack(bin_hvs)
            window_hv = torchhd.multiset(stacked_bins)  # Multiset operation
            window_hv = torchhd.normalize(window_hv)
        else:
            window_hv = torch.zeros(self.dims, device=self.device)  # If empty, return zero vector

        print(f"[DEBUG] Finished encoding window {start_bin}-{end_bin}")
        return window_hv


