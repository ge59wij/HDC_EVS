import h5py
import numpy as np
import torch
import os
import time


class GestureDataset:
    def __init__(self, h5_folder, load_npy=False):
        self.h5_files = sorted([f for f in os.listdir(h5_folder) if f.endswith(".h5")])
        self.h5_folder = h5_folder
        self.load_npy = load_npy

    def __getitem__(self, idx):
        h5_path = os.path.join(self.h5_folder, self.h5_files[idx])
        npy_path = h5_path.replace(".h5", "_bbox.npy")

        start_time = time.time()

        # Load HDF5 data
        with h5py.File(h5_path, "r") as f:
            event_data = torch.tensor(f["data"][:], dtype=torch.float32)

        h5_time = time.time() - start_time
        bbox_time = 0
        filtered_bboxes = None
        bbox_data = None

        if self.load_npy and os.path.exists(npy_path):
            npy_start_time = time.time()
            bbox_data = np.load(npy_path, allow_pickle=True)

            if bbox_data.dtype.names is not None:
                bbox_data = np.column_stack([bbox_data[name] for name in bbox_data.dtype.names])

            if bbox_data.ndim == 1 and bbox_data.shape[0] % 8 == 0:
                bbox_data = bbox_data.reshape(-1, 8)
            elif bbox_data.ndim == 1:
                print(f"Warning: Unexpected bbox_data shape {bbox_data.shape}. Reshaping might fail.")

            start_ts, end_ts = event_data.shape[0], event_data.shape[0]
            filtered_bboxes = bbox_data[(bbox_data[:, 0] >= start_ts) & (bbox_data[:, 0] <= end_ts)]
            bbox_time = time.time() - npy_start_time

        total_time = time.time() - start_time

        if bbox_data is not None:
            print(f"Loaded bbox_data shape: {bbox_data.shape}")

        return event_data, filtered_bboxes, h5_time, bbox_time, total_time
    def __len__(self):
        return len(self.h5_files)


H5_FOLDER = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/train"
dataset_with_npy = GestureDataset(H5_FOLDER, load_npy=True)
dataset_without_npy = GestureDataset(H5_FOLDER, load_npy=False)

n_samples = 10
times_with_npy = []
times_without_npy = []

for i in range(n_samples):
    _, _, h5_time, bbox_time, total_time_with_npy = dataset_with_npy[i]
    _, _, h5_time_only, _, total_time_without_npy = dataset_without_npy[i]

    times_with_npy.append(total_time_with_npy)
    times_without_npy.append(total_time_without_npy)

print(f"\n Average HDF5 Load Time: {np.mean(times_without_npy):.4f} sec/sample")
print(f" Average HDF5 + NPY Load Time: {np.mean(times_with_npy):.4f} sec/sample")
print(f" Extra Time for Loading NPY: {np.mean(times_with_npy) - np.mean(times_without_npy):.4f} sec/sample")
