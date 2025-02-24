import h5py
'''
#.h5 file
h5_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_left_close_fast_sitting_recording_013_2021-09-14_14-18-22.h5"

with h5py.File(h5_path, "r") as h5_file:
    print("Keys in the HDF5 file:")


    def explore_hdf5_group(group, indent=0):
        for key in group.keys():
            print("  " * indent + f"- {key}: {type(group[key])}")
            if isinstance(group[key], h5py.Group):
                explore_hdf5_group(group[key], indent + 1)
            elif isinstance(group[key], h5py.Dataset):
                # Print dataset shape and dtype
                dataset = group[key]
                print("  " * (indent + 1) + f"Shape: {dataset.shape}, Dtype: {dataset.dtype}")


    explore_hdf5_group(h5_file)
print("sep")

with h5py.File("/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_left_close_fast_sitting_recording_013_2021-09-14_14-18-22.h5", "r") as f:
    # List all datasets in the file
    print("Datasets:", list(f.keys()))
    
    
    
    

    # Access the dataset
    data = f["data"]

    # Print dataset attributes
    print("Shape:", data.shape)
    print("Data type:", data.dtype)

    # Access a specific sample
    sample = data[32]
    print("Sample shape:", sample.shape)
import matplotlib.pyplot as plt
import numpy as np

plt.imshow(sample[0], cmap="gray")
plt.title("Sample 0 - Channel 0")
plt.show()


import h5py
import h5py
h5_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_left_close_fast_sitting_recording_013_2021-09-14_14-18-22.h5"
h5_path2 = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/scissors_right_close_fast_sitting_recording_003_2021-09-14_11-14-38.h5"

with h5py.File(h5_path, "r") as h5_file:
    # Print all datasets and groups in the file
    print("HDF5 Structure:")
    for key in h5_file.keys():
        print(f"- {key}: {type(h5_file[key])}")


    # Check for attributes or metadata
    for key in h5_file.keys():
        dataset = h5_file[key]
        print(f"\nDataset: {key}")
        print(f"Shape: {dataset.shape}")
        print(f"Attributes: {dataset.attrs.items()}")
    for key, val in dataset.attrs.items():
        print(key,val)

print("----------------------------------------")

with h5py.File(h5_path2, "r") as h5_file:
    # Print all datasets and groups in the file
    print("HDF5 Structure:")
    for key in h5_file.keys():
        print(f"- {key}: {type(h5_file[key])}")

    # Check for attributes/ metadata
    for key in h5_file.keys():
        dataset = h5_file[key]
        print(f"\nDataset: {key}")
        print(f"Shape: {dataset.shape}")
        print(f"Attributes: {dataset.attrs.items()}")
    for key, val in dataset.attrs.items():
        print(key,val)

'''
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Path to the HDF5 file
h5_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_left_close_fast_sitting_recording_013_2021-09-14_14-18-22.h5"
h5_path = "/space/chair-nas/tosy/testh5/rock_left_far_fast_standing_recording_005_2021-09-14_11-26-11.h5"

# Open the HDF5 file and read the dataset
with h5py.File(h5_path, "r") as f:
    data = f["data"]  # Assuming 'data' is the dataset key
    num_frames = data.shape[0]  # Total number of time steps

    print(f"Dataset Shape: {data.shape}")  # (time, channel, height, width)

    # Loop through all frames and plot them
    for i in range(num_frames):
        frame = data[i, 0]  # Extracting the first channel

        plt.imshow(frame, cmap="hot")
        plt.title(f"Frame {i} - Channel 0")
        plt.axis("off")
        plt.show()
