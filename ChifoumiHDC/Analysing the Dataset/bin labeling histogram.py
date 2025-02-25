import os
import h5py
import numpy as np

SRC_FOLDER = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/val/"
DEST_FOLDER = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/processed/Bin Labeled/VAL BIN LABELED/"
os.makedirs(DEST_FOLDER, exist_ok=True)

# List of selected samples to process
selected_samples = {
    "scissors_right_far_slow_standing_recording_019_2021-09-14_15-04-40.h5": [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    "scissors_left_far_slow_sitting_recording_013_2021-09-14_14-17-01.h5": [],
    "scissors_left_close_slow_standing_recording_022_2021-09-14_15-21-36.h5":[],
    "scissors_left_far_slow_sitting_recording_001_2021-09-14_10-47-08.h5" :[],
    "scissor_200212_133959_0_0.h5" :[],
    "scissors_right_far_slow_standing_recording_001_2021-09-14_10-52-12.h5":[],
    "scissor_200212_094850_0_0.h5": [50,51,52,53,54,55,56,67],
    "scissors_left_far_slow_standing_recording_017_2021-09-14_14-52-41.h5" :[51],
    "scissors_recording_only_scissor2__2021-09-14_13-23-41.h5":[],
    "scissor_200212_092940_0_0.h5":[61, 62,63,64,65,66,67,68,69,70,71],
    "scissors_left_far_slow_standing_recording_009_2021-09-14_13-44-17.h5" :[48 ,49],
    "scissor_200212_133908_0_0.h5" :[],
    "scissor_200212_092419_0_0.h5" :[36,37,38,39,40,41,42, 70,71,72,73,74,75,76],
    "scissor_200212_143032_0_0.h5":[],
    "scissors_right_close_slow_sitting_recording_002_2021-09-14_11-03-06.h5":[],
    "scissor_200212_094102_0_0.h5":[38,39,49,40,41,42,43,44,45,46,47,48,50, 65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87
                                    ,88,89,90,91,92,93,94,95,96,97,98,99,100,101,116,119],
    "scissors_right_close_slow_standing_recording_003_2021-09-14_11-14-09.h5":[],
    "scissors_left_far_slow_standing_recording_015_2021-09-14_14-37-16.h5":[70,71],
    "scissors_right_far_slow_standing_recording_002_2021-09-14_11-04-17.h5":[],
    "scissors_right_far_slow_sitting_recording_021_2021-09-14_15-17-13.h5":[58,59,60,61,62,63,64,65],
    "scissors_left_far_slow_standing_recording_021_2021-09-14_15-15-29.h5":[],

    "rock_right_close_slow_standing_recording_001_2021-09-14_10-51-09.h5":[],
    "rock_left_far_fast_sitting_recording_019_2021-09-14_15-03-19.h5":[],
    "rock_200212_135257_0_0.h5":[],
    "rock_right_close_slow_standing_recording_016_2021-09-14_14-48-29.h5":[],
    "rock_left_far_slow_sitting_recording_018_2021-09-14_14-57-10.h5":[],
    "rock_recording_only_rock2__2021-09-14_13-20-16.h5":[],
    "rock_right_close_slow_sitting_recording_008_2021-09-14_13-30-29.h5": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                                                                                                                   23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                                                                                                                   36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                                                                                                                   49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
    "rock_200212_092220_0_0.h5":[], "rock_left_far_fast_standing_recording_000_2021-09-14_10-41-32.h5":[45,46],
    "rock_200212_090607_0_0.h5":[36,37,38,39],
    "rock_left_close_slow_sitting_recording_011_2021-09-14_14-00-52.h5": [23,24,25,26,27],
    "rock_200212_154845_0_0.h5":[], "rock_02R_0_0_2350000_4650000.h5":[],"rock_left_close_slow_sitting_recording_010_2021-09-14_13-52-52.h5":[],
    "rock_left_close_slow_standing_recording_017_2021-09-14_14-52-25.h5":[43,44],
    "rock_recording_only_rock2__2021-09-14_13-20-31.h5":[], "rock_left_far_fast_standing_recording_021_2021-09-14_15-16-26.h5":[],
    "rock_left_far_slow_sitting_recording_008_2021-09-14_13-27-50.h5":[44,45,46,47],
    "rock_200212_153525_0_0.h5":[], "rock_200212_154741_0_0.h5":[],"rock_left_far_slow_standing_recording_007_2021-09-14_11-44-42.h5":[],
    "rock_200212_133949_0_0.h5":[], "rock_200212_143118_0_0.h5":[], "rock_right_close_slow_sitting_recording_005_2021-09-14_11-26-30.h5":[],
    "rock_right_close_slow_sitting_recording_003_2021-09-14_11-13-33.h5":[], "rock_200212_092758_0_0.h5":[64,65,66,67,68,69,70,71],
    "rock_200212_092920_0_0.h5":[56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71], "rock_200212_133954_0_0.h5":[],
    "rock_200212_133837_0_0.h5":[],



    "paper_left_close_slow_standing_recording_001_2021-09-14_10-48-16.h5":[],"paper_left_far_fast_standing_recording_013_2021-09-14_14-20-11.h5":[],
    "paper_right_far_fast_standing_recording_004_2021-09-14_11-22-40.h5":[], "paper_left_close_fast_sitting_recording_009_2021-09-14_13-44-34.h5":[],
    "paper_left_far_fast_standing_recording_026_2021-09-14_15-49-57.h5":[], "paper_right_far_fast_standing_recording_009_2021-09-14_13-48-54.h5":[30,31,32,33],
    "paper_recording_only_paper_2021-09-14_13-10-51.h5":[30], "paper_recording_only_paper_2021-09-14_13-11-55.h5":[],
    "paper_right_far_fast_sitting_recording_009_2021-09-14_13-47-26.h5":[], "paper_recording_only_paper2__2021-09-14_13-22-27.h5":[],
    "paper_left_close_fast_sitting_recording_003_2021-09-14_11-11-47.h5":[],
    "paper_right_far_fast_sitting_recording_027_2021-09-14_16-01-16.h5":[], "paper_left_far_slow_sitting_recording_023_2021-09-14_15-28-09.h5":[],
    "paper_200212_154717_0_0.h5":[], "paper_recording_only_paper2__2021-09-14_13-22-20.h5":[39],
    "paper_left_far_fast_sitting_recording_022_2021-09-14_15-22-32.h5":[], "paper_left_far_slow_sitting_recording_016_2021-09-14_14-45-45.h5":[43],
    "paper_left_close_slow_standing_recording_014_2021-09-14_14-27-29.h5":[54,55,56,57,58,63],
    "paper_right_far_slow_sitting_recording_023_2021-09-14_15-31-17.h5":[],
    "paper_right_far_fast_standing_recording_003_2021-09-14_11-15-28.h5":[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 87, 88],
    "paper_left_close_slow_standing_recording_009_2021-09-14_13-43-58.h5":[40,41,42,43,44,45,46,47],
    "paper_left_far_slow_sitting_recording_022_2021-09-14_15-20-28.h5":[33,34],
    "paper_left_far_slow_sitting_recording_017_2021-09-14_14-52-12.h5":[46],
    "paper_left_far_slow_standing_recording_009_2021-09-14_13-44-15.h5":[31, 32, 33, 34, 52, 53, 54, 55, 56, 57]
}

BACKGROUND_LABEL = 404

for filename, empty_bins in selected_samples.items():
    src_path = os.path.join(SRC_FOLDER, filename)
    dest_path = os.path.join(DEST_FOLDER, filename)

    if not os.path.exists(src_path):
        print(f"Warning: {filename} not found in source folder. Skipping.")
        continue

    with h5py.File(src_path, "r") as f:
        event_data = f["data"][:]  # Shape: (T, 2, H, W)
        class_id = f["class_id"][()].item()  # Ensure it's an integer

    T = event_data.shape[0]  # Number of time bins

    # Initialize labels array with the same class_id for all bins
    labels = np.full((T,), class_id, dtype=np.int32)

    # Apply manual background labeling
    for bin_idx in empty_bins:
        if 0 <= bin_idx < T:
            labels[bin_idx] = BACKGROUND_LABEL

    with h5py.File(dest_path, "w") as f:
        f.create_dataset("data", data=event_data)  # Keep original data
        f.create_dataset("class_id", data=np.array([class_id], dtype=np.int32))  # Store single integer
        f.create_dataset("labels", data=labels, dtype=np.int32)  # Store per-bin labels

    print(f"Processed {filename} | Background bins: {empty_bins}")

print("All selected HDF5 files labeled and saved in /VAL BIN LABELED/")
