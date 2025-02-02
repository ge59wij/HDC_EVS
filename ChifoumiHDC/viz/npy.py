import numpy as np
sample_npy_path = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT/train/paper_01P_0_0_0_2250000_bbox.npy"

try:
    labels = np.load(sample_npy_path, allow_pickle=True)
    print(f"Loaded {len(labels)} bounding boxes from {sample_npy_path}\n")

    for i in range(len(labels)):
        print(labels[i])  # Print the first bounding box entry for inspection

except Exception as e:
    print(f"Error loading {sample_npy_path}: {e}")

#bbox structure:  (ts, x_min, y_min, x_max, y_max, class_id, ?, ?)
