import os
import numpy as np
import h5py
import cv2

TENSOR_FILE = "/space/chair-nas/tosy/DATASETS/Gen3_Chifoumi_H5_HistoQuantized/train/paper_01P_0_0_2250000_4125000.h5"
LABEL_FILE = TENSOR_FILE.split(".h5")[0] + "_bbox.npy"
assert os.path.isfile(TENSOR_FILE), "Check your .h5 tensor input path!"
assert os.path.isfile(LABEL_FILE), "Check your label input path!"
LABEL_DICT = {"0": "paper", "1": "rock", "2": "scissor"}
VIZ_WINDOW = "Dataset Visualization"
def get_delta_t(npy_data):
    deltaTs = npy_data["ts"][1:] - npy_data["ts"][:-1]
    delta_t = np.unique(deltaTs).item()
    return delta_t

labels = np.load(LABEL_FILE)
deltaT = get_delta_t(labels)
with h5py.File(TENSOR_FILE, "r") as f:
    height, width = f["data"].shape[1:3]
    data = f["data"][:]
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.namedWindow(VIZ_WINDOW)
for idx, tensor_frame in enumerate(data):  # Loop through tensor frames
    img = cv2.normalize(tensor_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    t = labels["ts"][idx]
    if t in labels["ts"]:
        label = LABEL_DICT[str(labels[labels["ts"] == t]["class_id"].item())]
        cv2.putText(img, label, (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

    cv2.imshow(VIZ_WINDOW, img)
    cv2.waitKey(50)

cv2.destroyWindow(VIZ_WINDOW)
