import numpy as np

data = np.load("/space/chair-nas/tosy/H5_Custom_HistoChifoumi/val/paper_200206_180138_0_0_bbox.npy")
npy_path = "//space/chair-nas/tosy/H5_Custom_HistoChifoumi/val/paper_200206_180138_0_0_bbox.npy"
#data3= np.load("//space/chair-nas/tosy/Gen3_Chifoumi_DAT/train/scissor_200212_093346_0_0_bbox.npy")

print(npy_path)
bbox_labels = np.load(npy_path)
print("Bounding Box Labels Shape:", bbox_labels.shape)
print("Sample Bounding Boxes:",
      bbox_labels)

print("___________")
print(data)

import torchhd
#print("torchhd version:" , torchhd.__version__)

