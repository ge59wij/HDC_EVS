import h5py
file_path = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi/train/paper_03P_0_0_1875000_3400000.h5"
file_path2 = '/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_right_far_slow_standing_recording_020_2021-09-14_15-10-36.h5'

with h5py.File(file_path, "r") as f:
    if "data" in f:
        data_group = f["data"]
        print("Attributes of 'data':")
        for key, value in data_group.attrs.items():
            print(f"{key}: {value}")
    else:
        print("'data' group not found")

with h5py.File(file_path2, "r") as f:
    if "data" in f:
        data_group = f["data"]
        print("\nAttributes of 'data':")
        for key, value in data_group.attrs.items():
            print(f"{key}: {value}")
    else:
        print("'data' group not found")

