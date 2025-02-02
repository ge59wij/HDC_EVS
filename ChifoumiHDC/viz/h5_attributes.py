import h5py
file_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/train/paper_03P_0_0_1875000_3400000.h5"

#file_path2 = '/space/chair-nas/tosy/customchifoumi/train/paper_right_far_slow_standing_recording_020_2021-09-14_15-10-36.h5'
file_path2 = '/space/chair-nas/tosy/customdownsampled/train/paper_right_far_slow_standing_recording_020_2021-09-14_15-10-36.h5'

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
        print("Attributes of 'data':")
        for key, value in data_group.attrs.items():
            print(f"{key}: {value}")
    else:
        print("'data' group not found")

#python generate_hdf5.py /space/chair-nas/tosy/Gen3_Chifoumi_DAT/test/*dat -o /space/chair-nas/tosy/customdownsampled/test --delta-t 1000 --preprocess histo_quantized --normalization_quantized --num-workers 12 --box-labels /space/chair-nas/tosy/Gen3_Chifoumi_DAT/test/*npy --height_width 120 160
#python generate_hdf5.py /space/chair-nas/tosy/Gen3_Chifoumi_DAT/val/*dat -o /space/chair-nas/tosy/customdownsampled/val --delta-t 1000 --preprocess histo_quantized --normalization_quantized --num-workers 12 --box-labels /space/chair-nas/tosy/Gen3_Chifoumi_DAT/val/*npy --height_width 120 160
#python generate_hdf5.py /space/chair-nas/tosy/Gen3_Chifoumi_DAT/train/*dat -o /space/chair-nas/tosy/customdownsampled/train --delta-t 1000 --preprocess histo_quantized --normalization_quantized --num-workers 12 --box-labels /space/chair-nas/tosy/Gen3_Chifoumi_DAT/train/*npy --height_width 120 160