import os
import glob
import subprocess

DATASET_ROOT = "/space/chair-nas/tosy/Gen3_Chifoumi_DAT"
OUTPUT_ROOT = "/space/chair-nas/tosy/H5_Custom_HistoChifoumi"
GENERATE_SCRIPT = "/home/tosy/Desktop/metavision/sdk/ml/python_samples/generate_hdf5/generate_hdf5.py"

DELTA_T = 10000
PREPROCESS = "histo_quantized"
NEG_BIT_LEN = 4
TOTAL_BIT_LEN = 8
HEIGHT = "120"
WIDTH = "160"
MAX_VAL = 10

param_log_path = os.path.join(OUTPUT_ROOT, "generation_params.txt")
param_text = f"""Command-line parameters used for dataset generation:
-------------------------------------------------
--delta-t {DELTA_T}
--preprocess {PREPROCESS}
--neg_bit_len_quantized {NEG_BIT_LEN}
--total_bit_len_quantized {TOTAL_BIT_LEN}
--height_width {HEIGHT} {WIDTH}
--max_val {MAX_VAL}
-------------------------------------------------
"""

os.makedirs(OUTPUT_ROOT, exist_ok=True)

with open(param_log_path, "w") as f:
    f.write(param_text)

splits = ["train", "val", "test"]
for split in splits:
    split_input_path = os.path.join(DATASET_ROOT, split)
    split_output_path = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(split_output_path, exist_ok=True)

    dat_files = sorted(glob.glob(os.path.join(split_input_path, "*.dat")))

    for dat_file in dat_files:
        if dat_file.endswith("_cd.dat"):
            bbox_file = dat_file.replace("_cd.dat", "_bbox.npy")
        elif dat_file.endswith("_td.dat"):
            bbox_file = dat_file.replace("_td.dat", "_bbox.npy")
        else:
            print(f"Skipping {dat_file} (Unknown format)")
            continue
        if not os.path.exists(bbox_file):
            print(f"Skipping {dat_file} (No matching bbox file found)")
            continue


        command = [
            "python", GENERATE_SCRIPT,
            dat_file,
            "--box-labels", bbox_file,
            "-o", split_output_path,
            "--delta-t", str(DELTA_T),
            "--preprocess", PREPROCESS,
            "--neg_bit_len_quantized", str(NEG_BIT_LEN),
            "--total_bit_len_quantized", str(TOTAL_BIT_LEN),
            "--height_width", HEIGHT, WIDTH,
            "--max_val", str(MAX_VAL),
        ]

        print(f" Processing {dat_file}...")
        subprocess.run(command)
        print(f" Finished processing {dat_file}!")

print("\n All splits processed! Check the output in:", OUTPUT_ROOT)
