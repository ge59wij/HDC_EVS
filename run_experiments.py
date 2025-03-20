import os
import torch
import gc
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.main_enc import main
from MAIN.util import create_experiment_root, create_unique_run_folder

experiment_root = create_experiment_root("/space/chair-nas/tosy/experiments/")

#  Set baseline parameters
baseline_params = {
    "max_samples_train": 100,
    "max_samples_test": 50,
    "dims": 4000,
    "k": 6,
    "ngram": 4,
    "timewindow": 50_000,
    "window_size_ms": 600_000,
    "overlap_ms": 0,
    "save": True
}
#["event_hd_timepermutation", "stem_hd" , ""]
#["thermometer", "linear", "kxk_ngram"]
# Encoding methods to test
encoding_methods = ["event_hd_timepermutation", "kxk_ngram", "linear","event_hd_timeinterpolation", "kxk_ngram"]

# Parameters to test
param_grid = {
    "dims": [6000],
    "window_size_ms": [300_000, 600_000],
    "overlap_ms": [0, 20_000],
    "k": [5, 9, 20],
    "timewindow": [90_000, 50_000 , 130_000],
    "ngram": [4, 5, 9]
}

datasets = ["chifoumi"]

# Run experiments
for dataset_name in datasets:
    for encoding_method in encoding_methods:
        for param_name, values in param_grid.items():
            for param_value in values:
                # Update params for this test
                test_params = baseline_params.copy()
                test_params[param_name] = param_value
                test_params["TIME_INTERPOLATION_METHOD"] = encoding_method
                test_params["dataset_name"] = dataset_name  # Ensure dataset is switched

                # Create run folder
                run_folder = create_unique_run_folder(
                    experiment_root, encoding_method, param_name, param_value, dataset_name
                )
                test_params["run_folder"] = run_folder

                print(f"\n Running: {encoding_method} | {param_name} = {param_value} | Dataset: {dataset_name}")
                main_enc.load_pickle_dataset()
                # Run experiment
                main(**test_params)

                # Clear cache between runs
                torch.cuda.empty_cache()
                gc.collect()

print("\nAll experiments completed!")

def load_pickle_dataset(dataset_path, split, max_samples):
    """
    Returns List[Tuple]: A list of tuples (events, class_id), where events are the event tuples (t, (x, y), p).
    """
    split_path = os.path.join(dataset_path, split)
    files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith('.pkl')]
    random.shuffle(files)
    dataset = []
    for file in files[:max_samples]:
        with open(file, 'rb') as f:
            events, class_id = pickle.load(f)
            events = np.array(
                [(int(t), int(x), int(y), int(p)) for t, x, y, p in events],
                dtype=np.int32  # Ensures all values are stored as int32
            )
        dataset.append((events, class_id))
    print(f"Loaded {len(dataset)} samples from {split} split.")
    return dataset
