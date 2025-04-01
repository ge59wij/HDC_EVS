import torch
import os
import pickle
import torchhd
from MAIN import util
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.grasphdencoding import Raw_events_HDEncoder
from MAIN.DAT_approach.grasphd_appraoch.with_pickle.seperaterawencoderfromhist import Raw_events_HDEncoder_Enhanced
import random
import numpy as np
import tonic
import os
import torch
import gc
from MAIN.util import create_experiment_root, create_unique_run_folder

import gc
gc.collect()
#os.environ["OMP_NUM_THREADS"] = "8"
#torch.set_num_threads(8)
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)
torch.manual_seed(40)
np.random.seed(40)
random.seed(40)
#resource.setrlimit(resource.RLIMIT_AS, (10_000_000_000, 10_000_000_000))  # 2GB limit

#["event_hd_timepermutation", "stem_hd" , "event_hd_timeinterpolation"]
#["thermometer", "linear", "kxk_ngram"]



def main(
    TIME_INTERPOLATION_METHOD="linear",
    dims=5000,
    k=6,
    ngram=5,
    timewindow=50000,
    window_size_ms=600_000,
    overlap_ms=0,
    save=True,
    run_folder=None,
    dataset_name = "mnist",
):
    """Runs the encoding and training pipeline with configurable parameters."""

    # Choose encoding method based on time interpolation technique
    ENCODING_METHOD = Raw_events_HDEncoder
    if TIME_INTERPOLATION_METHOD in ["thermometer", "linear", "kxk_ngram"]:
        ENCODING_METHOD = Raw_events_HDEncoder_Enhanced



    print(f"[INFO] Sliding Window = {window_size_ms}")


    # Initialize Encoder
    encoder = ENCODING_METHOD(
        height=height, width=width, dims=dims, time_subwindow=timewindow, k=k, device=device,
        max_time=window_size_ms, time_method=TIME_INTERPOLATION_METHOD,
        WINDOW_SIZE_MS=window_size_ms, OVERLAP_MS=overlap_ms, ngram=ngram
    )

    # Encode Training & Testing Data
    encoded_train, labels_train = util.encode_dataset(dataset_train, encoder, split_name=train_split)
    encoded_test, labels_test = util.encode_dataset(dataset_test, encoder, split_name=test_split)

    util.plot_heatmap(encoded_train, labels_train, k, window_size_ms, dims, 10, TIME_INTERPOLATION_METHOD,save, run_folder, "train")
    util.plot_heatmap(encoded_test, labels_test, k, window_size_ms, dims, 10, TIME_INTERPOLATION_METHOD,save, run_folder, "test")
    del encoder
    gc.collect()

    # ==================== Training & Testing ==================== #

    accuracies = {}
    train_accuracies = {}

    for method in ["Vanilla", "OnlineHD"]:
        print(f"\n[TRAINING] Training model using {method}...")
        # Train Model
        model, train_acc = util.train_model(encoded_train, labels_train, method, debug=True, d=dims)
        train_accuracies[method] = train_acc

        # Test Model on Training Data
        train_acc, train_preds = util._test_model(model, encoded_train, labels_train)
        print(f"[INFO] {method} Train Accuracy: {train_acc:.3f}%")

        # Test Model on Test Data
        test_acc, test_preds = util._test_model(model, encoded_test, labels_test)
        accuracies[method] = test_acc
        print(f"[INFO] {method} Test Accuracy: {test_acc:.3f}%")

        # Save Confusion Matrices
        util.plot_confusion_matrix(labels_train, train_preds, save, run_folder, f"Train_{method}")
        util.plot_confusion_matrix(labels_test, test_preds, save, run_folder, f"Test_{method}")

        # Free Memory After Each Model
        del model
        gc.collect()

    # ==================== Save Results ==================== #

    if save:
        util.save_hyperparameters(run_folder, {
            "Methods": list(accuracies.keys()),
            "Train_Accuracies": train_accuracies,
            "Test_Accuracies": accuracies,
            "k": k,
            "Timewindow": timewindow,
            "DIMS": dims,
            "Train_samples": len(dataset_train),
            "Test_samples": len(dataset_test),
            "Method": TIME_INTERPOLATION_METHOD,
        })






def load__dataset(dataset_path, split, max_samples, dataset_name):
    """
    Loads dataset based on dataset_name.
    - Chifoumi (pickle files) -> loads & shifts timestamps.
    - NMNIST (Tonic) -> loads exactly `max_samples` per digit (0-9) using Tonic's API.
    """
    if dataset_name == "chifoumi":
        dataset = load_pickle_dataset(dataset_path, split, max_samples)
        return dataset

    elif dataset_name == "nmnist":
        dataset = tonic.datasets.NMNIST(save_to=dataset_path, train=(split == "train"))
        dataset_list = []
        dtype = np.dtype([("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", np.int8)])

        # Store counts per digit class
        class_counts = {i: 0 for i in range(10)}

        for i in range(len(dataset)):
            events, class_id = dataset[i]  # Get (events, label) directly from Tonic

            # Ensure we don't exceed max_samples per class
            if class_counts[class_id] < max_samples:
                structured_events = np.array([(t, x, y, p) for x, y, t, p in events], dtype=dtype)
                dataset_list.append((structured_events, class_id))
                class_counts[class_id] += 1

            # Stop early if we have enough samples for all digits
            if all(count >= max_samples for count in class_counts.values()):
                break

        print(f"Loaded {len(dataset_list)} NMNIST samples for {split} split.")
        return dataset_list

    else:
        raise ValueError("Unknown dataset! Choose 'chifoumi' or 'nmnist'.")


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

if __name__ == "__main__":

    device = "cpu"
    print(f"Using device: {device}")

    # Default dataset settings
    dataset_path = "/space/chair-nas/tosy/data/"
    train_split = "Train"
    test_split = "Test"
    height, width = 34, 34
    dataset_name = "chifoumi"
    max_samples_train, max_samples_test = 50, 40
    # Adjust settings for Chifoumi dataset
    if dataset_name == "chifoumi":
        dataset_path = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/Smaller samples/"
        height, width = 480, 640
        test_split = "Test"

    # Load datasets
    dataset_train = load__dataset(dataset_path, train_split, max_samples_train, dataset_name)
    dataset_test = load__dataset(dataset_path, test_split, max_samples_test, dataset_name)


    experiment_root = create_experiment_root("/space/chair-nas/tosy/experiments/Chifoumi2")

    #  Set baseline parameters
    baseline_params = {
        "dims": 4000,
        "k": 6,
        "ngram": 4,
        "timewindow": 50_000,
        "window_size_ms": 600_000,
        "overlap_ms": 0,
        "save": True
    }
    # ["event_hd_timepermutation", "stem_hd" , ""]
    # ["thermometer", "linear", "kxk_ngram"]
    # Encoding methods to test
    encoding_methods = [ "event_hd_timepermutation", "event_hd_timeinterpolation"]

    # Parameters to test
    param_grid = {
        "dims": [5000],
        "window_size_ms": [300_000, 600_000],
        "overlap_ms": [0, 20_000],
        "k": [5, 9, 20],
        "timewindow": [50_000, 90_000],
        "ngram": [4]
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
                    # Run experiment
                    main(**test_params)

                    # Clear cache between runs
                    torch.cuda.empty_cache()
                    gc.collect()

    print("\nAll experiments completed!")


'''
stem: Interpolation is done dimension-wise (not concatenation!) for spatial. for temp: STEMHD does use concatenation for  1D temporal interpolation.
"A proportion (1 - α) of Tj is taken from the first vector and α from the next one. The two parts are concatenated to form the new time hypervector."


stem_hd:    spatial:  dimensionwise interpolation. weighted combination of neighboring hypervectors based on distance. same as eventhd, grasphd.
            temporal: STEMHD does use concatenation for its 1D temporal interpolation.   ##########One per bin
                        A proportion (1-alpha) of Tk is taken from first vector, and alpha from tj+1 from next one.
                        concat together. 

event_hd:   spatial:    uses weighted sum per element to interpolate positions.
            temporal:   event_hd_timepermutation: Uses permutation-based encoding, time step is encoded using permutations of a base hypervector  for each t.
                        eventhd_timeinterpolation:Uses weighted sum per element for interpolation (NOT concatenation).  Multiple—one per intermediate timestamp

grasp_hd:   spatial:    weighted sum per dimension (similar to EventHD for time)
            temporal:   Uses weighted sum per element to blend temporal hypervectors (same as EventHD time interpolation). Multiple—one per intermediate timestamp
            same as eventhd timeineterpolation.

STEMHD = Uses concatenation for time, not for space. in time: One per bin
EventHD = Uses weighted sum for both space and time (no concatenation).
GraspHD = Uses weighted sum for both space and time (no concatenation).

'''
