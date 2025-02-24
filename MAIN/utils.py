from tabulate import tabulate
import os

def print_summary_table(train_dataset, val_dataset, test_dataset, batch_size, num_epochs, dimensions):
    table = [
        ["Training Samples", len(train_dataset)],
        ["Validation Samples", len(val_dataset)],
        ["Test Samples", len(test_dataset)],
        ["Batch Size", batch_size],
        ["Number of Epochs", num_epochs],
        ["Hypervector Dimensions", dimensions],
    ]
    print("\nDataset and Training Configuration Summary:")
    print(tabulate(table))

def validate_dataset_path(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    label_map_path = os.path.join(dataset_path, "label_map_dictionary.json")
    if not os.path.isfile(label_map_path):
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")

def validate_classes(num_classes):
    if num_classes <= 0:
        raise ValueError("No classes found in the json dictionary")