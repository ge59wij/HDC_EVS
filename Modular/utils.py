from tabulate import tabulate

def print_summary_table(train_dataset, val_dataset, test_dataset, batch_size, num_epochs, dimensions, height, width):
    table = [
        ["Training Samples", len(train_dataset)],
        ["Validation Samples", len(val_dataset)],
        ["Test Samples", len(test_dataset)],
        ["Batch Size", batch_size],
        ["Number of Epochs", num_epochs],
        ["Hypervector Dimensions", dimensions],
        ["Dims", height, "x" , width],
    ]
    print("\nDataset and Training Configuration Summary:")
    print(tabulate(table))
