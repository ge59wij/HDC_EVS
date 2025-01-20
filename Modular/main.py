from Modular.DatasetLoader import EventDatasetLoader
from utils import print_summary_table
from Modular.HDCencoding.Encoding1 import Encoding1

def main():
    # Paths and configuration
    dataset_path = "/space/chair-nas/tosy/Simple_chifoumi/"
    encoding_method = Encoding1  # Choose Encoding1 or Encoding2

    # Load datasets
    train_dataset = EventDatasetLoader(dataset_path, "train")
    val_dataset = EventDatasetLoader(dataset_path, "val")
    test_dataset = EventDatasetLoader(dataset_path, "test")

    # Initialize encoding
    encoder = encoding_method(
        dimensions=8000, max_time=150, height=120, width=160
    )

    # Print summary
    print_summary_table(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=1,
        num_epochs=10,
        dimensions=8000,
        height=120,
        width=160,
    )

    # Train and evaluate
    encoder.train(train_dataset, val_dataset)
    encoder.evaluate(test_dataset)

if __name__ == "__main__":
    main()
