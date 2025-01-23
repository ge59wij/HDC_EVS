from Modular.DatasetLoader import EventDatasetLoader
from utils import print_summary_table
from Modular.HDCencoding.Encoding1 import Encoding1

def main():
    dataset_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized"

    encoding_method = Encoding1

    # Load datasets
    train_dataset = EventDatasetLoader(dataset_path, "train")
    val_dataset = EventDatasetLoader(dataset_path, "val")
    test_dataset = EventDatasetLoader(dataset_path, "test")
    DIMENSIONS = 8000
    MAX_TIME = 150
    HEIGHT = 120
    WIDTH = 160
    BATCHES = 32
    NUM_EPOCHS = 3



    # Initialize encoding
    #encoder = encoding_method(dimensions= DIMENSIONS, max_time= MAX_TIME, height=HEIGHT, width=WIDTH)
    encoder = encoding_method(dimensions= DIMENSIONS, max_time= MAX_TIME, height=HEIGHT, width=WIDTH,  )

    print_summary_table( train_dataset, val_dataset, test_dataset, batch_size= BATCHES, num_epochs= NUM_EPOCHS , dimensions=DIMENSIONS, height=HEIGHT, width=WIDTH, )

    # Train and evaluate
    encoder.train(train_dataset, val_dataset)
    encoder.evaluate(test_dataset)

if __name__ == "__main__":
    main()
