from torch.onnx import export

from Modular.DatasetLoader import EventDatasetLoader
from utils import validate_dataset_path, validate_classes , print_summary_table

from Modular.HDCencoding.Encoding1 import Encoding1
from Modular.HDCencoding.Encoding2 import Encoding2
from torch.utils.data import DataLoader # Using PyTorch DataLoader for batching and shuffling
import os
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)


DIMENSIONS = 8000
MAX_TIME = 150
HEIGHT = 120
WIDTH = 160
NUM_EPOCHS = 3
BATCH_SIZE = 8 #32 still padding issue
NUM_WORKERS = 8  #number of subprocesses (CPU threads) used for loading data from dataset into batches during training, 32 available



HDC_method = Encoding2  ####here enc


def main():
    dataset_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized"
    validate_dataset_path(dataset_path)

    # Load datasets
    train_dataset = EventDatasetLoader(dataset_path, "train", MAX_TIME)
    val_dataset = EventDatasetLoader(dataset_path, "val", MAX_TIME)
    test_dataset = EventDatasetLoader(dataset_path, "test", MAX_TIME)
    num_classes = train_dataset.num_classes
    validate_classes(num_classes)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print_summary_table(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE,
                        num_epochs=NUM_EPOCHS, dimensions=DIMENSIONS, height=HEIGHT, width=WIDTH)


    # Init Encoding
    encoder = HDC_method(dimensions=DIMENSIONS, max_time=MAX_TIME, height=HEIGHT, width=WIDTH,
                         batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, num_classes=num_classes).to(device)


    # Training + val and evaluate
    encoder.train(train_loader, val_loader)
    encoder.evaluate(test_loader)



if __name__ == "__main__":
    main()
