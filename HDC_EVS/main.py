from torch.utils.data import DataLoader
from HDC_EVS.Chifoumi_h5_loading_pad import EventDatasetLoader, collate_fn
from HDC_EVS.utils import validate_dataset_path, validate_classes, print_summary_table
from enc import GestureHDEventEncoder
from HDC_EVS.heatmap import visualize_sample_hv_distances
from TrainValTest import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", device)
HDC_encoder = GestureHDEventEncoder  # GraspHDEventEncoder


def main():
    #root_dir = "/space/chair-nas/tosy/customdownsampled/"
    root_dir = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized"
    train_split, val_split, test_split = "train", "val", "test"

    batch_size, num_workers, num_epochs, dimensions = 1, 0, 3, 6000
    sample_size, max_time = 20 , 2000 # heatmap

    validate_dataset_path(root_dir)
    train_dataset = EventDatasetLoader(root_dir, train_split)
    val_dataset = EventDatasetLoader(root_dir, val_split)
    test_dataset = EventDatasetLoader(root_dir, test_split)
    num_classes = train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else 3
    validate_classes(num_classes)
    print_summary_table(train_dataset, val_dataset, test_dataset, batch_size, num_epochs, dimensions)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=num_workers)

    #encoder = HDC_encoder(height=120, width=160, max_time=10000, dims=dimensions, chunk_size=3, device=device) #height w 120 160 #get rid of chunks
    encoder = HDC_encoder(height=120, width=160, max_time=max_time, dims=dimensions, device=device)

    visualize_sample_hv_distances(train_dataloader, encoder, sample_size)

    class_centroids = train_one_epoch(train_dataloader, encoder, dimensions, num_classes, device)

    validate_one_epoch(val_dataloader, encoder, class_centroids, device)

    evaluation(test_dataloader, encoder, class_centroids, device)


if __name__ == "__main__":
    main()
