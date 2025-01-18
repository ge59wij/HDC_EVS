import torch
from Datasethandling_arrays1 import EventDatasetLoader

class TorchHDCEncoding:
    def __init__(self, root_dir, split):
        self.dataset_loader = EventDatasetLoader(root_dir, split)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_tensors(self):
        """
        Converts event data and labels from NumPy arrays to a tuple of (event_tensor, label_tensor), both as PyTorch tensors
        """
        for i in range(len(self.dataset_loader)):
            # load sample as NumPy arrays
            np_data, np_label = self.dataset_loader[i]
            # Converting to PyTorch tensors
            event_tensor = torch.tensor(np_data, dtype=torch.float32, device=self.device)
            label_tensor = torch.tensor(int(np_label), dtype=torch.long, device=self.device)

            yield event_tensor, label_tensor


# deb
if __name__ == "__main__":
    dataset_path = "/space/chair-nas/tosy/Gen3_Chifoumi_H5_HistoQuantized/"
    split = "train"
    encoder = TorchHDCEncoding(dataset_path, split)

    '''for idx, (event_tensor, label_tensor) in enumerate(encoder.get_tensors()):
        print(f"Sample {idx}: Event Tensor Shape {event_tensor.shape}, Label Tensor {label_tensor}")
        
    '''