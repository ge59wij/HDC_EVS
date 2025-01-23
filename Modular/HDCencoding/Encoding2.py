import torch
import torch.nn as nn
from torchhd import embeddings, bind, multiset, ngrams, hard_quantize
from torchhd.models import Centroid
import torchmetrics
from tqdm import tqdm

class Encoding2:
    def __init__(self, dimensions, max_time, height, width, batch_size, num_epochs, num_classes):
        self.dimensions = dimensions
        self.max_time = max_time
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.encoder = self.Encoder(dimensions, height, width)
        self.model = None

