from torch.utils.data import DataLoader
from MAIN.utils import validate_dataset_path, validate_classes, print_summary_table
from TrainValTest import *
import torch
import torchhd
import os
from tqdm import tqdm
from torchhd.models import Centroid
import random
import torchmetrics
import torchhd.utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import time

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=8)


TRAINING_METHOD = "adaptive"  # "centroid" "adaptive" "iterative"
LEARNING_RATE = 0.5
ENCODING_METHOD =  # x
TIME_INTERPOLATION_METHOD =
#device = torch.device("cuda" if torch.cuda.is_available() else "soundcpu")
device = "cpu"
print("Using", device)


train_dataset =
val_dataset =
test_dataset =
num_classes = 4
#print_summary_table(train_dataset, val_dataset, test_dataset, batch_size, num_epochs, dimensions)

train_dataloader = DataLoader()
train_dataloader = DataLoader()
train_dataloader = DataLoader()

encoder = HIST_encoder(height=120, width=160, max_time=max_time, dims=dimensions, device=device)

call heatamps from pickle etc.

class_centroids = train_one_epoch(train_dataloader, encoder, dimensions, num_classes, device)
evaluation(test_dataloader, encoder, class_centroids, device)


if __name__ == "__main__":
    main()
