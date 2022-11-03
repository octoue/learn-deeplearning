import torch
import torch.nn as nn
import torch.optim as optim

import math
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor
import time

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loaders
batch_size = 64
train_dataloader = DataLoader(training_data,batch_size = batch_size)
test_dataloader = DataLoader(test_data,batch_size = batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") # [64, 1, 28, 28]
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class VisionTransformer(nn.Module):
    pass