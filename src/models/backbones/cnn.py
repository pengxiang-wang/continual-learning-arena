from torch import nn

import torch

import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.te = nn.ModuleDict()  # task embeddings over features
        self.mask_gate = nn.Sigmoid()
        self.test_mask = None
        
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.te["conv1"] = nn.Embedding(1, 32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.te["conv2"] = nn.Embedding(1, 64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.te["conv3"] = nn.Embedding(1, 64)

        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.te["conv4"] = nn.Embedding(1, )


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        
        return x
