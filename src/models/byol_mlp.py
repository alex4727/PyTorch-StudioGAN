import torch
import torch.nn as nn

class MLP(nn.Module):
    #Default hidden size is for image net.. this needs to be manually modified for datasets like CIFAR10
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            #nn.BatchNorm1d(hidden_size),   #No batch norm for MLP due to spectral normalization.
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
    def forward(self, x):
        return self.net(x)

