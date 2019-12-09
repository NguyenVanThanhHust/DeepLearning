import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class BasicConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.forward_cnn = nn.Sequential(
            nn.Conv2D(self.in_channels, self.out_channels, kernel_size = (3,3), stride = 1, padding = 1),
            nn.MaxPool2d(),
            nn.BatchNorm2d(),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        x = self.forward_cnn(x)
        return x

class Layer2Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.forward_cnn = nn.Sequential(
            BasicConvLayer(self.in_channels, self.out_channels),
            BasicConvLayer(self.out_channels, self.out_channels)
        )
    def forward(self, x):
        x = self.forward_cnn(x)
        return x

class Layer3Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.forward_cnn = nn.Sequential(
            BasicConvLayer(self.in_channels, self.out_channels),
            BasicConvLayer(self.in_channels, self.out_channels),
            BasicConvLayer(self.out_channels, self.out_channels)
        )
    def forward(self, x):
        x = self.forward_cnn(x)
        return x

class FphbNet(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 512):
        super().__init__()