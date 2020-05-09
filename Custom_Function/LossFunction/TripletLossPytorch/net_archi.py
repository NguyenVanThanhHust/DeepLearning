import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import *
import torchvision
import matplotlib.pyplot as plt
from dataset import SiameseNetworkDataset
from config import Config

class SimpleLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(SimpleLayer, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.forward_cnn = nn.Sequential(
        nn.Conv2d(self.in_channels, self.out_channels, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True), 
        nn.BatchNorm2d(16)
    )
  def forward(self, x):
    output = self.forward_cnn(x)
    return output

class SimpleForwardNet(nn.Module):
  def __init__(self):
    super(SimpleForwardNet, self).__init__()
    self.cnn_layer_1 = SimpleLayer(in_channels = 1, out_channels = 16)
    self.cnn_layer_2 = SimpleLayer(in_channels = 16, out_channels = 32)
    self.cnn_layer_3 = SimpleLayer(in_channels = 32, out_channels = 64)
    self.fully_conn = nn.Sequential(
        nn.Linear(64*7*7, 1000), 
        nn.Linear(1000, 10)
    )
  def forward(self, x):
    x = self.cnn_layer_1(x)
    x = self.cnn_layer_2(x)
    x = self.cnn_layer_3(x)
    output = self.fully_conn(x)
    return output

class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.cnn1 = SimpleForwardNet()
    self.cnn2 = SimpleForwardNet()
  def forward(self, input_1, input_2):
    print("input 1 shape: ", input_1.shape)
    print("input 2 shape: ", input_2.shape)
    output_1  = self.cnn1(input_1)
    output_2  = self.cnn2(input_2)
    return output_1, output_2