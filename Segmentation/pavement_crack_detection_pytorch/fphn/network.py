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

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
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

class ConcatLayer(nn.Module):
    def __init__(self, input_layer, up_input_layer):
        super().__init__()
        self.input_layer = input_layer
        self.up_input_layer = up_input_layer
        self.upsample_width, self.upsample_height = up_input_layer.shape[2], up_input_layer.shape[3]
        self.size = (2 * self.upsample_width, 2 * self.upsample_height)
        self.reduce_dim_to_512 = nn.Conv2D(self.input_layer[1]+ self.up_input_layer[1], 512, kernel_size = (1,1))
        assert self.size[0]==self.input_layer[2] and self.size[1]==self.input_layer[3] ,'can not upsampling and concantenate, check dimension'
    def forward(self, input_layer, up_input_layer):
        upsample_map = Interpolate(self.size, mode='cubic')(up_input_layer)
        x = torch.cat([self.input_layer, upsample_map], 1)
        return x

class SideNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2D(512, 512, kernel_size=(1,1), stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(512, 512, kernel_size=(3,3), stride=1, padding=1)
    def forward(self, x):
        x = self.conv11(x)
        x = self.deconv(x)
        x = nn.ReLU(inplace = True)(x)
        return x

class BottomUpNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 

class FphbNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 512):
        super().__init__()
        self.bottom_up_layer_1 = Layer2Conv(3, 64)
        self.bottom_up_layer_2 = Layer2Conv(64, 128)
        self.bottom_up_layer_3 = Layer3Conv(128, 256)
        self.bottom_up_layer_4 = Layer3Conv(256, 512)
        self.bottom_up_layer_5 = Layer3Conv(512, 512)

    def forward(self, x):
        feature_map_layer_1 = self.bottom_up_layer_1(x)
        feature_map_layer_2 = self.bottom_up_layer_2(feature_map_layer_1)
        feature_map_layer_3 = self.bottom_up_layer_3(feature_map_layer_2)
        feature_map_layer_4 = self.bottom_up_layer_4(feature_map_layer_3)
        feature_map_layer_5 = self.bottom_up_layer_5(feature_map_layer_4)

        feature_pyramid_4 = ConcatLayer(feature_map_layer_4, feature_map_layer_5)
        feature_pyramid_3 = ConcatLayer(feature_map_layer_3, feature_pyramid_4)
        feature_pyramid_2 = ConcatLayer(feature_map_layer_2, feature_pyramid_3)
        feature_pyramid_1 = ConcatLayer(feature_map_layer_1, feature_pyramid_2)

        side_network_5 = SideNetwork()(feature_map_layer_5)
        side_network_4 = SideNetwork()(feature_pyramid_4)
        side_network_3 = SideNetwork()(feature_pyramid_3)
        side_network_2 = SideNetwork()(feature_pyramid_2)
        side_network_1 = SideNetwork()(feature_pyramid_1)


        return 