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
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size = (3,3), stride = 1, padding = 1),
            # nn.BatchNorm2d(num_features = self.out_channels), # replace with Instance norm to use batch norm = 1, ( out of mem cuda)
            nn.InstanceNorm2d(num_features = self.out_channels),
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
            BasicConvLayer(self.out_channels, self.out_channels),
            nn.MaxPool2d(kernel_size = (2,2))
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
            BasicConvLayer(self.out_channels, self.out_channels),
            BasicConvLayer(self.out_channels, self.out_channels),
            nn.MaxPool2d(kernel_size = (2,2))
        )
    def forward(self, x):
        x = self.forward_cnn(x)
        return x

class ConcatLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample_map = nn.Upsample(scale_factor=2, mode='bicubic')
        self.padding = nn.ReflectionPad2d((0, 1, 0, 1))
    def forward(self, input_layer, up_input_layer):
        upsample_map = nn.Upsample((input_layer.size[2], input_layer.size[3]), mode='bicubic')(up_input_layer)
        input_padding = self.padding(upsample_map)
        x = torch.cat([input_layer, input_padding], 1)
        print("x shape: ", x.size())
        return x

class SideNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(512, 512, kernel_size=(1,1), stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(512, 512, kernel_size=(3,3), stride=1, padding=1)
    def forward(self, x):
        x = self.conv11(x)
        x = self.deconv(x)
        x = nn.ReLU(inplace = True)(x)
        return x

class FphbNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 512):
        super().__init__()
        self.bottom_up_layer_1 = Layer2Conv(3, 64)
        self.bottom_up_layer_2 = Layer2Conv(64, 128)
        self.bottom_up_layer_3 = Layer3Conv(128, 256)
        self.bottom_up_layer_4 = Layer3Conv(256, 512)
        self.bottom_up_layer_5 = Layer3Conv(512, 512)
        self.final_layer =  nn.Sequential(
            nn.Conv2d(512, 3, kernel_size = (1, 1), stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = (2,2)),
            nn.InstanceNorm2d(num_features = out_channels),
            nn.ReLU(inplace = True)
        )
        self.size_network = SideNetwork()
        self.concat_layer = ConcatLayer()

    def forward(self, x):
        feature_map_layer_1 = self.bottom_up_layer_1(x)
        feature_map_layer_2 = self.bottom_up_layer_2(feature_map_layer_1)
        feature_map_layer_3 = self.bottom_up_layer_3(feature_map_layer_2)
        feature_map_layer_4 = self.bottom_up_layer_4(feature_map_layer_3)
        feature_map_layer_5 = self.bottom_up_layer_5(feature_map_layer_4)

        feature_pyramid_4 = self.concat_layer(feature_map_layer_4, feature_map_layer_5)
        feature_pyramid_3 = self.concat_layer(feature_map_layer_3, feature_pyramid_4)
        feature_pyramid_2 = self.concat_layer(feature_map_layer_2, feature_pyramid_3)
        feature_pyramid_1 = self.concat_layer(feature_map_layer_1, feature_pyramid_2)

        side_network_5 = self.size_network(feature_map_layer_5)
        side_network_4 = self.size_network(feature_pyramid_4)
        side_network_3 = self.size_network(feature_pyramid_3)
        side_network_2 = self.size_network(feature_pyramid_2)
        side_network_1 = self.size_network(feature_pyramid_1)

        final_concatenate = torch.cat([side_network_5, side_network_4, side_network_3, side_network_2, side_network_1], 1)
        final_outputs = final_layer(final_concatenate)
        return [side_network_1, side_network_2, side_network_3, side_network_4, side_network_5, final_outputs]