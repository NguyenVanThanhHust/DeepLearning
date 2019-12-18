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
        size_input = list(input_layer.size())
        upsample_map = nn.Upsample((size_input[2], size_input[3]), mode='bicubic')(up_input_layer)
        size_upsample = list(upsample_map.size())
        concat = torch.cat([input_layer, upsample_map], 1)
        reduce_depth =  nn.Conv2d(size_input[1] +size_upsample[1], 128, kernel_size = (1,1), stride = 1, padding = 1)(concat)
        # norm = nn.InstanceNorm2d(num_features = 128)(reduce_depth)
        # output = nn.ReLU(inplace = True)(norm)
        return output
            
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
        self.bottom_up_layer_3 = Layer3Conv(128, 128)
        self.final_layer =  nn.Sequential(
            nn.Conv2d(128, 3, kernel_size = (1, 1), stride = 1, padding = 1),
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

        feature_pyramid_3 = self.concat_layer(feature_map_layer_2, feature_map_layer_3)
        feature_pyramid_2 = self.concat_layer(feature_map_layer_2, feature_pyramid_3)
        feature_pyramid_1 = self.concat_layer(feature_map_layer_1, feature_pyramid_2)

        side_network_3 = self.size_network(feature_map_layer_3)
        side_network_2 = self.size_network(feature_pyramid_2)
        side_network_1 = self.size_network(feature_pyramid_1)

        final_concatenate = torch.cat([ side_network_3, side_network_2, side_network_1], 1)
        final_outputs = final_layer(final_concatenate)
        return [side_network_1, side_network_2, side_network_3, final_outputs]