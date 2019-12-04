import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.outchannels = out_channels
    def forward(self, x):
        return 

class FphbNet(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 512):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(self.in_channels, out_channels = 64, kernel_size = 3, padding = 35)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)
        self.conv7 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1)
        self.conv8 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1)

        self.conv_fuse_5 = nn.Conv2d(in_channels = 1536, out_channels = 512, kernel_size = 3, padding = 1, )
        self.deconv_5 = torch.nn.ConvTranspose2d(512, 512, kernel_size = 32, stride = 16)
        self.conv_fuse_4 = nn.Conv2d(in_channels = 1536, out_channels = 512, kernel_size = 3, padding = 1, )
        self.deconv_4 = torch.nn.ConvTranspose2d(256, 256, kernel_size = 32, stride = 16)
        self.conv_fuse_3 = nn.Conv2d(in_channels = 1536, out_channels = 512, kernel_size = 3, padding = 1, )
        self.deconv_3 = torch.nn.ConvTranspose2d(512, 512, kernel_size = 32, stride = 16)

    def forward(self, x):
        conv1_1 = self.conv1(x)
        relu1_1 = F.relu(conv1_1)
        conv1_2 = self.conv2(relu1_1)
        relu1_2 = F.relu(conv1_1)
        pool1 = F.max_pool2d(relu1_2, (2, 2))

        conv2_1 = self.conv3(pool1)
        relu2_1 = F.relu(conv2_1)
        conv2_2 = self.conv4(relu2_1)
        relu2_2 = F.relu(conv2_2)
        pool2 = F.max_pool2d(relu2_2, (2, 2))

        conv3_1 = self.conv5(pool2)
        relu3_1 = F.relu(conv3_1)
        conv3_2 = self.conv6(relu3_1)
        relu3_2 = F.relu(conv3_2)
        conv3_3 = self.conv6(relu3_2)
        relu3_3 = F.relu(conv3_3)
        pool3 = F.max_pool2d(relu3_3, (2, 2))

        conv4_1 = self.conv7(pool3)
        relu4_1 = F.relu(conv4_1)
        conv4_2 = self.conv8(relu4_1)
        relu4_2 = F.relu(conv4_2)
        conv4_3 = self.conv8(relu4_2)
        relu4_3 = F.relu(conv4_3)
        pool4 = F.max_pool2d(relu4_3, (2, 2))

        conv5_1 = self.conv7(pool4)
        relu5_1 = F.relu(conv5_1)
        conv5_2 = self.conv8(relu5_1)
        relu5_2 = F.relu(conv5_2)
        conv5_3 = self.conv8(relu5_2)
        relu5_3 = F.relu(conv5_3)
        pool5 = F.max_pool2d(relu5_3, (2, 2))

        concat_conv5 = torch.cat([conv5_3, conv5_2, conv5_1], 0)

        conv5_fuse = self.conv_fuse(concat_conv5)
        score_dsn5 = self.conv8(conv5_fuse)
        upsample_16 = self.deconv(score_dsn5)

        concat_conv4 = torch.cat([conv4_3, conv4_2, conv4_1], 0)
        conv4_fuse = self.conv_fuse(concat_conv4)
        score_dsn4 = self.conv8(conv5_fuse)
        concat_54 = torch.cat([upsample_16, score_dsn4], 0)
        



        return F.upsample_bilinear(score, x.size()[2:])


class FCN16(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        feats = list(models.vgg16(pretrained=True).features.children())
        self.feats = nn.Sequential(*feats[0:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.score_fconn = nn.Conv2d(4096, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        feat4 = self.feat4(feats)
        feat5 = self.feat5(feat4)
        fconn = self.fconn(feat5)

        score_feat4 = self.score_feat4(feat4)
        score_fconn = self.score_fconn(fconn)

        score = F.upsample_bilinear(score_fconn, score_feat4.size()[2:])
        score += score_feat4

        return F.upsample_bilinear(score, x.size()[2:])

class UNetDec(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers += [nn.Dropout(.5)]
        layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.dec1 = UNetDec(3, 64)
        self.dec2 = UNetDec(64, 128)
        self.dec3 = UNetDec(128, 256)
        self.dec4 = UNetDec(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc4 = UNetEnc(1024, 512, 256)
        self.enc3 = UNetEnc(512, 256, 128)
        self.enc2 = UNetEnc(256, 128, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        center = self.center(dec4)
        enc4 = self.enc4(torch.cat([
            center, F.upsample_bilinear(dec4, center.size()[2:])], 1))
        enc3 = self.enc3(torch.cat([
            enc4, F.upsample_bilinear(dec3, enc4.size()[2:])], 1))
        enc2 = self.enc2(torch.cat([
            enc3, F.upsample_bilinear(dec2, enc3.size()[2:])], 1))
        enc1 = self.enc1(torch.cat([
            enc2, F.upsample_bilinear(dec1, enc2.size()[2:])], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])
