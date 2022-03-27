import torch
import torch.nn as nn
from .unet_modules import DoubleConv, DownSampleConv, UpSampleConv, OutputConv


class UNet(nn.Module):
    """U-Net implementation"""
    def __init__(self, config):
        super(UNet, self).__init__()
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.bilinear = config['bilinear_sample']
        self.config = config

        self.in_conv = DoubleConv(self.in_channels, 64)
        self.down1 = DownSampleConv(64, 128)
        self.down2 = DownSampleConv(128, 256)
        self.down3 = DownSampleConv(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = DownSampleConv(512, 1024 // factor)
        self.up1 = UpSampleConv(1024, 512 // factor, self.bilinear)
        self.up2 = UpSampleConv(512, 256 // factor, self.bilinear)
        self.up3 = UpSampleConv(256, 128 // factor, self.bilinear)
        self.up4 = UpSampleConv(128, 64, self.bilinear)
        self.out_conv = OutputConv(64, self.out_channels)
        self.loss_func = nn.MSELoss()

    def forward(self, data, mode):
        x, labels = data['inputs'], data['labels']
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        assert x.shape == data['inputs'].shape
        if mode != 'test':
            loss = self.loss_func(x, labels)
            return loss
        else:
            if self.config['scale']:
                x *= 256.0
            x = torch.clamp(x, 0, 255)
            return x, labels
