# Autoenkoder iz rada [1]
# Ova verzija čuva istu ideju arhitekture, ali je kod sređen i dosledan

import torch
import torch.nn as nn
from collections import OrderedDict


class AutoEncoder2(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder deo
        self.encoder1 = AutoEncoder2._block(1, 6, 7, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = AutoEncoder2._block(6, 12, 5, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Conv2d(12, 18, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv2d(18, 12, kernel_size=3, padding=1)

        # Decoder deo
        self.unpool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder2 = AutoEncoder2._block2(12, 6, 5, 2)

        self.unpool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder1 = AutoEncoder2._block2(6, 1, 7, 3)

        # Završna konvolucija za finu rekonstrukciju
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.encoder1(x)
        x = self.pool1(x)

        x = self.encoder2(x)
        x = self.pool2(x)

        x = self.encoder3(x)
        x = self.encoder4(x)

        # Decoder
        x = self.unpool2(x)
        x = self.decoder2(x)

        x = self.unpool1(x)
        x = self.decoder1(x)

        x = self.conv(x)

        return x

    @staticmethod
    def _block(in_channels, features, k, p):
        # Konvolutivni blok u encoder-u
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, features, kernel_size=k, padding=p)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(features, features, kernel_size=k, padding=p)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

    @staticmethod
    def _block2(in_channels, features, k, p):
        # Konvolutivni blok u decoder-u
        return nn.Sequential(OrderedDict([
            ('conv1', nn.ConvTranspose2d(in_channels, features, kernel_size=k, padding=(k - 1) // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.ConvTranspose2d(features, features, kernel_size=k, padding=(k - 1) // 2)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))