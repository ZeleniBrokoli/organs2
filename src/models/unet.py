# U-Net varijanta korišćena kao autoenkoder (abandoned idea)
# Ideja je da se slika prvo enkodira u latentni prostor, pa zatim rekonstruiše

import torch
import torch.nn as nn
from collections import OrderedDict


class UNet(nn.Module):
    def __init__(self, in_channels=1, init_features=4, out_channels=1):
        super().__init__()

        features = init_features

        # Encoder deo
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(features, 2 * features)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        # Nakon dve pooling operacije, dimenzije slike su 7x7 za ulaz 28x28
        self.flatten_size = 2 * features * 7 * 7
        self.bottleneck1 = nn.Linear(self.flatten_size, 100)
        self.bottleneck2 = nn.Linear(100, 20)
        self.bottleneck3 = nn.Linear(20, 100)
        self.bottleneck4 = nn.Linear(100, self.flatten_size)

        self.dropout = nn.Dropout(0.2)

        # Decoder deo
        self.upconv2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder2 = UNet._block(2 * features, features)

        self.upconv1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.decoder1 = UNet._block(features, out_channels)

        # Završna konvolucija za finu rekonstrukciju
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        s1 = self.encoder1(x)
        x = self.pool1(s1)

        s2 = self.encoder2(x)
        x = self.pool2(s2)

        # Bottleneck
        x = torch.flatten(x, start_dim=1)
        x = self.bottleneck1(x)
        x = torch.relu(x)
        x = self.bottleneck2(x)
        x = torch.relu(x)
        x = self.bottleneck3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.bottleneck4(x)

        # Vraćanje u prostorni oblik
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, 7, 7)

        # Decoder
        x = self.upconv2(x)
        x = self.decoder2(x)

        x = self.upconv1(x)
        x = self.decoder1(x)

        x = self.conv(x)

        return x

    @staticmethod
    def _block(in_channels, features):
        # Jedan konvolutivni blok: Conv -> ReLU -> BatchNorm -> Conv -> ReLU -> BatchNorm
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, features, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU(inplace=True)),
            ('bn1', nn.BatchNorm2d(features)),
            ('conv2', nn.Conv2d(features, features, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('bn2', nn.BatchNorm2d(features)),
        ]))


#ovo prebaci u main, ali vidi je l ces da ga obrises prvo ili ostaje
# from models.unet import UNet
# net_unet = UNet(in_channels=1, init_features=4, out_channels=1)
# cnn_params = sum(p.numel() for p in net_unet.parameters() if p.requires_grad)
#
# print('Trainable params:', cnn_params)
```

Pošalji sledeću skriptu.
