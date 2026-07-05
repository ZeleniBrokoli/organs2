# Klasifikator/encoder zasnovan na arhitekturi iz rada [1]
# Izlaz vraća logits, bez softmax-a, pa je model spreman za CrossEntropyLoss

import torch
import torch.nn as nn
from collections import OrderedDict


class AutoEncoderClass2(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder deo
        self.encoder1 = AutoEncoderClass2._block(1, 6, 7, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = AutoEncoderClass2._block(6, 12, 5, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Conv2d(12, 18, kernel_size=3, padding=1)

        # Posle encoder dela dobijamo dimenziju 18 x 7 x 7
        self.classifier = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(18 * 7 * 7, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('lin2', nn.Linear(512, 64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('lin3', nn.Linear(64, 11)),
        ]))

    def forward(self, x):
        # Encoder
        x = self.encoder1(x)
        x = self.pool1(x)

        x = self.encoder2(x)
        x = self.pool2(x)

        x = self.encoder3(x)

        # Pretvaranje mape osobina u vektor
        x = torch.flatten(x, start_dim=1)

        # Klasifikacioni deo
        x = self.classifier(x)

        return x

    @staticmethod
    def _block(in_channels, features, k, p):
        # Konvolutivni blok korišćen u encoder-u
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, features, kernel_size=k, padding=p)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(features, features, kernel_size=k, padding=p)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

