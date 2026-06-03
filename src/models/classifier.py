import torch
import torch.nn as nn
from collections import OrderedDict


class AutoEncoderClass(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = AutoEncoderClass._block1(1, 6, 7, 3)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = AutoEncoderClass._block1(6, 12, 5, 2)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Conv2d(12, 18, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(18 * 3 * 3, 512)),
            ('relu1', nn.ReLU(inplace=True)),
            ('lin2', nn.Linear(512, 64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('lin3', nn.Linear(64, 11)),
        ]))

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool1(x)

        x = self.encoder2(x)
        x = self.pool2(x)

        x = self.encoder3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _block1(in_channels, features, k, p):
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, features, kernel_size=k, padding=p)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(features, features, kernel_size=k, padding=0)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))