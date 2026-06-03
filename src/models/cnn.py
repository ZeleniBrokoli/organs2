# Referentni CNN za poređenje sa autoenkoderom

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, init_features=4):
        super().__init__()

        features = init_features

        # Prvi konvolutivni blok
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Pooling smanjuje prostornu dimenziju slike
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=3)

        # Drugi konvolutivni blok
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, 2 * features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Automatsko određivanje dimenzije za prvi fully-connected sloj
        # Ovo je bolje nego da se ručno hardkoduje npr. 162
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            dummy = self.encoder1(dummy)
            dummy = self.pool1(dummy)
            dummy = self.encoder2(dummy)
            self._flattened_size = dummy.view(1, -1).size(1)

        # Klasifikacioni deo mreže
        self.fc = nn.Sequential(
            nn.Linear(self._flattened_size, 25),
            nn.ReLU(inplace=True),
            nn.Linear(25, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 11)
        )

    def forward(self, x):
        # Prolazak kroz konvolutivni deo
        x = self.encoder1(x)
        x = self.pool1(x)
        x = self.encoder2(x)

        # Pretvaranje u vektor pre fully-connected dela
        x = torch.flatten(x, 1)

        # Klasifikacija
        x = self.fc(x)
        return x


#ovo sam prebacila u main
# net_cnn = CNN(init_features=4)
# cnn_params = sum(p.numel() for p in net_cnn.parameters() if p.requires_grad)
#
# print('Trainable params:', cnn_params)