# Napuštena ideja: veoma jednostavan linearni autoenkoder
# Koristi se samo za eksperimentisanje

import torch
import torch.nn as nn


class Simple(nn.Module):
    def __init__(self, in_channels=1, init_features=4):
        super().__init__()

        # Jedan linearni sloj koji mapira 784 -> 784
        self.linear = nn.Linear(784, 784)

    def forward(self, x):
        # Pretvaranje slike iz oblika (B, 1, 28, 28) u (B, 784)
        x = torch.flatten(x, start_dim=1)

        # Linearna transformacija
        x = self.linear(x)

        # Vraćanje u oblik slike
        x = x.view(x.size(0), 1, 28, 28)

        return x

#ovo prebaci u main ako ti treba ovde ne
# net_simple = Simple()