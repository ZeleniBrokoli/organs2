# Pojednostavljena verzija autoenkodera iz rada [1]
# Ideja je da se sačuva osnovna struktura, ali da kod bude stabilniji i čistiji
# mreza prima sliku, dodaje sum tokom treninga, onda uci da iz sumne verzije rekonstruise original
# kao klasican denoising autoenkoder
# u odnosu na [1] izbaceno pseudo-vezbanje

import time
import torch
import torch.nn as nn
from collections import OrderedDict


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder deo
        self.encoder1 = AutoEncoder._block1(1, 6, 7)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = AutoEncoder._block1(6, 12, 5)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = nn.Conv2d(12, 18, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv2d(18, 12, kernel_size=3, padding=1)

        # Umesto scale_factor=7/3 koristimo fiksne dimenzije da ne zavisi od zaokruživanja
        self.unpool2 = nn.Upsample(size=(7, 7), mode='nearest')

        # Decoder deo
        self.decoder2 = AutoEncoder._block2(12, 6, 5)
        self.unpool1 = nn.Upsample(size=(22, 22), mode='nearest')
        self.decoder1 = AutoEncoder._block2(6, 1, 7)

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

        return x

    @staticmethod
    def _block1(in_channels, features, k):
        # Konvolutivni blok u encoder-u
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, features, kernel_size=k, padding=k // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(features, features, kernel_size=k, padding=0)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

    @staticmethod
    def _block2(in_channels, features, k):
        # Transponovani konvolutivni blok u decoder-u
        return nn.Sequential(OrderedDict([
            ('conv1', nn.ConvTranspose2d(in_channels, features, kernel_size=k, padding=(k - 1) // 2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.ConvTranspose2d(features, features, kernel_size=k, padding=0)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))


# Trening i validacija za pretreniranje autoenkodera
# Ulaz se dodatno šumi, a cilj je da mreža rekonstruiše originalnu sliku

def train(net, dataloader, optimizer, loss_func, epoch, device):
    net.train()  # mreža je u train modu

    total_loss = 0.0
    batches = 0
    t0 = time.time()

    for batch_idx, (data, _) in enumerate(dataloader):
        # Za autoenkoder je cilj isti kao ulaz
        data = data.to(device)
        target = data.clone()

        batches += 1

        # Dodavanje Gaussian šuma
        noise = torch.randn_like(data) * 0.1
        data_noisy = data + noise

        # Resetovanje gradijenata
        optimizer.zero_grad()

        # Forward i loss
        prediction = net(data_noisy)
        loss = loss_func(prediction, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(dataloader.dataset),
                    100. * (batch_idx + 1) / len(dataloader),
                    loss.item()
                ),
                flush=True
            )

    t1 = time.time()
    av_loss = total_loss / batches

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss), flush=True)
    print('Time for epoch = ', t1 - t0)

    return av_loss


def val(net, val_dataloader, loss_func, epoch, device):
    net.eval()  # mreža je u eval modu

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_dataloader):
            data = data.to(device)
            target = data.clone()

            batches += 1

            # i ovde može da se doda šum ako želimo da merimo denoising sposobnost
            noise = torch.randn_like(data) * 0.1
            data_noisy = data + noise

            prediction = net(data_noisy)
            loss = loss_func(prediction, target)

            total_loss += loss.item()

    av_loss = total_loss / batches

    print('Validation set: Average loss: {:.4f}'.format(av_loss), flush=True)
    print('\n')

    return av_loss