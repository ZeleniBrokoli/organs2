# Trening i validacija za pretreniranje autoenkodera
# Ulaz se dodatno šumi, a cilj je da mreža rekonstruiše originalnu sliku

import time
import torch


def train(net, dataloader, optimizer, loss_func, epoch, device, log_dict=None):
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

    if log_dict is not None:
        log_dict.setdefault("epochs", []).append({
            "epoch": int(epoch),
            "train_loss": float(av_loss),
            "time": float(t1 - t0)
        })

    return av_loss


def val(net, val_dataloader, loss_func, epoch, device, log_dict=None):
    net.eval()  # mreža je u eval modu

    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(val_dataloader):
            data = data.to(device)
            target = data.clone()

            batches += 1

            # I ovde može da se doda šum ako želiš da meriš denoising sposobnost
            noise = torch.randn_like(data) * 0.1
            data_noisy = data + noise

            prediction = net(data_noisy)
            loss = loss_func(prediction, target)

            total_loss += loss.item()

    av_loss = total_loss / batches

    print('Validation set: Average loss: {:.4f}'.format(av_loss), flush=True)
    print('\n')

    if log_dict is not None:
        log_dict.setdefault("epochs", []).append({
            "epoch": int(epoch),
            "val_loss": float(av_loss)
        })

    return av_loss



#saving the pretraining model OVO PREBACILA U MAIN
# torch.save(net_autoencoder.state_dict(),'/content/drive/MyDrive/organsDB/autoencoder.pt')
#
# torch.save(net_unet.state_dict(),'/content/drive/MyDrive/organsDB/autoencoder2.pt')
#
# torch.save(net_autoencoder2.state_dict(),'/content/drive/MyDrive/organsDB/autoencoder2_paper.pt')