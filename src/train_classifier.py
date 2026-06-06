# Trening i validacija za klasifikaciju
# Ova verzija radi sa izlazom mreže bez softmax-a i sa CrossEntropyLoss

import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def train2(net, dataloader, optimizer, loss_func, epoch, device, log_dict=None):
    net.train()  # mreža je u train modu

    total_loss = 0.0
    pred_store = []
    true_store = []
    batches = 0

    t0 = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        # Slanje podataka na uređaj
        data = data.to(device)
        target = target.to(device)

        # Ako su labele one-hot, pretvaramo ih u indekse klasa
        if target.ndim > 1 and target.size(1) > 1:
            target_cls = torch.argmax(target, dim=1).long()
        else:
            target_cls = target.view(-1).long()

        batches += 1

        # Reset gradijenata
        optimizer.zero_grad()

        # Forward pass
        prediction = net(data)

        # Loss
        loss = loss_func(prediction, target_cls)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Predikcija i tačne klase za accuracy
        pred_cls = torch.argmax(prediction, dim=1)

        pred_store.append(pred_cls.detach().cpu().numpy())
        true_store.append(target_cls.detach().cpu().numpy())

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
    pred_store = np.concatenate(pred_store)
    true_store = np.concatenate(true_store)
    acc = accuracy_score(true_store, pred_store)

    print('\nTraining set: Average loss: {:.4f}'.format(av_loss), flush=True)
    print('Training set: Average Acc: {:.4f}'.format(acc), flush=True)
    print('Time for epoch = ', t1 - t0)

    # Čuvanje rezultata epohe za kasniji JSON izveštaj
    if log_dict is not None:
        log_dict["train"].append({
            "epoch": int(epoch),
            "loss": float(av_loss),
            "accuracy": float(acc),
            "time": float(t1 - t0)
        })

    return av_loss, acc


def val2(net, val_dataloader, loss_func, epoch, device, log_dict=None):
    net.eval()  # mreža je u eval modu

    total_loss = 0.0
    pred_store = []
    true_store = []
    batches = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloader):
            data = data.to(device)
            target = target.to(device)

            # Ako su labele one-hot, pretvaramo ih u indekse klasa
            if target.ndim > 1 and target.size(1) > 1:
                target_cls = torch.argmax(target, dim=1).long()
            else:
                target_cls = target.view(-1).long()

            batches += 1

            prediction = net(data)
            loss = loss_func(prediction, target_cls)

            total_loss += loss.item()

            pred_cls = torch.argmax(prediction, dim=1)

            pred_store.append(pred_cls.detach().cpu().numpy())
            true_store.append(target_cls.detach().cpu().numpy())

    av_loss = total_loss / batches
    pred_store = np.concatenate(pred_store)
    true_store = np.concatenate(true_store)
    acc = accuracy_score(true_store, pred_store)

    print('Validation set: Average loss: {:.4f}'.format(av_loss), flush=True)
    print('Validation set: Average Acc: {:.4f}'.format(acc), flush=True)
    print('\n')

    # Čuvanje rezultata validacije za kasniji JSON izveštaj
    if log_dict is not None:
        log_dict["val"].append({
            "epoch": int(epoch),
            "loss": float(av_loss),
            "accuracy": float(acc)
        })

    return av_loss, acc