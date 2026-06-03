# Evaluacija modela na test skupu

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def predict_classifier(net, test_dataloader, device):
    # Predikcija za klasifikacioni model
    net.eval()

    pred_store = []
    true_store = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data = data.to(device)
            target = target.to(device)

            prediction = net(data)

            # Ako su labele one-hot, pretvaramo ih u indekse klasa
            if target.ndim > 1 and target.size(1) > 1:
                target_cls = torch.argmax(target, dim=1)
            else:
                target_cls = target.view(-1)

            pred_cls = torch.argmax(prediction, dim=1)

            pred_store.append(pred_cls.cpu().numpy())
            true_store.append(target_cls.cpu().numpy())

    pred_store = np.concatenate(pred_store)
    true_store = np.concatenate(true_store)

    acc = accuracy_score(true_store, pred_store)

    print('Test set: Average Acc: {:.4f}'.format(acc), flush=True)
    print('\n')

    return pred_store, true_store, acc


def predict_autoencoder(net, test_dataloader, device):
    # Procena autoenkodera na test skupu
    # Računa prosečnu MSE grešku rekonstrukcije
    net.eval()

    pred_store = []
    true_store = []
    total_loss = 0.0
    batches = 0

    class_loss = nn.MSELoss()

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_dataloader):
            data = data.to(device)
            target = data.clone()  # cilj je originalna slika

            batches += 1

            prediction = net(data)
            loss = class_loss(prediction, target)

            total_loss += loss.item()

            pred_store.append(prediction.detach().cpu().numpy())
            true_store.append(target.detach().cpu().numpy())

    av_loss = total_loss / batches

    pred_store = np.concatenate(pred_store, axis=0)
    true_store = np.concatenate(true_store, axis=0)

    print('Test set: Average loss: {:.4f}'.format(av_loss), flush=True)
    print('\n')

    return pred_store, true_store, av_loss