# Glavni fajl projekta:
# 1) učitavanje i pregled podataka
# 2) pretreniranje autoenkodera
# 3) klasifikacija sa prenetim težinama
# 4) test evaluacija

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data import prepare_data, NumpyDataset
from augment import augment_data
from models.autoencoder2 import AutoEncoder2
from models.classifier2 import AutoEncoderClass2
from train_autoencoder import train, val
from train_classifier import train2, val2
from eval import predict_classifier, predict_autoencoder
from visualize import show_reconstruction
from utils.logger import create_run_dir, save_json


# Putanje
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "organsmnist.npz"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"


def show_class_distribution(y_train, y_val, y_test):
    # Vizualizacija raspodele klasa u train, validation i test skupu
    labels = np.arange(11)

    train_counts = np.sum(y_train, axis=0)
    val_counts = np.sum(y_val, axis=0)
    test_counts = np.sum(y_test, axis=0)

    print("Train distribucija:")
    print(train_counts)

    print("Validation distribucija:")
    print(val_counts)

    print("Test distribucija:")
    print(test_counts)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.pie(train_counts, labels=labels)
    plt.title("Train")

    plt.subplot(1, 3, 2)
    plt.pie(val_counts, labels=labels)
    plt.title("Validation")

    plt.subplot(1, 3, 3)
    plt.pie(test_counts, labels=labels)
    plt.title("Test")

    plt.tight_layout()
    plt.show()


def run_autoencoder_pretraining(X_train, y_train, X_val, y_val, device, run_dir, ae_log):
    # Pretreniranje autoenkodera

    net_autoencoder = AutoEncoder2().to(device)

    # Skupovi podataka
    # train_dataset = NumpyDataset(X_train, y_train) VRV TREBA OVO OVO JE TEST OVAKO
    train_dataset = NumpyDataset(X_train, X_train)
    # val_dataset = NumpyDataset(X_val, y_val) OVAKO TREBA VRV OVO JE TEST
    val_dataset = NumpyDataset(X_val, X_val)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False)

    # Funkcija greške za rekonstrukciju
    class_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net_autoencoder.parameters(), lr=0.001)
    # Ako želiš SGD umesto Adam-a, koristi ovu liniju i obriši Adam:
    # optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=0.001)

    losses = []
    max_epochs = 5

    for epoch in range(1, max_epochs + 1):
        train_loss = train(
            net_autoencoder,
            train_dataloader,
            optimizer,
            class_loss,
            epoch,
            device,
            log_dict=ae_log
        )
        val_loss = val(
            net_autoencoder,
            val_dataloader,
            class_loss,
            epoch,
            device,
            log_dict=ae_log
        )
        losses.append([train_loss, val_loss])

    losses = np.array(losses).T
    print(losses.shape)

    epochs = np.arange(1, max_epochs + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses[0, :], label='Train')
    plt.plot(epochs, losses[1, :], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "autoencoder_loss.png", dpi=200)
    plt.close()

    # Čuvanje loga pretreniranja
    save_json(ae_log, run_dir / "autoencoder_log.json")

    # Saving the pretraining model
    autoencoder_path = run_dir / "autoencoder2_paper.pt"
    torch.save(net_autoencoder.state_dict(), autoencoder_path)

    print(f"Model sačuvan u: {autoencoder_path}")

    return net_autoencoder, autoencoder_path


def run_classifier_training(X_train, y_train, X_val, y_val, X_test, y_test, device, run_dir, classifier_log, autoencoder_path):
    # Data augmentation
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    # Klasifikacija sa prenetim težinama
    net_aeclass2 = AutoEncoderClass2().to(device)

    # Učitavanje težina iz pretreniranja, uz strict=False jer classifier nije isti
    net_aeclass2.load_state_dict(
        torch.load(str(autoencoder_path), map_location=device),
        strict=False
    )

    # Dataset za treniranje i validaciju
    train_dataset = NumpyDataset(X_train_aug, y_train_aug)
    val_dataset = NumpyDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False)

    # Za ovu verziju koristimo CrossEntropyLoss jer model vraća logits
    class_loss = nn.CrossEntropyLoss()

    # Ako želiš da treniraš samo classifier deo, ostavi ovako:
    # optimizer = optim.Adam(net_aeclass2.classifier.parameters(), lr=0.0001)

    #TRECA SRECA
    optimizer = optim.Adam(net_aeclass2.parameters(), lr=1e-4)

    # prvo sam probala ovo
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)

    losses = []
    max_epochs = 20

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train2(
            net_aeclass2,
            train_dataloader,
            optimizer,
            class_loss,
            epoch,
            device,
            log_dict=classifier_log
        )
        val_loss, val_acc = val2(
            net_aeclass2,
            val_dataloader,
            class_loss,
            epoch,
            device,
            log_dict=classifier_log
        )
        losses.append([train_loss, train_acc, val_loss, val_acc])

    losses = np.array(losses).T
    print(losses.shape)

    epochs = np.arange(1, max_epochs + 1)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses[0, :])
    plt.plot(epochs, losses[2, :])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses[1, :])
    plt.plot(epochs, losses[3, :])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.savefig(run_dir / "classifier_metrics.png", dpi=200)
    plt.close()

    # Čuvanje loga klasifikatora
    save_json(classifier_log, run_dir / "classifier_log.json")

    # Test evaluacija klasifikatora
    test_dataset = NumpyDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    pred, true, acc = predict_classifier(net_aeclass2, test_dataloader, device)
    print("Final test accuracy:", acc)

    return net_aeclass2, acc, pred, true


def run_autoencoder_test(X_test, y_test, device, net_autoencoder):
    # Test procena autoenkodera
    test_dataset = NumpyDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)

    pred, true, av_loss = predict_autoencoder(net_autoencoder, test_dataloader, device)
    print("Final test reconstruction loss:", av_loss)

    return pred, true, av_loss


def main():
    # Učitavanje već pripremljenih podataka iz foldera data
    [X_train, y_train], [X_val, y_val], [X_test, y_test] = prepare_data(str(DATA_PATH))

    # Po potrebi se može koristiti manji skup za brže testiranje
    # X_train = X_train[:140]
    # y_train = y_train[:140]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Prikaz raspodele klasa
    show_class_distribution(y_train, y_val, y_test)

    # Kreiranje foldera za težine ako ne postoji
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Folder za ovaj run
    run_dir = create_run_dir()

    # Logovi za ovaj run
    ae_log = {"epochs": []}
    classifier_log = {"train": [], "val": []}

    num_labeled = int(0.01 * len(X_train))

    indices = np.random.choice(len(X_train), num_labeled, replace=False)

    X_labeled = X_train[indices]
    y_labeled = y_train[indices]

    # 1) Pretreniranje autoenkodera
    net_autoencoder, autoencoder_path = run_autoencoder_pretraining(
        X_train, y_train, X_val, y_val, device, run_dir, ae_log
    )

    # 2) Test autoenkodera
    _, _, ae_test_loss = run_autoencoder_test(X_test, y_test, device, net_autoencoder)

    # 3) Klasifikacija sa prenetim težinama
    net_aeclass2, clf_test_acc, pred, true = run_classifier_training(
        X_labeled, y_labeled, X_val, y_val, X_test, y_test, device, run_dir, classifier_log, autoencoder_path
    )

    # Sažetak rezultata
    save_json(
        {
            "autoencoder_test_loss": float(ae_test_loss),
            "classifier_test_accuracy": float(clf_test_acc)
        },
        run_dir / "summary.json"
    )

    # Predikcije testa (ne koristim ovo, ali neka stoji zakomentarisano, jer postoji u prethodnim pokretanjima)
    # save_json(
    #     {
    #         "predictions": np.asarray(pred).tolist(),
    #         "targets": np.asarray(true).tolist()
    #     },
    #     run_dir / "test_predictions.json"
    # )

    show_reconstruction(
        net_autoencoder,
        X_test,
        device,
        save_path=run_dir / "reconstruction_example.png"
    )


if __name__ == "__main__":
    main()