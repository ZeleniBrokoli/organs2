import numpy as np
import warnings
import torch
from torch.utils.data import Dataset

warnings.filterwarnings(action='ignore', category=FutureWarning)


# Standardizacija podataka po pikselima:
# oduzima se srednja vrednost i deli standardnom devijacijom
def centring(X):
    epsilon = 1e-7  # zaštita od deljenja nulom
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + epsilon)
    return X


# Pretvaranje labela u one-hot zapis
def to_one_hot(y, num_classes):
    y = y.squeeze()
    return np.eye(num_classes)[y]


def prepare_data(pth_to_data):
    print('Loading Data')

    # Učitavanje podataka
    dataset = np.load(pth_to_data)
    X_train, y_train = dataset['train_images'], dataset['train_labels']
    X_val, y_val = dataset['val_images'], dataset['val_labels']
    X_test, y_test = dataset['test_images'], dataset['test_labels']

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    # Prikaz raspodele klasa u svakom skupu
    print('Data Splits')

    print('Train set:')
    unique, counts = np.unique(y_train, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f'Label {c}: {cnt}')

    print('Val set:')
    unique, counts = np.unique(y_val, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f'Label {c}: {cnt}')

    print('Test set:')
    unique, counts = np.unique(y_test, return_counts=True)
    for c, cnt in zip(unique, counts):
        print(f'Label {c}: {cnt}')

    # Standardizacija podataka
    print('Centring Data')
    X_train = centring(X_train)
    X_val = centring(X_val)
    X_test = centring(X_test)

    # Dodavanje kanala za CNN
    X_train = np.reshape(X_train, (-1, 1, 28, 28))
    X_val = np.reshape(X_val, (-1, 1, 28, 28))
    X_test = np.reshape(X_test, (-1, 1, 28, 28))

    # Pretvaranje labela u one-hot zapis
    print('Converting labels to one hot')
    num_classes = len(np.unique(y_train))
    y_train = to_one_hot(y_train, num_classes)
    y_val = to_one_hot(y_val, num_classes)
    y_test = to_one_hot(y_test, num_classes)

    print('X_data: ', X_train.shape, X_val.shape, X_test.shape)
    print('y data: ', y_train.shape, y_val.shape, y_test.shape)

    return [X_train, y_train], [X_val, y_val], [X_test, y_test]


# Skup podataka napravljen od NumPy nizova
# Koristi se za učitavanje podataka u PyTorch DataLoader
class NumpyDataset(Dataset):
    def __init__(self, data, target):
        # Pretvaranje NumPy nizova u torch tensor
        self.data = torch.from_numpy(data).float()  # posle za CE mi treba long
        self.target = torch.from_numpy(target).float()

    def __getitem__(self, index):
        # Vraća jedan uzorak i odgovarajuću labelu
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        # Ukupan broj uzoraka u skupu
        return len(self.data)