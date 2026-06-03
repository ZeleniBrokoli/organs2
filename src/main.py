import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Učitavanje već pripremljenih podataka iz Drive-a
pth = '/content/drive/MyDrive/organsDB/organsmnist.npz'

[X_train, y_train], [X_val, y_val], [X_test, y_test] = prepare_data(pth_to_data=pth)

# Po potrebi se može koristiti manji skup za brže testiranje
# X_train = X_train[:140]
# y_train = y_train[:140]

# Automatski izbor GPU-a ako je dostupan
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

from models.cnn import CNN

net_cnn = CNN(init_features=4)

cnn_params = sum(
    p.numel() for p in net_cnn.parameters()
    if p.requires_grad
)

print('Trainable params:', cnn_params)


from models.autoencoder import AutoEncoder
net_autoencoder = AutoEncoder()
cnn_params = sum(p.numel() for p in net_autoencoder.parameters() if p.requires_grad)

print('Trainable params:', cnn_params)


from models.classifier import AutoEncoderClass
import torch

net_aeclass = AutoEncoderClass()

net_aeclass.load_state_dict(
    torch.load('.../autoencoder.pt'),
    strict=False
)

from models.autoencoder2 import AutoEncoder2
net_autoencoder2 = AutoEncoder2()
cnn_params = sum(p.numel() for p in net_autoencoder2.parameters() if p.requires_grad)

print('Trainable params:', cnn_params)

from models.classifier2 import AutoEncoderClass2
import torch

net_aeclass2 = AutoEncoderClass2()

net_aeclass2.load_state_dict(
    torch.load('/content/drive/MyDrive/organsDB/autoencoder2_paper.pt'),
    strict=False
)

# Pretreniranje autoenkodera

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score  # nije potrebno za autoenkoder, ali može da ostane ako ga koristiš kasnije
import torch.optim as optim
from torch.utils.data import DataLoader

# Izbor modela za pretreniranje
net = net_autoencoder
net = net.to(device)

# Skupovi podataka
train_dataset = numpy_dataset(X_train, y_train)
val_dataset = numpy_dataset(X_val[:2000], y_val[:2000])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False)

# Funkcija greške za rekonstrukciju
class_loss = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
# Ako želiš SGD umesto Adam-a, koristi ovu liniju i obriši Adam:
# optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=0.001)

losses = []
max_epochs = 5

for epoch in range(1, max_epochs + 1):
    train_loss = train(net, train_dataloader, optimizer, class_loss, epoch, device)
    val_loss = val(net, val_dataloader, class_loss, epoch, device)
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
plt.show()


#saving the pretraining model
torch.save(net_autoencoder.state_dict(),'/content/drive/MyDrive/organsDB/autoencoder.pt')
torch.save(net_unet.state_dict(),'/content/drive/MyDrive/organsDB/autoencoder2.pt')
torch.save(net_autoencoder2.state_dict(),'/content/drive/MyDrive/organsDB/autoencoder2_paper.pt')


# Trening za klasifikaciju

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Dataset za treniranje i validaciju
train_dataset = numpy_dataset(X_train_aug, y_train_aug)
val_dataset = numpy_dataset(X_val[:30], y_val[:30])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False)

# Model
net = net_aeclass2.to(device)

# Za ovu verziju koristimo CrossEntropyLoss jer model vraća logits
class_loss = nn.CrossEntropyLoss()

# Ako želiš da treniraš samo classifier deo, ostavi ovako:
optimizer = optim.Adam(net.classifier.parameters(), lr=0.0001)

# Ako želiš fine-tuning celog modela, onda umesto gore koristi:
# optimizer = optim.Adam(net.parameters(), lr=0.0001)

losses = []
max_epochs = 20

for epoch in range(1, max_epochs + 1):
    train_loss, train_acc = train2(net, train_dataloader, optimizer, class_loss, epoch, device)
    val_loss, val_acc = val2(net, val_dataloader, class_loss, epoch, device)
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
plt.show()

from torch.utils.data import DataLoader

test_dataset = numpy_dataset(X_test, y_test)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False
)

pred, true = predict(
    net_aeclass2,
    test_dataloader,
    device
)

test_dataset = numpy_dataset(X_test, y_test)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    drop_last=False
)

pred, true = predict2(
    net_autoencoder,
    test_dataloader,
    device
)