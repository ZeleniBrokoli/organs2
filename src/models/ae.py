#very basic autoencoder, found online, poor recovery results

# Jednostavan potpuno povezani autoenkoder
# Koristi se kao osnovni model za poređenje
# samo enkoder i dekoder sloj

import torch


class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 784 -> 18
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),

            torch.nn.Linear(36, 18)
        )

        # Decoder: 18 -> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),

            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),

            torch.nn.Linear(128, 28 * 28),

            # Ograničava izlaz na opseg [0, 1]
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Pretvaranje slike u vektor dimenzije 784
        x = torch.flatten(x, start_dim=1)

        # Latentna reprezentacija
        encoded = self.encoder(x)

        # Rekonstrukcija
        decoded = self.decoder(encoded)

        # Vraćanje u oblik slike
        decoded = decoded.view(-1, 1, 28, 28)

        return decoded


#ovo treba da se prebaci u main, ako ga koristim na kraju
# from models.ae import AE
# net_ae = AE()