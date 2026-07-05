import numpy as np
import matplotlib.pyplot as plt
import torch


# Vizualizacija originalne slike, slike sa šumom i rekonstrukcije autoenkodera
def show_reconstruction(model, X_test, device, idx=89, save_path=None):

    model.eval()

    # Originalna slika
    original = X_test[idx][0]

    # Dodavanje Gaussian šuma
    def add_noise(img):
        noise = np.random.normal(0.0, 0.1, img.shape).astype(np.float32)
        return img + noise

    noisy = add_noise(original)

    # Priprema ulaza za mrežu: (1, 1, 28, 28)
    x = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstruction = model(x)

    # Pretvaranje izlaza u NumPy format za prikaz
    reconstruction = reconstruction.squeeze().cpu().numpy()
    reconstruction = np.clip(reconstruction, 0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Sa šumom')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction, cmap='gray')
    plt.title('Rekonstrukcija')
    plt.axis('off')

    plt.tight_layout()

    # Čuvanje slike ako je zadat izlazni fajl
    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.show()
    plt.close()