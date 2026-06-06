import numpy as np
from scipy.ndimage import rotate


# Dodavanje Gaussian šuma
def add_noise(data):
    noise = np.random.normal(0.0, 0.1, data.shape).astype(np.float32)
    return data + noise


# Data augmentation
def augment_data(X_train, y_train):
    # Rotacija za +-15 stepeni, kad je bila rotacija od 90, bila je tacnost 37%
    X_train_rot_pos = np.array([
        rotate(img, angle=15, reshape=False, order=1, mode='nearest')
        for img in X_train[:110, 0]
    ])
    X_train_rot_pos = X_train_rot_pos.reshape(-1, 1, 28, 28)

    X_train_rot_neg = np.array([
        rotate(img, angle=-15, reshape=False, order=1, mode='nearest')
        for img in X_train[:110, 0]
    ])
    X_train_rot_neg = X_train_rot_neg.reshape(-1, 1, 28, 28)

    # Vertikalno preslikavanje
    X_train_flip = np.array([np.flipud(img) for img in X_train[:110, 0]])
    X_train_flip = X_train_flip.reshape(-1, 1, 28, 28)

    X_train_noise = np.array([add_noise(img) for img in X_train[:110, 0]])
    X_train_noise = X_train_noise.reshape(-1, 1, 28, 28)

    # Formiranje proširenog trening skupa
    X_train_aug = np.concatenate(
        (
            X_train[:110],
            X_train_rot_pos,
            X_train_rot_neg,
            X_train_flip,
            X_train_noise
        ),
        axis=0
    )

    y_train_aug = np.concatenate(
        (
            y_train[:110],
            y_train[:110],
            y_train[:110],
            y_train[:110],
            y_train[:110]
        ),
        axis=0
    )

    print("Originalni skup:", X_train[:110].shape)
    print("Augmentirani skup:", X_train_aug.shape)

    return X_train_aug, y_train_aug