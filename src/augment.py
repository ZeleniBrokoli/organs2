import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import shift
from scipy.ndimage import zoom


# Dodavanje Gaussian šuma
def add_noise(data):
    noise = np.random.normal(0.0, 0.05, data.shape).astype(np.float32)
    return data + noise


# Data augmentation
def augment_data(X_train, y_train):
    num_labeled = int(0.01 * len(X_train))
    indices = np.random.choice(len(X_train), num_labeled, replace=False)

    # Rotacija za +-15 stepeni, kad je bila rotacija od 90, bila je tacnost 37%
    X_train_rot_pos = np.array([
        rotate(img, angle=15, reshape=False, order=1, mode='nearest')
        # for img in X_train[:num_labeled, 0]
        for img in X_train[indices, 0]
    ])
    X_train_rot_pos = X_train_rot_pos.reshape(-1, 1, 28, 28)

    X_train_rot_neg = np.array([
        rotate(img, angle=-15, reshape=False, order=1, mode='nearest')
        for img in X_train[indices, 0]
    ])
    X_train_rot_neg = X_train_rot_neg.reshape(-1, 1, 28, 28)

    # Vertikalno preslikavanje (mozda nije pametno, jer se dobiju anatomski nemoguce stvari)
    X_train_flip = np.array([np.flipud(img) for img in X_train[indices, 0]])
    X_train_flip = X_train_flip.reshape(-1, 1, 28, 28)

    # translacija
    X_train_shift = np.array([
        shift(img, shift=(2, 1), mode='nearest')
        for img in X_train[indices, 0]
    ])

    X_train_shift = X_train_shift.reshape(-1, 1, 28, 28)

    # zoom
    X_train_zoom = np.array([
        zoom(img, zoom=1.1, order=1)
        for img in X_train[indices, 0]
    ])

    # vrati na 28x28
    X_train_zoom = np.array([
        np.resize(img, (28, 28))
        for img in X_train_zoom
    ])

    X_train_zoom = X_train_zoom.reshape(-1, 1, 28, 28)

    X_train_noise = np.array([add_noise(img) for img in X_train[indices, 0]])
    X_train_noise = X_train_noise.reshape(-1, 1, 28, 28)

    # Formiranje proširenog trening skupa
    X_train_aug = np.concatenate(
        (
            X_train[indices],
            X_train_rot_pos,
            X_train_rot_neg,
            X_train_shift,
            X_train_zoom,
            X_train_flip,
            X_train_noise
        ),
        axis=0
    )

    y_train_aug = np.concatenate(
        (
            y_train[indices],
            y_train[indices],
            y_train[indices],
            y_train[indices],
            y_train[indices],
            y_train[indices],
            y_train[indices]
        ),
        axis=0
    )

    print("Originalni skup:", X_train[indices].shape)
    print("Augmentirani skup:", X_train_aug.shape)

    return X_train_aug, y_train_aug