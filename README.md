# OrganSMNIST Classification using Denoising Autoencoder and Transfer Learning

This project implements a semi-supervised image 
classification pipeline for the OrganSMNIST dataset.
A denoising convolutional autoencoder is first pretrained
to reconstruct noisy medical images. The learned encoder is
then transferred to a classifier, allowing classification
with only a small fraction of labeled data.

## Project Overview

The project consists of two main stages:

1. **Denoising Autoencoder Pretraining**
   - Gaussian noise is added to input images.
   - The autoencoder learns to reconstruct the original images.
   - The encoder learns meaningful feature representations without using labels.

2. **Classifier Training**
   - The pretrained encoder weights are transferred to a convolutional classifier.
   - Only **1% of the training images are labeled**.
   - Data augmentation is applied to increase the size and diversity of the labeled dataset.

---

## Dataset

The project uses the **OrganSMNIST** dataset from the MedMNIST collection.

Images are:
- grayscale
- size **28 × 28**
- 11 organ classes

The dataset is stored in:

```
data/organsmnist.npz
```

---

## Data Preparation

The preprocessing pipeline includes:

- loading OrganSMNIST data
- normalization
- one-hot encoding of labels
- reshaping images to

```
(1, 28, 28)
```

for convolutional neural networks.

---

## Data Augmentation

To improve classification performance with only 1% labeled data, several augmentation techniques are applied:

- Gaussian noise
- rotation (+15° and −15°)
- translation
- zoom with center crop

The augmented images are combined with the original labeled images before classifier training.

---

## Autoencoder

The implemented denoising autoencoder contains:

- convolutional encoder
- max-pooling
- convolutional bottleneck
- upsampling decoder
- transposed convolutions

The model is trained using **Mean Squared Error (MSE)** reconstruction loss.

---

## Classifier

The classifier reuses the encoder learned during autoencoder pretraining.

The transferred feature extractor is followed by fully connected layers that predict one of the 11 organ classes.

Training uses:

- CrossEntropyLoss
- Adam optimizer

---

## Training Pipeline

The complete workflow is:

1. Load dataset
2. Pretrain denoising autoencoder
3. Save pretrained weights
4. Select 1% labeled training samples
5. Apply data augmentation
6. Initialize classifier using pretrained encoder
7. Train classifier
8. Evaluate on the test set
9. Visualize reconstructed images

---

## Evaluation

The project reports:

- reconstruction loss of the autoencoder
- classification accuracy
- reconstructed image example
- training and validation curves

---

## Requirements

Main dependencies:

- Python 3.10+
- PyTorch
- NumPy
- SciPy
- Matplotlib

Install packages with

```bash
pip install torch torchvision numpy scipy matplotlib
```

---

## Running the Project

Run the complete pipeline with

```bash
python src/main.py
```