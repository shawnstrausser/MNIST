"""
data.py — Data loading and preprocessing.

Handles downloading the MNIST dataset (if needed) and wrapping it in
DataLoaders that feed batches of images to the training/evaluation loops.

Preprocessing steps:
  1. ToTensor()  — converts the 28x28 grayscale image to a PyTorch tensor
                   with values in [0, 1]
  2. Normalize() — shifts values so the mean is ~0 and std is ~1,
                   which helps the model learn faster and more stably.
                   (0.1307 and 0.3081 are the precomputed MNIST mean/std.)
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, BATCH_SIZE


def get_transforms():
    """Build the image preprocessing pipeline."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_data_loaders():
    """Return (train_loader, test_loader) ready to iterate over."""
    transform = get_transforms()

    train_dataset = datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader
