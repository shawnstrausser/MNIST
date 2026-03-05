from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_DIR, BATCH_SIZE


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])


def get_data_loaders():
    transform = get_transforms()

    train_dataset = datasets.MNIST(DATA_DIR, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader
