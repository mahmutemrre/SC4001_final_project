"""
Fashion-MNIST dataset loading and augmentation utilities.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_transforms(augment=True):
    """Return train and test transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ]) if augment else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    return train_transform, test_transform


def get_dataloaders(data_dir="./data", batch_size=64, num_workers=2, augment=True):
    """Load Fashion-MNIST and return train/test DataLoaders."""
    train_tf, test_tf = get_transforms(augment)

    train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_tf)

    pin = torch.cuda.is_available()  # pin_memory only helps on CUDA
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader
