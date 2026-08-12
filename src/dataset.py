"""
=========================================================
Privacy-Preserving Federated Learning Framework

Module : Dataset
Version: 3.0 (Final Dissertation Release)

Author : Khawar Akram
=========================================================
"""
# ==========================================================
# DATASET UTILITIES
# ==========================================================

import os
import pickle

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from config import *

# ==========================================================
# IMAGE TRANSFORMS
# ==========================================================

def get_transforms():

    train_transform = transforms.Compose([

        transforms.RandomCrop(32, padding=4),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

    ])

    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

    ])

    return train_transform, test_transform


# ==========================================================
# LOAD DATASETS
# ==========================================================

def get_datasets():

    train_transform, test_transform = get_transforms()

    train_dataset = datasets.CIFAR10(
        root=DATASET_PATH,
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATASET_PATH,
        train=False,
        download=False,
        transform=test_transform
    )

    return train_dataset, test_dataset


# ==========================================================
# CREATE DATALOADERS
# ==========================================================

def get_dataloaders():

    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return train_loader, test_loader


# ==========================================================
# CLIENT PARTITIONS
# ==========================================================

CLIENT_INDEX_FILE = os.path.join(
    DATASET_PATH,
    "client_indices.pkl"
)


def save_client_indices(client_indices):

    with open(CLIENT_INDEX_FILE, "wb") as f:
        pickle.dump(client_indices, f)

    print("Client partitions saved.")


def load_client_indices():

    with open(CLIENT_INDEX_FILE, "rb") as f:
        client_indices = pickle.load(f)

    return client_indices


# ==========================================================
# CLIENT DATALOADER
# ==========================================================

def get_client_loader(client_id):

    client_indices = load_client_indices()

    train_dataset, _ = get_datasets()

    subset = Subset(
        train_dataset,
        client_indices[client_id]
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return loader
