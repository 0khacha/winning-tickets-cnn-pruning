"""
Dataset Loaders and Preprocessing Pipelines.
Supports CIFAR-10 (source dataset) and SVHN (primary target dataset for transfer),
with fixed deterministic validation splits and standard augmentations.
Also includes support for Fashion-MNIST (with channel expansion & padding) and CIFAR-100 subset.
"""

import os
from typing import Dict
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Standard normalization statistics
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "num_classes": 10
    },
    "fashion_mnist": {
        "mean": (0.2860, 0.2860, 0.2860),
        "std": (0.3530, 0.3530, 0.3530),
        "num_classes": 10
    }
}

def get_transforms(dataset_name: str = "cifar10", augment: bool = True):
    dataset_name = dataset_name.lower()
    stats = DATASET_STATS.get(dataset_name, DATASET_STATS["cifar10"])
    
    if dataset_name == "cifar10":
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"])
            ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"])
        ])
        return train_transform, eval_transform

    elif dataset_name == "svhn":
        # Note: Do NOT use horizontal flip on SVHN since digit orientation matters (e.g. 6 vs 9)
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(stats["mean"], stats["std"])
            ])
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"])
        ])
        return train_transform, eval_transform

    elif dataset_name == "fashion_mnist":
        # 1-channel 28x28 -> pad/resize to 32x32, duplicate to 3 channels for architectural compatibility
        base_ops = [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"])
        ]
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                *base_ops
            ])
        else:
            train_transform = transforms.Compose(base_ops)
        eval_transform = transforms.Compose(base_ops)
        return train_transform, eval_transform

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataloaders(
    dataset_name: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    val_split_ratio: float = 0.10,
    seed: int = 42,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """
    Builds deterministic train, validation, and test dataloaders.
    Validation set is sampled once using a fixed seed generator.
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()
    train_transform, eval_transform = get_transforms(dataset_name, augment=True)

    if dataset_name == "cifar10":
        # Download and instantiate datasets with appropriate transforms
        full_train_aug = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        full_train_eval = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=eval_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=eval_transform
        )
        total_train = len(full_train_aug)
        val_size = int(total_train * val_split_ratio)
        train_size = total_train - val_size

        # Deterministic split indices
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        train_subset = Subset(full_train_aug, train_indices)
        val_subset = Subset(full_train_eval, val_indices)

    elif dataset_name == "svhn":
        # SVHN has 'train' (~73k) and 'test' (~26k)
        full_train_aug = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=True, transform=train_transform
        )
        full_train_eval = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=False, transform=eval_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=eval_transform
        )
        total_train = len(full_train_aug)
        val_size = int(total_train * val_split_ratio)
        train_size = total_train - val_size

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        train_subset = Subset(full_train_aug, train_indices)
        val_subset = Subset(full_train_eval, val_indices)

    elif dataset_name == "fashion_mnist":
        full_train_aug = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        full_train_eval = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=False, transform=eval_transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=eval_transform
        )
        total_train = len(full_train_aug)
        val_size = int(total_train * val_split_ratio)
        train_size = total_train - val_size

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        train_subset = Subset(full_train_aug, train_indices)
        val_subset = Subset(full_train_eval, val_indices)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_dataset),
        "num_classes": DATASET_STATS[dataset_name]["num_classes"]
    }
