"""
Dataset utilities — FaceGenerationVAE

Provides CelebA data loading with the same augmentation pipeline
used in the original Kaggle training run.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path


def build_transforms(image_size: int, augment: bool = True) -> transforms.Compose:
    """
    Build the image transform pipeline.

    Training augmentations match the original notebook:
      - Resize + CenterCrop to square
      - RandomHorizontalFlip (p=0.5)
      - RandomRotation ±15°
      - ColorJitter (brightness, contrast, saturation 0.3)
    """
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    aug = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ]
    if augment:
        # Insert augmentations before ToTensor
        pipeline = base[:2] + aug + [base[-1]]
    else:
        pipeline = base
    return transforms.Compose(pipeline)


def build_dataloaders(
    data_root: str,
    image_size: int = 128,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    download: bool = True,
):
    """
    Build CelebA train/val DataLoaders.

    Downloads CelebA (~1.3 GB) to data_root on first run if download=True.

    Args:
        data_root:   Directory where CelebA data is stored / will be downloaded.
        image_size:  Spatial resolution for cropped output (default 128).
        batch_size:  Batch size for both loaders.
        val_split:   Fraction of training data held out for validation.
        num_workers: DataLoader worker processes.
        download:    Whether to auto-download CelebA (requires Google Drive access).

    Returns:
        (train_loader, val_loader, in_channels)
    """
    train_tf = build_transforms(image_size, augment=True)
    val_tf = build_transforms(image_size, augment=False)

    full_dataset = datasets.CelebA(
        root=data_root,
        split="train",
        download=download,
        transform=train_tf,
    )

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Val uses non-augmented transforms
    val_dataset.dataset = datasets.CelebA(
        root=data_root,
        split="train",
        download=False,
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Infer channels from first sample
    sample, _ = full_dataset[0]
    in_channels = sample.shape[0]

    return train_loader, val_loader, in_channels
