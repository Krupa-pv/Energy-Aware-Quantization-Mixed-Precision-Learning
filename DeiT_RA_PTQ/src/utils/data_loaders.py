import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import os


def get_cifar100_loaders(batch_size: int = 128, num_workers: int = 4,
                         data_dir: str = './data',
                         calibration_samples: int = 1000) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders for training, calibration, and testing.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        data_dir: Directory to store/load CIFAR-100 data
        calibration_samples: Number of samples for calibration set

    Returns:
        train_loader: Training data loader
        calibration_loader: Calibration data loader (subset of training)
        test_loader: Test data loader
    """
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Only normalization for test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761])
    ])

    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Create calibration subset
    calibration_indices = torch.randperm(len(train_dataset))[:calibration_samples].tolist()
    calibration_dataset = Subset(train_dataset, calibration_indices)

    # For calibration, use test_transform (no augmentation)
    calibration_dataset_no_aug = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=test_transform
    )
    calibration_dataset = Subset(calibration_dataset_no_aug, calibration_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    calibration_loader = DataLoader(
        calibration_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, calibration_loader, test_loader


def get_imagenet_loaders(batch_size: int = 128, num_workers: int = 4,
                        data_dir: str = './data/imagenet',
                        calibration_samples: int = 1000,
                        val_only: bool = True) -> Tuple[Optional[DataLoader], DataLoader, DataLoader]:
    """
    Get ImageNet data loaders.

    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        data_dir: Directory containing ImageNet data
        calibration_samples: Number of samples for calibration
        val_only: If True, only return validation loaders (for PTQ)

    Returns:
        train_loader: Training data loader (None if val_only=True)
        calibration_loader: Calibration data loader
        val_loader: Validation data loader
    """
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Training transform (if needed)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = None
    if not val_only and os.path.exists(os.path.join(data_dir, 'train')):
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=train_transform
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )

    # Validation dataset
    val_dataset = None
    if os.path.exists(os.path.join(data_dir, 'val')):
        val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=val_transform
        )
    elif os.path.exists(os.path.join(data_dir, 'validation')):
        val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'validation'),
            transform=val_transform
        )

    if val_dataset is None:
        print(f"Warning: ImageNet validation set not found at {data_dir}")
        print("Creating dummy loaders for demonstration purposes.")
        # Create a dummy dataset for testing
        from torch.utils.data import TensorDataset
        dummy_images = torch.randn(calibration_samples, 3, 224, 224)
        dummy_labels = torch.randint(0, 1000, (calibration_samples,))
        val_dataset = TensorDataset(dummy_images, dummy_labels)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Create calibration subset from validation
    calibration_indices = torch.randperm(len(val_dataset))[:calibration_samples].tolist()
    calibration_dataset = Subset(val_dataset, calibration_indices)

    calibration_loader = DataLoader(
        calibration_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, calibration_loader, val_loader


def get_calibration_loader(dataset_name: str = 'cifar100', batch_size: int = 128,
                          num_samples: int = 1000, **kwargs) -> DataLoader:
    """
    Convenience function to get only a calibration loader.

    Args:
        dataset_name: 'cifar100' or 'imagenet'
        batch_size: Batch size
        num_samples: Number of calibration samples
        **kwargs: Additional arguments for dataset-specific loaders

    Returns:
        calibration_loader: Calibration data loader
    """
    if dataset_name.lower() == 'cifar100':
        _, calibration_loader, _ = get_cifar100_loaders(
            batch_size=batch_size,
            calibration_samples=num_samples,
            **kwargs
        )
    elif dataset_name.lower() == 'imagenet':
        _, calibration_loader, _ = get_imagenet_loaders(
            batch_size=batch_size,
            calibration_samples=num_samples,
            val_only=True,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return calibration_loader
