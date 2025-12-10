"""
Data Loading Utilities
======================
Load CIFAR-100 and ImageNet data for experiments.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Optional, Tuple
import os


def get_cifar100_loader(
    data_root: str = "./data",
    batch_size: int = 64,
    train: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """
    Get CIFAR-100 dataloader.
    
    Args:
        data_root: Root directory for data
        batch_size: Batch size
        train: If True, load training set; else validation set
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader for CIFAR-100
    """
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    
    dataset = datasets.CIFAR100(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_imagenet_calibration_loader(
    data_root: str = "./data/imagenet",
    batch_size: int = 32,
    num_samples: int = 1000,
    num_workers: int = 4
) -> DataLoader:
    """
    Get ImageNet calibration dataloader.
    
    Note: Requires ImageNet validation set to be downloaded.
    For calibration, we only need a small subset.
    
    Args:
        data_root: Root directory containing 'val' folder
        batch_size: Batch size
        num_samples: Number of samples for calibration
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader for ImageNet calibration
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_dir = os.path.join(data_root, 'val')
    
    if not os.path.exists(val_dir):
        print(f"Warning: ImageNet validation directory not found at {val_dir}")
        print("Using synthetic data for calibration instead.")
        return get_synthetic_imagenet_loader(batch_size, num_samples)
    
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    
    # Take subset for calibration
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    subset = Subset(dataset, indices)
    
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_synthetic_imagenet_loader(
    batch_size: int = 32,
    num_samples: int = 1000
) -> DataLoader:
    """
    Get synthetic ImageNet-like data for calibration when real data unavailable.
    
    Args:
        batch_size: Batch size
        num_samples: Number of synthetic samples
        
    Returns:
        DataLoader with synthetic data
    """
    class SyntheticImageNet(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Random image with ImageNet stats
            image = torch.randn(3, 224, 224)
            # Apply normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            label = torch.randint(0, 1000, (1,)).item()
            return image, label
    
    dataset = SyntheticImageNet(num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_calibration_subset(
    dataloader: DataLoader,
    num_samples: int = 256
) -> DataLoader:
    """
    Get a small subset of data for calibration.
    
    Args:
        dataloader: Original dataloader
        num_samples: Number of samples to extract
        
    Returns:
        DataLoader with subset
    """
    all_images = []
    all_labels = []
    
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(img.size(0) for img in all_images) >= num_samples:
            break
    
    images = torch.cat(all_images, dim=0)[:num_samples]
    labels = torch.cat(all_labels, dim=0)[:num_samples]
    
    dataset = torch.utils.data.TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=dataloader.batch_size, shuffle=False)


def evaluate_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    top_k: Tuple[int, ...] = (1, 5)
) -> dict:
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Neural network model
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation
        top_k: Tuple of k values for top-k accuracy
        
    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    
    correct = {k: 0 for k in top_k}
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            for k in top_k:
                _, pred = outputs.topk(k, 1, True, True)
                correct_k = pred.eq(labels.view(-1, 1).expand_as(pred)).any(dim=1).sum().item()
                correct[k] += correct_k
            
            total += labels.size(0)
    
    accuracies = {f'top{k}': 100.0 * correct[k] / total for k in top_k}
    accuracies['total_samples'] = total
    
    return accuracies


