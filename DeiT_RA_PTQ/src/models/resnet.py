import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


def get_resnet18_cifar100(pretrained: bool = False, num_classes: int = 100) -> nn.Module:
    """
    Get ResNet-18 model adapted for CIFAR-100.

    Standard ResNet-18 is designed for ImageNet (224x224), but CIFAR-100
    uses 32x32 images. We adapt the first layer accordingly.

    Args:
        pretrained: Whether to use pretrained weights (on ImageNet)
        num_classes: Number of output classes

    Returns:
        model: ResNet-18 model
    """
    # Get standard ResNet-18
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    # Adapt for CIFAR-100 (32x32 images)
    # Replace the first 7x7 conv with a 3x3 conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove the first max pooling layer by making it identity
    model.maxpool = nn.Identity()

    # Adjust the final fully connected layer for CIFAR-100 (100 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def train_resnet18_cifar100(model: nn.Module, train_loader, test_loader,
                           epochs: int = 100, learning_rate: float = 0.1,
                           device: str = 'cuda', weight_decay: float = 5e-4) -> nn.Module:
    """
    Train ResNet-18 on CIFAR-100.

    Args:
        model: ResNet-18 model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to train on
        weight_decay: Weight decay for optimizer

    Returns:
        model: Trained model
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                               momentum=0.9, weight_decay=weight_decay)

    # Learning rate schedule: decay at epochs 50, 75, 90
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50, 75, 90],
                                                     gamma=0.1)

    best_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total

        # Validation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total

        scheduler.step()

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.3f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc

    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")
    return model


def save_resnet18_checkpoint(model: nn.Module, path: str, epoch: int,
                            accuracy: float, optimizer: Optional[torch.optim.Optimizer] = None):
    """
    Save ResNet-18 checkpoint.

    Args:
        model: Model to save
        path: Save path
        epoch: Current epoch
        accuracy: Current accuracy
        optimizer: Optional optimizer state
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_resnet18_checkpoint(model: nn.Module, path: str,
                            optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[nn.Module, dict]:
    """
    Load ResNet-18 checkpoint.

    Args:
        model: Model architecture
        path: Checkpoint path
        optimizer: Optional optimizer to load state into

    Returns:
        model: Model with loaded weights
        checkpoint_info: Dictionary with checkpoint information
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 0),
        'accuracy': checkpoint.get('accuracy', 0.0),
    }

    print(f"Checkpoint loaded from {path} (Epoch {checkpoint_info['epoch']}, Acc: {checkpoint_info['accuracy']:.2f}%)")

    return model, checkpoint_info
