"""
Model Loading Utilities
=======================
Load pretrained ResNet-18 and DeiT-Tiny models for quantization experiments.

Uses the actual trained models from teammates:
- ResNet-18: Salma's CIFAR-100 trained model (Resnet_Cifar100_PTQ)
- DeiT-Tiny: Krupa's ImageNet pretrained model from timm (DeiT_RA_PTQ)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available. Install with: pip install timm")

from torchvision import models


# Path to Salma's trained ResNet-18 checkpoint
# Structure: ~/final/Resnet_Cifar100_PTQ/checkpoints/resnet18_cifar100_trained.pth
# This code runs from: ~/final/Mixed-precision-and-energy-modeling/
RESNET_CHECKPOINT_PATH = os.path.join(
    os.path.expanduser(
        "~"), "final", "Resnet_Cifar100_PTQ", "checkpoints", "resnet18_cifar100_trained.pth"
)


def load_resnet18(
    num_classes: int = 100,
    pretrained: bool = True,
    device: str = "cuda",
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """
    Load ResNet-18 model trained by Salma on CIFAR-100.

    Uses the exact same architecture as Salma's training script:
    - Standard ResNet-18 from torchvision
    - Only the final FC layer is changed to 100 classes
    - Loads weights from Resnet_Cifar100_PTQ/checkpoints/resnet18_cifar100_trained.pth

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        pretrained: Whether to load Salma's trained weights (default: True)
        device: Device to load model on
        checkpoint_path: Optional custom path to checkpoint

    Returns:
        ResNet-18 model with Salma's trained weights
    """
    # Use same architecture as Salma's training script
    # resnet18(weights=None, num_classes=100) - standard ResNet-18 with 100-class output
    model = models.resnet18(weights=None, num_classes=num_classes)

    # Load Salma's trained checkpoint
    if pretrained:
        ckpt_path = checkpoint_path or RESNET_CHECKPOINT_PATH

        if os.path.exists(ckpt_path):
            print(f"Loading Salma's trained ResNet-18 from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✓ Successfully loaded Salma's CIFAR-100 trained weights")
        else:
            print(f"⚠ Warning: Checkpoint not found at {ckpt_path}")
            print("  Using randomly initialized weights instead.")
            print(
                "  To fix: ensure Resnet_Cifar100_PTQ/checkpoints/resnet18_cifar100_trained.pth exists")

    model = model.to(device)
    model.eval()
    return model


def load_deit_tiny(
    pretrained: bool = True,
    device: str = "cuda"
) -> nn.Module:
    """
    Load DeiT-Tiny model from timm (same as Krupa's DeiT_RA_PTQ experiments).

    Uses the exact same model as Krupa's experiments:
    - DeiT-Tiny with patch size 16, input size 224
    - ImageNet-1K pretrained weights from timm
    - 1000 output classes for ImageNet classification

    Args:
        pretrained: Whether to load ImageNet pretrained weights (default: True)
        device: Device to load model on

    Returns:
        DeiT-Tiny model (same as Krupa's experiments)
    """
    if not TIMM_AVAILABLE:
        raise ImportError(
            "timm library is required. Install with: pip install timm")

    print("Loading DeiT-Tiny from timm (same model as Krupa's DeiT_RA_PTQ experiments)")
    model = timm.create_model(
        'deit_tiny_patch16_224',
        pretrained=pretrained,
        num_classes=1000
    )

    if pretrained:
        print("✓ Successfully loaded ImageNet-pretrained DeiT-Tiny weights from timm")

    model = model.to(device)
    model.eval()
    return model


def get_model_layers(model: nn.Module, model_type: str) -> Dict[str, nn.Module]:
    """
    Get quantizable layers from a model.

    Args:
        model: The neural network model
        model_type: Either 'resnet' or 'deit'

    Returns:
        Dictionary mapping layer names to layer modules
    """
    layers = {}

    if model_type == 'resnet':
        # Get all Conv2d and Linear layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                layers[name] = module

    elif model_type == 'deit':
        # Get attention projections and MLP layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers[name] = module

    return layers


def get_layer_info(model: nn.Module, model_type: str) -> List[Dict]:
    """
    Get information about each layer for energy modeling.

    Args:
        model: The neural network model
        model_type: Either 'resnet' or 'deit'

    Returns:
        List of dicts with layer info (name, type, params, MACs estimate)
    """
    layer_info = []

    for name, module in model.named_modules():
        info = {'name': name, 'module': module}

        if isinstance(module, nn.Conv2d):
            # MACs = K² × Cin × Cout × H × W
            info['type'] = 'conv'
            info['kernel_size'] = module.kernel_size
            info['in_channels'] = module.in_channels
            info['out_channels'] = module.out_channels
            info['params'] = sum(p.numel() for p in module.parameters())

        elif isinstance(module, nn.Linear):
            # MACs = in_features × out_features
            info['type'] = 'linear'
            info['in_features'] = module.in_features
            info['out_features'] = module.out_features
            info['macs'] = module.in_features * module.out_features
            info['params'] = sum(p.numel() for p in module.parameters())

        else:
            continue

        layer_info.append(info)

    return layer_info


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_type: str):
    """Print a summary of the model architecture."""
    print(f"\n{'='*60}")
    print(f"Model Summary: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Total Parameters: {count_parameters(model):,}")

    layers = get_model_layers(model, model_type)
    print(f"Quantizable Layers: {len(layers)}")

    print(f"\nLayer breakdown:")
    for name, module in layers.items():
        if isinstance(module, nn.Conv2d):
            print(
                f"  {name}: Conv2d({module.in_channels}, {module.out_channels}, k={module.kernel_size})")
        elif isinstance(module, nn.Linear):
            print(f"  {name}: Linear({module.in_features}, {module.out_features})")
    print(f"{'='*60}\n")
