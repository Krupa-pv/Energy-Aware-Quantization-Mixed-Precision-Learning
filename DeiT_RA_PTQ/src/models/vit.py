import torch
import torch.nn as nn
from typing import Tuple
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available. Install with: pip install timm")


def get_deit_tiny(pretrained: bool = True, num_classes: int = 1000) -> nn.Module:
    """
    Get DeiT-Tiny model from timm.

    DeiT (Data-efficient Image Transformer) is a vision transformer
    trained on ImageNet.

    Args:
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes

    Returns:
        model: DeiT-Tiny model
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required. Install with: pip install timm")

    model = timm.create_model('deit_tiny_patch16_224',
                             pretrained=pretrained,
                             num_classes=num_classes)

    return model


def get_vit_small(pretrained: bool = True, num_classes: int = 1000) -> nn.Module:
    """
    Get ViT-Small model from timm.

    Args:
        pretrained: Whether to load pretrained weights
        num_classes: Number of output classes

    Returns:
        model: ViT-Small model
    """
    if not TIMM_AVAILABLE:
        raise ImportError("timm library is required. Install with: pip install timm")

    model = timm.create_model('vit_small_patch16_224',
                             pretrained=pretrained,
                             num_classes=num_classes)

    return model


def analyze_vit_structure(model: nn.Module):
    """
    Analyze and print the structure of a Vision Transformer model.

    Args:
        model: ViT model to analyze
    """
    print(f"\n{'='*60}")
    print("Vision Transformer Structure Analysis")
    print(f"{'='*60}")

    # Count different layer types
    linear_layers = []
    attention_layers = []
    norm_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
        elif 'attn' in name.lower():
            attention_layers.append(name)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            norm_layers.append(name)

    print(f"\nLinear Layers: {len(linear_layers)}")
    if len(linear_layers) <= 20:
        for layer in linear_layers:
            print(f"  - {layer}")

    print(f"\nAttention Modules: {len(attention_layers)}")
    if len(attention_layers) <= 20:
        for layer in attention_layers:
            print(f"  - {layer}")

    print(f"\nNormalization Layers: {len(norm_layers)}")

    # Get model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    print(f"\n{'='*60}\n")


def get_vit_attention_modules(model: nn.Module) -> dict:
    """
    Extract attention modules from a ViT model.

    Args:
        model: ViT model

    Returns:
        attention_modules: Dictionary of {name: module} for attention layers
    """
    attention_modules = {}

    for name, module in model.named_modules():
        if 'attn' in name.lower() and hasattr(module, 'forward'):
            attention_modules[name] = module

    return attention_modules


def quantize_vit_attention(model: nn.Module, bits: int = 8,
                          ranking_weight: float = 0.5) -> nn.Module:
    """
    Apply quantization specifically to attention modules in a ViT.

    Args:
        model: Vision Transformer model
        bits: Number of bits for quantization
        ranking_weight: Weight for ranking-aware quantization

    Returns:
        model: Model with quantized attention layers
    """
    from ..quantization.ranking_aware import apply_ranking_aware_quantization

    model = apply_ranking_aware_quantization(model, bits=bits, ranking_weight=ranking_weight)

    return model


def save_vit_checkpoint(model: nn.Module, path: str, accuracy: float):
    """
    Save ViT checkpoint.

    Args:
        model: Model to save
        path: Save path
        accuracy: Model accuracy
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
    }

    torch.save(checkpoint, path)
    print(f"ViT checkpoint saved to {path}")


def load_vit_checkpoint(model: nn.Module, path: str) -> Tuple[nn.Module, float]:
    """
    Load ViT checkpoint.

    Args:
        model: Model architecture
        path: Checkpoint path

    Returns:
        model: Model with loaded weights
        accuracy: Checkpoint accuracy
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    accuracy = checkpoint.get('accuracy', 0.0)

    print(f"ViT checkpoint loaded from {path} (Accuracy: {accuracy:.2f}%)")

    return model, accuracy
