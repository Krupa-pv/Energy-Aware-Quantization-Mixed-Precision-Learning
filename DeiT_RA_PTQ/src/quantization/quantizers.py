import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SymmetricUniformQuantizer:
    """
    Symmetric uniform quantization for weights and activations.
    Supports both per-tensor and per-channel quantization.
    """

    def __init__(self, bits: int = 8, mode: str = 'per_tensor', channel_dim: int = 0):
        """
        Args:
            bits: Number of bits for quantization (4, 6, or 8)
            mode: 'per_tensor' or 'per_channel'
            channel_dim: Channel dimension for per-channel quantization (0 for weights)
        """
        self.bits = bits
        self.mode = mode
        self.channel_dim = channel_dim
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.scale = None

    def calibrate(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute quantization scale factor based on tensor statistics.

        Args:
            tensor: Input tensor to calibrate on

        Returns:
            scale: Quantization scale factor
        """
        if self.mode == 'per_tensor':
            max_val = tensor.abs().max()
            scale = max_val / self.qmax
        elif self.mode == 'per_channel':
            # Compute scale per channel
            # Reshape to bring channel_dim to front
            shape = list(range(len(tensor.shape)))
            shape[0], shape[self.channel_dim] = shape[self.channel_dim], shape[0]
            tensor_permuted = tensor.permute(*shape)

            # Flatten all dimensions except channel
            num_channels = tensor_permuted.shape[0]
            tensor_flat = tensor_permuted.reshape(num_channels, -1)

            # Compute max per channel
            max_vals = tensor_flat.abs().max(dim=1)[0]
            scale = max_vals / self.qmax

            # Reshape scale to broadcast correctly
            scale_shape = [1] * len(tensor.shape)
            scale_shape[self.channel_dim] = num_channels
            scale = scale.reshape(scale_shape)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Avoid division by zero
        scale = torch.clamp(scale, min=1e-8)
        self.scale = scale
        return scale

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor using symmetric uniform quantization.

        Args:
            tensor: Input tensor

        Returns:
            quantized_tensor: Quantized and dequantized tensor
            scale: Quantization scale used
        """
        if self.scale is None:
            self.calibrate(tensor)

        # Quantize
        tensor_q = torch.round(tensor / self.scale)
        tensor_q = torch.clamp(tensor_q, self.qmin, self.qmax)

        # Dequantize for simulation
        tensor_dq = tensor_q * self.scale

        return tensor_dq, self.scale

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convenience method for quantization."""
        tensor_dq, _ = self.quantize(tensor)
        return tensor_dq


def quantize_tensor(tensor: torch.Tensor, bits: int = 8, mode: str = 'per_tensor',
                   channel_dim: int = 0) -> Tuple[torch.Tensor, float]:
    """
    Convenience function to quantize a tensor.

    Args:
        tensor: Input tensor
        bits: Number of bits for quantization
        mode: 'per_tensor' or 'per_channel'
        channel_dim: Channel dimension for per-channel quantization

    Returns:
        quantized_tensor: Quantized and dequantized tensor
        scale: Quantization scale(s) used
    """
    quantizer = SymmetricUniformQuantizer(bits=bits, mode=mode, channel_dim=channel_dim)
    return quantizer.quantize(tensor)


class QuantizedModule(nn.Module):
    """
    Wrapper to quantize a module's weights and activations during forward pass.
    """

    def __init__(self, module: nn.Module, weight_bits: int = 8,
                 activation_bits: int = 8, weight_mode: str = 'per_channel'):
        """
        Args:
            module: Original module to quantize
            weight_bits: Bits for weight quantization
            activation_bits: Bits for activation quantization
            weight_mode: 'per_tensor' or 'per_channel' for weights
        """
        super().__init__()
        self.module = module
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_mode = weight_mode

        # Quantize weights once during initialization
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    channel_dim = 0  # Output channel dimension
                    quantizer = SymmetricUniformQuantizer(
                        bits=weight_bits,
                        mode=weight_mode,
                        channel_dim=channel_dim
                    )
                    module.weight.data = quantizer(module.weight.data)

        # Activation quantizer
        self.act_quantizer = SymmetricUniformQuantizer(bits=activation_bits, mode='per_tensor')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized activations."""
        # Quantize input activations
        x_q = self.act_quantizer(x)

        # Forward through module (weights already quantized)
        out = self.module(x_q)

        return out


def quantize_model(model: nn.Module, weight_bits: int = 8, activation_bits: int = 8,
                  weight_mode: str = 'per_channel', skip_layers: list = None) -> nn.Module:
    """
    Apply post-training quantization to a model.

    Args:
        model: Model to quantize
        weight_bits: Bits for weight quantization
        activation_bits: Bits for activation quantization
        weight_mode: 'per_tensor' or 'per_channel' for weights
        skip_layers: List of layer types to skip quantization

    Returns:
        quantized_model: Model with quantized weights and activations
    """
    if skip_layers is None:
        skip_layers = [nn.BatchNorm2d, nn.LayerNorm, nn.Dropout]

    def _quantize_recursive(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Skip certain layers
            if any(isinstance(child, skip_type) for skip_type in skip_layers):
                continue

            # Quantize linear and conv layers
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                setattr(module, child_name, QuantizedModule(
                    child, weight_bits, activation_bits, weight_mode
                ))
            else:
                # Recursively quantize
                _quantize_recursive(child, full_name)

    _quantize_recursive(model)
    return model
