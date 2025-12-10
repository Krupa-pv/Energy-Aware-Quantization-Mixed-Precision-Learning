"""
Quantization Utilities
======================
Implements symmetric uniform quantization for measuring sensitivity
and applying mixed-precision quantization.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuantizationParams:
    """Parameters for quantizing a tensor"""
    scale: float
    zero_point: int
    bits: int
    qmin: int
    qmax: int


def compute_quantization_params(
    tensor: torch.Tensor,
    bits: int,
    symmetric: bool = True
) -> QuantizationParams:
    """
    Compute quantization parameters for a tensor.
    
    Args:
        tensor: Input tensor to quantize
        bits: Number of bits for quantization
        symmetric: Whether to use symmetric quantization
        
    Returns:
        QuantizationParams with scale, zero_point, etc.
    """
    if symmetric:
        # Symmetric quantization: range is [-max_abs, max_abs]
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        max_abs = tensor.abs().max().item()
        
        # Avoid division by zero
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / qmax
            
        zero_point = 0
    else:
        # Asymmetric quantization
        qmin = 0
        qmax = 2 ** bits - 1
        
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        
        scale = (t_max - t_min) / (qmax - qmin) if t_max != t_min else 1.0
        zero_point = int(round(qmin - t_min / scale))
    
    return QuantizationParams(
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        qmin=qmin,
        qmax=qmax
    )


def quantize_tensor(
    tensor: torch.Tensor,
    bits: int,
    symmetric: bool = True,
    per_channel: bool = False,
    channel_dim: int = 0
) -> Tuple[torch.Tensor, QuantizationParams]:
    """
    Quantize a tensor to specified bit width.
    
    Args:
        tensor: Input tensor
        bits: Target bit width (e.g., 4, 6, 8)
        symmetric: Use symmetric quantization
        per_channel: Quantize per channel (for weights)
        channel_dim: Dimension for per-channel quantization
        
    Returns:
        Tuple of (quantized_tensor, quantization_params)
    """
    if per_channel:
        # Per-channel quantization (better for weights)
        num_channels = tensor.shape[channel_dim]
        quantized = torch.zeros_like(tensor)
        
        for c in range(num_channels):
            idx = [slice(None)] * tensor.dim()
            idx[channel_dim] = c
            channel_tensor = tensor[tuple(idx)]
            
            params = compute_quantization_params(channel_tensor, bits, symmetric)
            q = torch.clamp(
                torch.round(channel_tensor / params.scale + params.zero_point),
                params.qmin,
                params.qmax
            )
            quantized[tuple(idx)] = (q - params.zero_point) * params.scale
            
        # Return params for first channel (simplified)
        params = compute_quantization_params(tensor, bits, symmetric)
        return quantized, params
    else:
        # Per-tensor quantization
        params = compute_quantization_params(tensor, bits, symmetric)
        
        # Quantize
        q = torch.clamp(
            torch.round(tensor / params.scale + params.zero_point),
            params.qmin,
            params.qmax
        )
        
        # Dequantize (fake quantization)
        dequantized = (q - params.zero_point) * params.scale
        
        return dequantized, params


def fake_quantize(
    tensor: torch.Tensor,
    bits: int,
    symmetric: bool = True
) -> torch.Tensor:
    """
    Fake quantize a tensor (quantize then dequantize).
    
    This simulates the effect of quantization while keeping
    the tensor in floating point for gradient computation.
    
    Args:
        tensor: Input tensor
        bits: Target bit width
        symmetric: Use symmetric quantization
        
    Returns:
        Fake-quantized tensor (same dtype as input)
    """
    quantized, _ = quantize_tensor(tensor, bits, symmetric)
    return quantized


class QuantizedLayer(nn.Module):
    """
    Wrapper that applies fake quantization to a layer's weights and activations.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        weight_bits: int = 8,
        activation_bits: int = 8,
        symmetric: bool = True,
        per_channel_weights: bool = True
    ):
        super().__init__()
        self.layer = layer
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.symmetric = symmetric
        self.per_channel_weights = per_channel_weights
        
        # Pre-quantize weights
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Quantize the layer's weights in-place."""
        if hasattr(self.layer, 'weight') and self.layer.weight is not None:
            with torch.no_grad():
                q_weight, _ = quantize_tensor(
                    self.layer.weight.data,
                    self.weight_bits,
                    self.symmetric,
                    per_channel=self.per_channel_weights,
                    channel_dim=0
                )
                self.layer.weight.data = q_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activations
        x_q = fake_quantize(x, self.activation_bits, self.symmetric)
        
        # Forward through layer (weights already quantized)
        out = self.layer(x_q)
        
        return out


def get_quantization_error(
    tensor: torch.Tensor,
    bits: int
) -> float:
    """
    Compute the quantization error (MSE) for a tensor.
    
    Args:
        tensor: Original tensor
        bits: Quantization bit width
        
    Returns:
        Mean squared error between original and quantized
    """
    quantized = fake_quantize(tensor, bits)
    mse = ((tensor - quantized) ** 2).mean().item()
    return mse


# Convenience functions for common bit widths
def quantize_8bit(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to 8 bits."""
    return fake_quantize(tensor, 8)


def quantize_6bit(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to 6 bits."""
    return fake_quantize(tensor, 6)


def quantize_4bit(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to 4 bits."""
    return fake_quantize(tensor, 4)


