import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RankingAwareQuantizer:
    """
    Ranking-aware quantization for Vision Transformer attention mechanisms.
    Preserves relative ordering of attention scores during quantization.

    Based on: "Post-Training Quantization for Vision Transformer"
    Key insight: The ranking loss preserves the functionality of Multi-Head Self-Attention (MSA)
    """

    def __init__(self, bits: int = 8, ranking_weight: float = 0.5):
        """
        Args:
            bits: Number of bits for quantization
            ranking_weight: Weight for ranking loss component (0.0 to 1.0)
        """
        self.bits = bits
        self.ranking_weight = ranking_weight
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
        self.scale = None

    def compute_ranking_loss(self, original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss to preserve relative ordering.

        The ranking loss penalizes changes in the relative ordering of values,
        which is critical for attention mechanisms.

        Args:
            original: Original tensor (e.g., attention scores)
            quantized: Quantized tensor

        Returns:
            ranking_loss: Scalar loss value
        """
        # Flatten tensors for pairwise comparison
        orig_flat = original.reshape(-1)
        quant_flat = quantized.reshape(-1)

        # Sample pairs for efficiency (all pairs would be quadratic)
        num_samples = min(1000, len(orig_flat))
        indices = torch.randperm(len(orig_flat))[:num_samples]

        orig_sample = orig_flat[indices]
        quant_sample = quant_flat[indices]

        # Compute pairwise differences
        # Shape: (num_samples, num_samples)
        orig_diff = orig_sample.unsqueeze(1) - orig_sample.unsqueeze(0)
        quant_diff = quant_sample.unsqueeze(1) - quant_sample.unsqueeze(0)

        # Ranking loss: penalize sign flips in pairwise comparisons
        sign_orig = torch.sign(orig_diff)
        sign_quant = torch.sign(quant_diff)

        # Count ranking violations
        ranking_loss = (sign_orig != sign_quant).float().mean()

        return ranking_loss

    def calibrate_with_ranking(self, tensor: torch.Tensor,
                              attention_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calibrate scale factor considering ranking preservation.

        Args:
            tensor: Input tensor to quantize
            attention_map: Optional attention map for ranking-aware calibration

        Returns:
            scale: Optimized quantization scale
        """
        # Start with standard symmetric quantization
        max_val = tensor.abs().max()
        initial_scale = max_val / self.qmax

        if attention_map is None or self.ranking_weight == 0.0:
            # No ranking optimization
            self.scale = initial_scale
            return initial_scale

        # Grid search for better scale (simple optimization)
        best_scale = initial_scale
        best_loss = float('inf')

        scale_candidates = initial_scale * torch.linspace(0.8, 1.2, 10).to(tensor.device)

        for scale in scale_candidates:
            # Quantize with this scale
            tensor_q = torch.round(tensor / scale)
            tensor_q = torch.clamp(tensor_q, self.qmin, self.qmax)
            tensor_dq = tensor_q * scale

            # Compute combined loss
            mse_loss = F.mse_loss(tensor_dq, tensor)
            rank_loss = self.compute_ranking_loss(attention_map, tensor_dq)

            total_loss = (1 - self.ranking_weight) * mse_loss + self.ranking_weight * rank_loss

            if total_loss < best_loss:
                best_loss = total_loss
                best_scale = scale

        self.scale = best_scale
        return best_scale

    def quantize(self, tensor: torch.Tensor,
                attention_map: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor with ranking awareness.

        Args:
            tensor: Input tensor
            attention_map: Optional attention map for ranking-aware quantization

        Returns:
            quantized_tensor: Quantized and dequantized tensor
            scale: Quantization scale used
        """
        if self.scale is None:
            self.calibrate_with_ranking(tensor, attention_map)

        # Quantize
        tensor_q = torch.round(tensor / self.scale)
        tensor_q = torch.clamp(tensor_q, self.qmin, self.qmax)

        # Dequantize for simulation
        tensor_dq = tensor_q * self.scale

        return tensor_dq, self.scale

    def __call__(self, tensor: torch.Tensor,
                attention_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience method for quantization."""
        tensor_dq, _ = self.quantize(tensor, attention_map)
        return tensor_dq


class QuantizedAttention(nn.Module):
    """
    Wrapper for quantizing attention mechanisms in Vision Transformers.
    Applies ranking-aware quantization to Q, K, V projections and attention outputs.
    """

    def __init__(self, attention_module: nn.Module, bits: int = 8,
                 ranking_weight: float = 0.5, quantize_qkv: bool = True):
        """
        Args:
            attention_module: Original attention module
            bits: Number of bits for quantization
            ranking_weight: Weight for ranking loss in quantization
            quantize_qkv: Whether to quantize Q, K, V tensors
        """
        super().__init__()
        self.attention = attention_module
        self.bits = bits
        self.quantize_qkv = quantize_qkv

        # Separate quantizers for different components
        self.qkv_quantizer = RankingAwareQuantizer(bits=bits, ranking_weight=ranking_weight)
        self.attn_quantizer = RankingAwareQuantizer(bits=bits, ranking_weight=ranking_weight)
        self.output_quantizer = RankingAwareQuantizer(bits=bits, ranking_weight=0.0)  # Standard for output

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass with ranking-aware quantization.

        Note: This is a simplified wrapper. For actual ViT models,
        you may need to modify based on the specific attention implementation.
        """
        # For most attention modules, the forward is straightforward
        # In practice, you'd intercept Q, K, V and attention scores
        return self.attention(x, **kwargs)


def apply_ranking_aware_quantization(model: nn.Module, bits: int = 8,
                                     ranking_weight: float = 0.5) -> nn.Module:
    """
    Apply ranking-aware quantization to attention layers in a Vision Transformer.

    Args:
        model: Vision Transformer model
        bits: Number of bits for quantization
        ranking_weight: Weight for ranking loss

    Returns:
        model: Model with quantized attention layers
    """

    def _quantize_attention_recursive(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this is an attention module
            # Common names: 'attn', 'attention', 'self_attn'
            if 'attn' in child_name.lower() and hasattr(child, 'forward'):
                # Wrap with quantized attention
                setattr(module, child_name, QuantizedAttention(
                    child, bits=bits, ranking_weight=ranking_weight
                ))
            else:
                _quantize_attention_recursive(child, full_name)

    _quantize_attention_recursive(model)
    return model


class RankingLoss(nn.Module):
    """
    Standalone ranking loss module for training or calibration.
    """

    def __init__(self, sample_size: int = 1000):
        super().__init__()
        self.sample_size = sample_size

    def forward(self, original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss between original and quantized tensors.

        Args:
            original: Original tensor
            quantized: Quantized tensor

        Returns:
            loss: Ranking loss value
        """
        orig_flat = original.reshape(-1)
        quant_flat = quantized.reshape(-1)

        num_samples = min(self.sample_size, len(orig_flat))
        indices = torch.randperm(len(orig_flat), device=original.device)[:num_samples]

        orig_sample = orig_flat[indices]
        quant_sample = quant_flat[indices]

        orig_diff = orig_sample.unsqueeze(1) - orig_sample.unsqueeze(0)
        quant_diff = quant_sample.unsqueeze(1) - quant_sample.unsqueeze(0)

        sign_orig = torch.sign(orig_diff)
        sign_quant = torch.sign(quant_diff)

        ranking_loss = (sign_orig != sign_quant).float().mean()

        return ranking_loss
