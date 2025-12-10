"""
Layer Sensitivity Measurement
=============================
Compute sensitivity of each layer to quantization for mixed-precision assignment.

Sensitivity is measured as: S_l = ||y_l^full - y_l^quant||_2

Higher sensitivity = layer is more affected by quantization = needs more bits
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

from .quantize import fake_quantize


@dataclass
class LayerSensitivity:
    """Stores sensitivity information for a single layer"""
    name: str
    sensitivity: float
    sensitivity_normalized: float  # Normalized to [0, 1]
    output_shape: Tuple[int, ...]
    num_params: int


class SensitivityAnalyzer:
    """
    Analyzes layer sensitivity to quantization.
    
    Uses a calibration dataset to measure how much each layer's
    output changes when quantized at different bit widths.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        reference_bits: int = 4  # Compare against aggressive quantization
    ):
        self.model = model
        self.device = device
        self.reference_bits = reference_bits
        
        # Storage for activations
        self.full_precision_outputs: Dict[str, torch.Tensor] = {}
        self.quantized_outputs: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
    def _register_hooks(self, quantize: bool = False):
        """Register forward hooks to capture layer outputs."""
        self.hooks = []
        storage = self.quantized_outputs if quantize else self.full_precision_outputs
        storage.clear()
        
        def make_hook(name, quantize_output):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    if quantize_output:
                        output_stored = fake_quantize(output.detach(), self.reference_bits)
                    else:
                        output_stored = output.detach()
                    
                    if name in storage:
                        # Accumulate outputs across batches
                        storage[name] = torch.cat([storage[name], output_stored.cpu()], dim=0)
                    else:
                        storage[name] = output_stored.cpu()
            return hook
        
        # Register hooks for quantizable layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = module.register_forward_hook(make_hook(name, quantize))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    @torch.no_grad()
    def compute_sensitivities(
        self,
        calibration_loader: torch.utils.data.DataLoader,
        num_batches: Optional[int] = None
    ) -> Dict[str, LayerSensitivity]:
        """
        Compute sensitivity for all layers using calibration data.
        
        Args:
            calibration_loader: DataLoader with calibration samples
            num_batches: Number of batches to use (None = all)
            
        Returns:
            Dictionary mapping layer names to LayerSensitivity objects
        """
        self.model.eval()
        
        # Pass 1: Collect full precision outputs
        print("Pass 1: Collecting full-precision outputs...")
        self._register_hooks(quantize=False)
        
        for i, (images, _) in enumerate(tqdm(calibration_loader, desc="Full precision")):
            if num_batches and i >= num_batches:
                break
            images = images.to(self.device)
            _ = self.model(images)
        
        self._remove_hooks()
        
        # Pass 2: Collect quantized outputs
        print(f"Pass 2: Collecting {self.reference_bits}-bit quantized outputs...")
        self._register_hooks(quantize=True)
        
        for i, (images, _) in enumerate(tqdm(calibration_loader, desc="Quantized")):
            if num_batches and i >= num_batches:
                break
            images = images.to(self.device)
            # Quantize input as well
            images_q = fake_quantize(images, self.reference_bits)
            _ = self.model(images_q)
        
        self._remove_hooks()
        
        # Compute sensitivities
        print("Computing layer sensitivities...")
        sensitivities = {}
        
        for name in self.full_precision_outputs:
            if name not in self.quantized_outputs:
                continue
                
            full_out = self.full_precision_outputs[name]
            quant_out = self.quantized_outputs[name]
            
            # Ensure same size (might differ due to batching edge cases)
            min_size = min(full_out.shape[0], quant_out.shape[0])
            full_out = full_out[:min_size]
            quant_out = quant_out[:min_size]
            
            # Compute L2 sensitivity: ||y_full - y_quant||_2
            diff = full_out - quant_out
            sensitivity = torch.norm(diff, p=2).item()
            
            # Also compute relative sensitivity (normalized by output magnitude)
            output_norm = torch.norm(full_out, p=2).item()
            sensitivity_relative = sensitivity / (output_norm + 1e-8)
            
            # Get layer info
            module = dict(self.model.named_modules())[name]
            num_params = sum(p.numel() for p in module.parameters())
            
            sensitivities[name] = LayerSensitivity(
                name=name,
                sensitivity=sensitivity,
                sensitivity_normalized=sensitivity_relative,
                output_shape=tuple(full_out.shape[1:]),
                num_params=num_params
            )
        
        # Normalize sensitivities to [0, 1] range
        if sensitivities:
            max_sens = max(s.sensitivity for s in sensitivities.values())
            min_sens = min(s.sensitivity for s in sensitivities.values())
            range_sens = max_sens - min_sens + 1e-8
            
            for name, sens in sensitivities.items():
                sens.sensitivity_normalized = (sens.sensitivity - min_sens) / range_sens
        
        # Clear stored outputs to free memory
        self.full_precision_outputs.clear()
        self.quantized_outputs.clear()
        
        return sensitivities


def compute_layer_sensitivity(
    model: nn.Module,
    layer_name: str,
    calibration_data: torch.Tensor,
    bits: int = 4,
    device: str = "cuda"
) -> float:
    """
    Compute sensitivity for a single layer.
    
    Args:
        model: The neural network
        layer_name: Name of the layer to analyze
        calibration_data: Input tensor for calibration
        bits: Bit width for quantization comparison
        device: Device to run on
        
    Returns:
        Sensitivity value (L2 norm of output difference)
    """
    model.eval()
    full_output = None
    quant_output = None
    
    def get_full_hook(module, input, output):
        nonlocal full_output
        full_output = output.detach().cpu()
    
    def get_quant_hook(module, input, output):
        nonlocal quant_output
        quant_output = fake_quantize(output.detach(), bits).cpu()
    
    # Get the layer
    layer = dict(model.named_modules())[layer_name]
    
    # Full precision pass
    hook = layer.register_forward_hook(get_full_hook)
    with torch.no_grad():
        _ = model(calibration_data.to(device))
    hook.remove()
    
    # Quantized pass
    hook = layer.register_forward_hook(get_quant_hook)
    with torch.no_grad():
        _ = model(fake_quantize(calibration_data, bits).to(device))
    hook.remove()
    
    # Compute sensitivity
    sensitivity = torch.norm(full_output - quant_output, p=2).item()
    return sensitivity


def compute_all_sensitivities(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    device: str = "cuda",
    reference_bits: int = 4,
    num_batches: int = 8
) -> Dict[str, LayerSensitivity]:
    """
    Convenience function to compute all layer sensitivities.
    
    Args:
        model: Neural network model
        calibration_loader: DataLoader with calibration data
        device: Device to use
        reference_bits: Bit width for sensitivity measurement
        num_batches: Number of calibration batches
        
    Returns:
        Dictionary of layer sensitivities
    """
    analyzer = SensitivityAnalyzer(model, device, reference_bits)
    return analyzer.compute_sensitivities(calibration_loader, num_batches)


def print_sensitivity_report(sensitivities: Dict[str, LayerSensitivity]):
    """Print a formatted report of layer sensitivities."""
    print("\n" + "=" * 70)
    print("Layer Sensitivity Report")
    print("=" * 70)
    print(f"{'Layer Name':<40} {'Sensitivity':>12} {'Normalized':>12}")
    print("-" * 70)
    
    # Sort by sensitivity (descending)
    sorted_sens = sorted(
        sensitivities.values(),
        key=lambda x: x.sensitivity,
        reverse=True
    )
    
    for sens in sorted_sens:
        name = sens.name if len(sens.name) <= 38 else "..." + sens.name[-35:]
        print(f"{name:<40} {sens.sensitivity:>12.4f} {sens.sensitivity_normalized:>12.4f}")
    
    print("=" * 70 + "\n")


