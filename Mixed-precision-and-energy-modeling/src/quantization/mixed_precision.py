"""
Mixed-Precision Quantization
============================
Assign different bit widths to different layers based on sensitivity.

High sensitivity → 8 bits (preserve accuracy)
Moderate sensitivity → 6 bits (balance)
Low sensitivity → 4 bits (maximize compression)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from .sensitivity import LayerSensitivity
from .quantize import fake_quantize, QuantizedLayer


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed-precision assignment"""
    
    # Available bit widths
    bit_options: List[int] = field(default_factory=lambda: [4, 6, 8])
    
    # Threshold percentiles for assignment
    # Layers with sensitivity in top X% get 8 bits
    high_threshold_percentile: float = 75.0
    # Layers with sensitivity in bottom X% get 4 bits  
    low_threshold_percentile: float = 25.0
    # Middle layers get 6 bits
    
    # Alternative: use absolute thresholds
    use_absolute_thresholds: bool = False
    high_threshold_absolute: float = 0.7  # Normalized sensitivity > 0.7 → 8 bits
    low_threshold_absolute: float = 0.3   # Normalized sensitivity < 0.3 → 4 bits


@dataclass
class BitwidthAssignment:
    """Stores the bitwidth assignment for a layer"""
    layer_name: str
    assigned_bits: int
    sensitivity: float
    reason: str  # Why this bitwidth was chosen


class MixedPrecisionAssigner:
    """
    Assigns bit widths to layers based on sensitivity analysis.
    """
    
    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig()
        
    def assign_bitwidths(
        self,
        sensitivities: Dict[str, LayerSensitivity]
    ) -> Dict[str, BitwidthAssignment]:
        """
        Assign bit widths to all layers based on sensitivity.
        
        Args:
            sensitivities: Dictionary of layer sensitivities
            
        Returns:
            Dictionary mapping layer names to BitwidthAssignment
        """
        if not sensitivities:
            return {}
        
        # Get sensitivity values
        sens_values = [s.sensitivity_normalized for s in sensitivities.values()]
        
        if self.config.use_absolute_thresholds:
            high_thresh = self.config.high_threshold_absolute
            low_thresh = self.config.low_threshold_absolute
        else:
            # Compute percentile thresholds
            high_thresh = np.percentile(sens_values, self.config.high_threshold_percentile)
            low_thresh = np.percentile(sens_values, self.config.low_threshold_percentile)
        
        # Assign bitwidths
        assignments = {}
        
        for name, sens in sensitivities.items():
            norm_sens = sens.sensitivity_normalized
            
            if norm_sens >= high_thresh:
                bits = 8
                reason = f"High sensitivity ({norm_sens:.3f} >= {high_thresh:.3f})"
            elif norm_sens <= low_thresh:
                bits = 4
                reason = f"Low sensitivity ({norm_sens:.3f} <= {low_thresh:.3f})"
            else:
                bits = 6
                reason = f"Moderate sensitivity ({low_thresh:.3f} < {norm_sens:.3f} < {high_thresh:.3f})"
            
            assignments[name] = BitwidthAssignment(
                layer_name=name,
                assigned_bits=bits,
                sensitivity=sens.sensitivity,
                reason=reason
            )
        
        return assignments
    
    def get_bitwidth_distribution(
        self,
        assignments: Dict[str, BitwidthAssignment]
    ) -> Dict[int, int]:
        """
        Get count of layers at each bitwidth.
        
        Returns:
            Dict mapping bitwidth to count
        """
        distribution = {4: 0, 6: 0, 8: 0}
        for assignment in assignments.values():
            distribution[assignment.assigned_bits] += 1
        return distribution
    
    def compute_average_bitwidth(
        self,
        assignments: Dict[str, BitwidthAssignment],
        sensitivities: Dict[str, LayerSensitivity]
    ) -> float:
        """
        Compute parameter-weighted average bitwidth.
        
        Args:
            assignments: Bitwidth assignments per layer
            sensitivities: Layer sensitivities (for param counts)
            
        Returns:
            Weighted average bitwidth
        """
        total_params = 0
        weighted_bits = 0
        
        for name, assignment in assignments.items():
            if name in sensitivities:
                params = sensitivities[name].num_params
                total_params += params
                weighted_bits += params * assignment.assigned_bits
        
        return weighted_bits / total_params if total_params > 0 else 0


def assign_bitwidths(
    sensitivities: Dict[str, LayerSensitivity],
    high_percentile: float = 75.0,
    low_percentile: float = 25.0
) -> Dict[str, BitwidthAssignment]:
    """
    Convenience function to assign bitwidths based on sensitivity.
    
    Args:
        sensitivities: Layer sensitivity dictionary
        high_percentile: Percentile threshold for 8-bit assignment
        low_percentile: Percentile threshold for 4-bit assignment
        
    Returns:
        Dictionary of bitwidth assignments
    """
    config = MixedPrecisionConfig(
        high_threshold_percentile=high_percentile,
        low_threshold_percentile=low_percentile
    )
    assigner = MixedPrecisionAssigner(config)
    return assigner.assign_bitwidths(sensitivities)


def apply_mixed_precision(
    model: nn.Module,
    assignments: Dict[str, BitwidthAssignment]
) -> nn.Module:
    """
    Apply mixed-precision quantization to a model.
    
    Note: This creates a copy with quantized weights.
    
    Args:
        model: Original model
        assignments: Bitwidth assignments per layer
        
    Returns:
        Model with quantized weights
    """
    import copy
    model_q = copy.deepcopy(model)
    
    for name, assignment in assignments.items():
        # Navigate to the layer
        parts = name.split('.')
        module = model_q
        parent = None
        last_name = None
        
        for part in parts:
            parent = module
            last_name = part
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        # Quantize weights
        if hasattr(module, 'weight') and module.weight is not None:
            with torch.no_grad():
                module.weight.data = fake_quantize(
                    module.weight.data,
                    assignment.assigned_bits
                )
        
        if hasattr(module, 'bias') and module.bias is not None:
            with torch.no_grad():
                module.bias.data = fake_quantize(
                    module.bias.data,
                    assignment.assigned_bits
                )
    
    return model_q


def print_assignment_report(
    assignments: Dict[str, BitwidthAssignment],
    sensitivities: Optional[Dict[str, LayerSensitivity]] = None
):
    """Print a formatted report of bitwidth assignments."""
    print("\n" + "=" * 80)
    print("Mixed-Precision Bitwidth Assignment Report")
    print("=" * 80)
    print(f"{'Layer Name':<40} {'Bits':>6} {'Sensitivity':>12} {'Reason':<20}")
    print("-" * 80)
    
    # Sort by bitwidth (descending), then by name
    sorted_assignments = sorted(
        assignments.values(),
        key=lambda x: (-x.assigned_bits, x.layer_name)
    )
    
    for assignment in sorted_assignments:
        name = assignment.layer_name
        if len(name) > 38:
            name = "..." + name[-35:]
        print(f"{name:<40} {assignment.assigned_bits:>6} {assignment.sensitivity:>12.4f}")
    
    # Print distribution
    distribution = {4: 0, 6: 0, 8: 0}
    for a in assignments.values():
        distribution[a.assigned_bits] += 1
    
    print("-" * 80)
    print(f"Distribution: 8-bit: {distribution[8]}, 6-bit: {distribution[6]}, 4-bit: {distribution[4]}")
    
    # Compute average bitwidth
    if sensitivities:
        assigner = MixedPrecisionAssigner()
        avg_bits = assigner.compute_average_bitwidth(assignments, sensitivities)
        print(f"Parameter-weighted average bitwidth: {avg_bits:.2f}")
    
    print("=" * 80 + "\n")


def create_uniform_assignment(
    layer_names: List[str],
    bits: int
) -> Dict[str, BitwidthAssignment]:
    """
    Create uniform bitwidth assignment (for baseline comparison).
    
    Args:
        layer_names: List of layer names
        bits: Uniform bitwidth to assign
        
    Returns:
        Dictionary of assignments
    """
    return {
        name: BitwidthAssignment(
            layer_name=name,
            assigned_bits=bits,
            sensitivity=0.0,
            reason=f"Uniform {bits}-bit"
        )
        for name in layer_names
    }


