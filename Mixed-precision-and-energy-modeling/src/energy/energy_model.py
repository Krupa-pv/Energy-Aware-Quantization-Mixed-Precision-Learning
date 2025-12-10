"""
Energy Model
============
Analytical model for estimating inference energy consumption.

E = Σ_l N^(l)_MAC × E_MAC(b_l) + Σ_l N^(l)_mem × E_DRAM(b_l)

Where:
- E_MAC(b) ∝ b² (MAC energy scales quadratically with bitwidth)
- E_DRAM(b) ∝ b (Memory access energy scales linearly with bitwidth)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LayerEnergyProfile:
    """Energy profile for a single layer"""
    name: str
    layer_type: str
    macs: int              # Multiply-accumulate operations
    memory_accesses: int   # Memory read/write operations
    params: int            # Number of parameters

    # Energy at different bitwidths
    energy_32bit: float = 0.0
    energy_8bit: float = 0.0
    energy_6bit: float = 0.0
    energy_4bit: float = 0.0


class EnergyModel:
    """
    Analytical energy model for neural network inference.

    Based on the energy model from the proposal:
    E = Σ N_MAC × E_MAC(b) + Σ N_mem × E_DRAM(b)

    Where:
    - E_MAC(b) ∝ b² (energy for multiply-accumulate scales with bit²)
    - E_DRAM(b) ∝ b (memory access energy scales with bits)
    """

    # Energy coefficients (relative units, normalized to 8-bit = 1.0)
    # These are approximate values based on hardware studies
    E_MAC_COEFF = 1.0   # Base MAC energy at 8-bit
    E_DRAM_COEFF = 200.0  # DRAM access is ~200x more expensive than MAC

    def __init__(
        self,
        mac_coeff: float = 1.0,
        dram_coeff: float = 200.0,
        reference_bits: int = 8
    ):
        """
        Initialize energy model.

        Args:
            mac_coeff: Base energy coefficient for MAC operations
            dram_coeff: Base energy coefficient for DRAM access
            reference_bits: Reference bitwidth for normalization
        """
        self.mac_coeff = mac_coeff
        self.dram_coeff = dram_coeff
        self.reference_bits = reference_bits

    def e_mac(self, bits: int) -> float:
        """
        Compute MAC energy scaling factor.

        E_MAC(b) ∝ b²
        """
        return self.mac_coeff * (bits / self.reference_bits) ** 2

    def e_dram(self, bits: int) -> float:
        """
        Compute DRAM access energy scaling factor.

        E_DRAM(b) ∝ b
        """
        return self.dram_coeff * (bits / self.reference_bits)

    def compute_layer_energy(
        self,
        macs: int,
        memory_accesses: int,
        bits: int
    ) -> Tuple[float, float, float]:
        """
        Compute energy for a single layer.

        Args:
            macs: Number of MAC operations
            memory_accesses: Number of memory accesses
            bits: Bitwidth for quantization

        Returns:
            Tuple of (total_energy, mac_energy, memory_energy)
        """
        mac_energy = macs * self.e_mac(bits)
        mem_energy = memory_accesses * self.e_dram(bits)
        total = mac_energy + mem_energy

        return total, mac_energy, mem_energy

    def compute_model_energy(
        self,
        layer_profiles: List[LayerEnergyProfile],
        bitwidth_assignments: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Compute total model energy with given bitwidth assignments.

        Args:
            layer_profiles: List of layer energy profiles
            bitwidth_assignments: Dict mapping layer name to bitwidth

        Returns:
            Dictionary with energy breakdown
        """
        total_mac_energy = 0.0
        total_mem_energy = 0.0
        layer_energies = {}

        for profile in layer_profiles:
            bits = bitwidth_assignments.get(profile.name, self.reference_bits)

            layer_total, mac_e, mem_e = self.compute_layer_energy(
                profile.macs,
                profile.memory_accesses,
                bits
            )

            layer_energies[profile.name] = {
                'total': layer_total,
                'mac': mac_e,
                'memory': mem_e,
                'bits': bits
            }

            total_mac_energy += mac_e
            total_mem_energy += mem_e

        return {
            'total_energy': total_mac_energy + total_mem_energy,
            'mac_energy': total_mac_energy,
            'memory_energy': total_mem_energy,
            'layer_energies': layer_energies
        }

    def compare_configurations(
        self,
        layer_profiles: List[LayerEnergyProfile],
        configurations: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict]:
        """
        Compare energy across multiple quantization configurations.

        Args:
            layer_profiles: List of layer energy profiles
            configurations: Dict mapping config name to bitwidth assignments
                           e.g., {'fp32': {...}, '8bit': {...}, 'mixed': {...}}

        Returns:
            Dictionary with energy comparison
        """
        results = {}

        for config_name, assignments in configurations.items():
            energy = self.compute_model_energy(layer_profiles, assignments)
            results[config_name] = energy

        # Compute relative savings (compared to FP32)
        if 'FP32' in results:
            fp32_energy = results['FP32']['total_energy']
            for config_name, result in results.items():
                result['relative_energy'] = result['total_energy'] / fp32_energy
                result['energy_savings'] = 1.0 - result['relative_energy']

        return results


def compute_conv_macs(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    output_size: Tuple[int, int]
) -> int:
    """
    Compute MACs for a Conv2d layer.

    MACs = K_h × K_w × C_in × C_out × H_out × W_out
    """
    k_h, k_w = kernel_size if isinstance(
        kernel_size, tuple) else (kernel_size, kernel_size)
    h_out, w_out = output_size
    return k_h * k_w * in_channels * out_channels * h_out * w_out


def compute_linear_macs(in_features: int, out_features: int) -> int:
    """
    Compute MACs for a Linear layer.

    MACs = in_features × out_features
    """
    return in_features * out_features


def compute_layer_macs(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    device: str = "cuda"
) -> List[LayerEnergyProfile]:
    """
    Compute MACs and memory accesses for all layers in a model.

    Args:
        model: Neural network model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run profiling on

    Returns:
        List of LayerEnergyProfile for each layer
    """
    profiles = []
    hooks = []

    def make_hook(name, module):
        def hook(mod, inp, out):
            profile = LayerEnergyProfile(
                name=name,
                layer_type=type(mod).__name__,
                macs=0,
                memory_accesses=0,
                params=sum(p.numel() for p in mod.parameters())
            )

            if isinstance(mod, nn.Conv2d):
                # MACs for conv layer
                out_h, out_w = out.shape[2], out.shape[3]
                k_h, k_w = mod.kernel_size
                profile.macs = compute_conv_macs(
                    mod.in_channels,
                    mod.out_channels,
                    (k_h, k_w),
                    (out_h, out_w)
                )
                # Memory: read weights + read input + write output
                profile.memory_accesses = (
                    profile.params +  # weights
                    inp[0].numel() +  # input
                    out.numel()       # output
                )

            elif isinstance(mod, nn.Linear):
                profile.macs = compute_linear_macs(
                    mod.in_features,
                    mod.out_features
                )
                # Memory: read weights + read input + write output
                profile.memory_accesses = (
                    profile.params +
                    inp[0].numel() +
                    out.numel()
                )

            profiles.append(profile)
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hook = module.register_forward_hook(make_hook(name, module))
            hooks.append(hook)

    # Forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(*input_size).to(device)
        _ = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return profiles


def estimate_model_energy(
    model: nn.Module,
    input_size: Tuple[int, int, int, int],
    bitwidth_assignments: Dict[str, int],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Convenience function to estimate model energy.

    Args:
        model: Neural network model
        input_size: Input tensor size (B, C, H, W)
        bitwidth_assignments: Dict mapping layer names to bitwidths
        device: Device to run on

    Returns:
        Energy estimation dictionary
    """
    # Get layer profiles
    profiles = compute_layer_macs(model, input_size, device)

    # Compute energy
    energy_model = EnergyModel()
    return energy_model.compute_model_energy(profiles, bitwidth_assignments)


def print_energy_report(
    energy_results: Dict[str, Dict],
    title: str = "Energy Comparison Report"
):
    """Print formatted energy comparison report."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"{'Configuration':<20} {'Total Energy':>15} {'Relative':>12} {'Savings':>12}")
    print("-" * 70)

    for config_name, result in energy_results.items():
        total = result['total_energy']
        relative = result.get('relative_energy', 1.0)
        savings = result.get('energy_savings', 0.0)

        print(f"{config_name:<20} {total:>15.2e} {relative:>11.2%} {savings:>11.2%}")

    print("=" * 70 + "\n")


def create_bitwidth_dict(
    layer_names: List[str],
    bits: int
) -> Dict[str, int]:
    """Create uniform bitwidth dictionary."""
    return {name: bits for name in layer_names}
