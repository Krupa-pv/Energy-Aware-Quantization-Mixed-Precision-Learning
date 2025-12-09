import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class EnergyModel:
    """
    Analytical energy model for quantized neural networks.

    Energy: E = Σ N_MAC^(l) * E_MAC(b_l) + Σ N_mem^(l) * E_DRAM(b_l)

    Where:
    - E_MAC(b) ∝ b^2 (multiplier energy scales quadratically with bit width)
    - E_DRAM(b) ∝ b (memory access energy scales linearly with bit width)
    - N_MAC: Number of multiply-accumulate operations
    - N_mem: Number of memory accesses (weight + activation reads)
    """

    def __init__(self, base_mac_energy: float = 1.0, base_dram_energy: float = 10.0,
                 base_bits: int = 8):
        """
        Args:
            base_mac_energy: Energy for one MAC operation at base_bits (normalized)
            base_dram_energy: Energy for one DRAM access at base_bits (normalized)
            base_bits: Reference bit width for normalization (typically 8)
        """
        self.base_mac_energy = base_mac_energy
        self.base_dram_energy = base_dram_energy
        self.base_bits = base_bits

    def compute_mac_energy(self, bits: int) -> float:
        """
        Compute energy for a single MAC operation at given bit width.

        E_MAC(b) = base_mac_energy * (b / base_bits)^2

        Args:
            bits: Bit width

        Returns:
            energy: Energy per MAC operation
        """
        return self.base_mac_energy * (bits / self.base_bits) ** 2

    def compute_dram_energy(self, bits: int) -> float:
        """
        Compute energy for a single DRAM access at given bit width.

        E_DRAM(b) = base_dram_energy * (b / base_bits)

        Args:
            bits: Bit width

        Returns:
            energy: Energy per DRAM access
        """
        return self.base_dram_energy * (bits / self.base_bits)

    def compute_layer_energy(self, layer: nn.Module, input_shape: Tuple[int, ...],
                           weight_bits: int, activation_bits: int) -> Dict[str, float]:
        """
        Compute energy consumption for a single layer.

        Args:
            layer: Neural network layer
            input_shape: Shape of input tensor (B, C, H, W) or (B, seq_len, dim)
            weight_bits: Bit width for weights
            activation_bits: Bit width for activations

        Returns:
            energy_dict: Dictionary with 'mac_energy', 'mem_energy', 'total_energy', 'macs', 'mem_accesses'
        """
        if isinstance(layer, nn.Linear):
            return self._compute_linear_energy(layer, input_shape, weight_bits, activation_bits)
        elif isinstance(layer, nn.Conv2d):
            return self._compute_conv2d_energy(layer, input_shape, weight_bits, activation_bits)
        else:
            # For other layers, return zero energy
            return {
                'mac_energy': 0.0,
                'mem_energy': 0.0,
                'total_energy': 0.0,
                'macs': 0,
                'mem_accesses': 0
            }

    def _compute_linear_energy(self, layer: nn.Linear, input_shape: Tuple[int, ...],
                              weight_bits: int, activation_bits: int) -> Dict[str, float]:
        """
        Compute energy for a Linear layer.

        Args:
            layer: nn.Linear layer
            input_shape: (batch_size, in_features) or (batch_size, seq_len, in_features)
            weight_bits: Bits for weights
            activation_bits: Bits for activations

        Returns:
            energy_dict: Energy breakdown
        """
        # Parse input shape
        if len(input_shape) == 2:
            batch_size, in_features = input_shape
            seq_len = 1
        elif len(input_shape) == 3:
            batch_size, seq_len, in_features = input_shape
        else:
            raise ValueError(f"Unsupported input shape for Linear: {input_shape}")

        out_features = layer.out_features

        # Number of MACs: batch_size * seq_len * in_features * out_features
        num_macs = batch_size * seq_len * in_features * out_features

        # Memory accesses:
        # - Weight reads: in_features * out_features
        # - Activation reads: batch_size * seq_len * in_features
        # - Activation writes: batch_size * seq_len * out_features
        weight_reads = in_features * out_features
        activation_reads = batch_size * seq_len * in_features
        activation_writes = batch_size * seq_len * out_features

        # Energy calculation
        # MAC uses both weight and activation bits (average for simplicity)
        avg_bits = (weight_bits + activation_bits) / 2
        mac_energy = num_macs * self.compute_mac_energy(avg_bits)

        # Memory energy
        weight_mem_energy = weight_reads * self.compute_dram_energy(weight_bits)
        activation_mem_energy = (activation_reads + activation_writes) * self.compute_dram_energy(activation_bits)
        mem_energy = weight_mem_energy + activation_mem_energy

        total_mem_accesses = weight_reads + activation_reads + activation_writes

        return {
            'mac_energy': mac_energy,
            'mem_energy': mem_energy,
            'total_energy': mac_energy + mem_energy,
            'macs': num_macs,
            'mem_accesses': total_mem_accesses
        }

    def _compute_conv2d_energy(self, layer: nn.Conv2d, input_shape: Tuple[int, ...],
                              weight_bits: int, activation_bits: int) -> Dict[str, float]:
        """
        Compute energy for a Conv2d layer.

        Args:
            layer: nn.Conv2d layer
            input_shape: (batch_size, in_channels, height, width)
            weight_bits: Bits for weights
            activation_bits: Bits for activations

        Returns:
            energy_dict: Energy breakdown
        """
        batch_size, in_channels, h_in, w_in = input_shape

        # Output dimensions
        kernel_h, kernel_w = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        stride_h, stride_w = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
        padding_h, padding_w = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)

        h_out = (h_in + 2 * padding_h - kernel_h) // stride_h + 1
        w_out = (w_in + 2 * padding_w - kernel_w) // stride_w + 1
        out_channels = layer.out_channels

        # Number of MACs
        # For each output pixel: kernel_h * kernel_w * in_channels multiplications
        num_macs = batch_size * out_channels * h_out * w_out * kernel_h * kernel_w * in_channels

        # Memory accesses
        # Weight reads: out_channels * in_channels * kernel_h * kernel_w
        # Activation reads: batch_size * in_channels * h_in * w_in
        # Activation writes: batch_size * out_channels * h_out * w_out
        weight_reads = out_channels * in_channels * kernel_h * kernel_w
        activation_reads = batch_size * in_channels * h_in * w_in
        activation_writes = batch_size * out_channels * h_out * w_out

        # Energy calculation
        avg_bits = (weight_bits + activation_bits) / 2
        mac_energy = num_macs * self.compute_mac_energy(avg_bits)

        weight_mem_energy = weight_reads * self.compute_dram_energy(weight_bits)
        activation_mem_energy = (activation_reads + activation_writes) * self.compute_dram_energy(activation_bits)
        mem_energy = weight_mem_energy + activation_mem_energy

        total_mem_accesses = weight_reads + activation_reads + activation_writes

        return {
            'mac_energy': mac_energy,
            'mem_energy': mem_energy,
            'total_energy': mac_energy + mem_energy,
            'macs': num_macs,
            'mem_accesses': total_mem_accesses
        }


def compute_model_energy(model: nn.Module, input_shape: Tuple[int, ...],
                        weight_bits: int = 8, activation_bits: int = 8,
                        layer_bits: Optional[Dict[str, int]] = None,
                        device: str = 'cpu') -> Dict[str, any]:
    """
    Compute total energy consumption for a model.

    Args:
        model: Neural network model
        input_shape: Input tensor shape
        weight_bits: Default bits for weights (if layer_bits not provided)
        activation_bits: Bits for activations
        layer_bits: Optional dict mapping layer names to bit widths (for mixed precision)
        device: Device to run on

    Returns:
        energy_report: Dictionary with energy breakdown and statistics
    """
    energy_model = EnergyModel()

    model.eval()
    model.to(device)

    total_mac_energy = 0.0
    total_mem_energy = 0.0
    total_macs = 0
    total_mem_accesses = 0

    layer_energies = {}

    # Create a dummy input to trace shapes
    dummy_input = torch.randn(*input_shape).to(device)

    # Dictionary to store intermediate shapes
    shapes = {}

    def shape_hook(name):
        def hook_fn(module, input, output):
            shapes[name] = output.shape
        return hook_fn

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hook = module.register_forward_hook(shape_hook(name))
            hooks.append(hook)

    # Forward pass to get shapes
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute energy for each layer
    current_shape = input_shape
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Determine bit width for this layer
            if layer_bits and name in layer_bits:
                layer_weight_bits = layer_bits[name]
            else:
                layer_weight_bits = weight_bits

            # Compute energy
            energy_dict = energy_model.compute_layer_energy(
                module, current_shape, layer_weight_bits, activation_bits
            )

            layer_energies[name] = {
                'bits': layer_weight_bits,
                **energy_dict
            }

            total_mac_energy += energy_dict['mac_energy']
            total_mem_energy += energy_dict['mem_energy']
            total_macs += energy_dict['macs']
            total_mem_accesses += energy_dict['mem_accesses']

            # Update shape for next layer
            if name in shapes:
                current_shape = shapes[name]

    total_energy = total_mac_energy + total_mem_energy

    energy_report = {
        'total_energy': total_energy,
        'mac_energy': total_mac_energy,
        'mem_energy': total_mem_energy,
        'total_macs': total_macs,
        'total_mem_accesses': total_mem_accesses,
        'mac_energy_percentage': (total_mac_energy / total_energy * 100) if total_energy > 0 else 0,
        'mem_energy_percentage': (total_mem_energy / total_energy * 100) if total_energy > 0 else 0,
        'layer_energies': layer_energies
    }

    return energy_report


def print_energy_report(energy_report: Dict[str, any], top_k: int = 10):
    """
    Print a formatted energy consumption report.

    Args:
        energy_report: Energy report from compute_model_energy
        top_k: Number of top energy-consuming layers to show
    """
    print(f"\n{'='*70}")
    print("ENERGY CONSUMPTION REPORT")
    print(f"{'='*70}")

    print(f"\nOverall Statistics:")
    print(f"  Total Energy:        {energy_report['total_energy']:.2e} (normalized units)")
    print(f"  MAC Energy:          {energy_report['mac_energy']:.2e} ({energy_report['mac_energy_percentage']:.1f}%)")
    print(f"  Memory Energy:       {energy_report['mem_energy']:.2e} ({energy_report['mem_energy_percentage']:.1f}%)")
    print(f"  Total MACs:          {energy_report['total_macs']:.2e}")
    print(f"  Total Mem Accesses:  {energy_report['total_mem_accesses']:.2e}")

    if 'layer_energies' in energy_report:
        sorted_layers = sorted(energy_report['layer_energies'].items(),
                             key=lambda x: x[1]['total_energy'], reverse=True)

        print(f"\nTop {top_k} Energy-Consuming Layers:")
        print(f"{'Layer Name':<35} {'Bits':<6} {'Total Energy':<15} {'% of Total'}")
        print("-" * 70)

        for i, (layer_name, layer_energy) in enumerate(sorted_layers[:top_k]):
            percentage = (layer_energy['total_energy'] / energy_report['total_energy'] * 100)
            print(f"{layer_name:<35} {layer_energy['bits']:<6} "
                  f"{layer_energy['total_energy']:<15.2e} {percentage:>6.1f}%")

    print(f"\n{'='*70}\n")


def compute_energy_reduction(baseline_energy: float, quantized_energy: float) -> Tuple[float, float]:
    """
    Compute energy reduction from quantization.

    Args:
        baseline_energy: Energy of full-precision model
        quantized_energy: Energy of quantized model

    Returns:
        reduction_ratio: Energy reduction ratio (quantized / baseline)
        reduction_percentage: Energy reduction percentage
    """
    reduction_ratio = quantized_energy / baseline_energy
    reduction_percentage = (1 - reduction_ratio) * 100

    return reduction_ratio, reduction_percentage
