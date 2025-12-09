import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .quantizers import SymmetricUniformQuantizer, QuantizedModule
from tqdm import tqdm


class SensitivityBasedMixedPrecision:
    """
    Mixed-precision quantization based on layer sensitivity analysis.

    Uses L2 norm difference between full-precision and quantized outputs
    to determine optimal bit allocation per layer.

    Sensitivity: S_l = ||y_l_full - y_l_quant||_2
    """

    def __init__(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader,
                 bit_candidates: List[int] = [4, 6, 8], device: str = 'cuda'):
        """
        Args:
            model: Model to analyze
            calibration_loader: DataLoader for calibration data
            bit_candidates: List of bit widths to consider [low, medium, high]
            device: Device to run calibration on
        """
        self.model = model
        self.calibration_loader = calibration_loader
        self.bit_candidates = sorted(bit_candidates)  # [4, 6, 8]
        self.device = device
        self.layer_sensitivity = {}
        self.layer_bits = {}

    def _get_quantizable_layers(self) -> Dict[str, nn.Module]:
        """
        Get all quantizable layers (Linear and Conv2d) from the model.

        Returns:
            layers: Dictionary of {layer_name: layer_module}
        """
        layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers[name] = module
        return layers

    def _register_hooks(self, layer_name: str, layer: nn.Module,
                       activations: Dict[str, torch.Tensor]) -> Tuple:
        """
        Register forward hooks to capture layer outputs.

        Args:
            layer_name: Name of the layer
            layer: Layer module
            activations: Dictionary to store activations

        Returns:
            hook: Hook handle
        """

        def hook_fn(module, input, output):
            activations[layer_name] = output.detach().cpu()

        hook = layer.register_forward_hook(hook_fn)
        return hook

    def compute_layer_sensitivity(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Compute sensitivity for each layer by comparing full-precision
        and quantized outputs.

        Args:
            num_samples: Number of calibration samples to use

        Returns:
            sensitivity_scores: Dictionary of {layer_name: sensitivity}
        """
        print("Computing layer sensitivity scores...")
        layers = self._get_quantizable_layers()

        self.model.eval()
        self.model.to(self.device)

        # Test with lowest bit candidate to measure sensitivity
        test_bits = self.bit_candidates[0]  # Use lowest bits (e.g., 4-bit)

        sensitivity_scores = {}

        for layer_name, layer in tqdm(layers.items(), desc="Analyzing layers"):
            full_precision_outputs = []
            quantized_outputs = []

            # Hook to capture outputs
            fp_activations = {}
            hook = self._register_hooks(layer_name, layer, fp_activations)

            # Collect full-precision outputs
            sample_count = 0
            with torch.no_grad():
                for batch in self.calibration_loader:
                    if sample_count >= num_samples:
                        break

                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)

                    # Forward pass
                    _ = self.model(inputs)

                    if layer_name in fp_activations:
                        full_precision_outputs.append(fp_activations[layer_name].clone())
                        fp_activations.clear()

                    sample_count += inputs.size(0)

            hook.remove()

            # Quantize layer weights
            original_weight = layer.weight.data.clone()
            quantizer = SymmetricUniformQuantizer(bits=test_bits, mode='per_channel')
            layer.weight.data = quantizer(layer.weight.data)

            # Collect quantized outputs
            q_activations = {}
            hook = self._register_hooks(layer_name, layer, q_activations)

            sample_count = 0
            with torch.no_grad():
                for batch in self.calibration_loader:
                    if sample_count >= num_samples:
                        break

                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)

                    _ = self.model(inputs)

                    if layer_name in q_activations:
                        quantized_outputs.append(q_activations[layer_name].clone())
                        q_activations.clear()

                    sample_count += inputs.size(0)

            hook.remove()

            # Restore original weights
            layer.weight.data = original_weight

            # Compute sensitivity: L2 norm of difference
            if len(full_precision_outputs) > 0 and len(quantized_outputs) > 0:
                fp_concat = torch.cat(full_precision_outputs, dim=0)
                q_concat = torch.cat(quantized_outputs, dim=0)

                sensitivity = torch.norm(fp_concat - q_concat, p=2).item()
                sensitivity_scores[layer_name] = sensitivity
            else:
                sensitivity_scores[layer_name] = 0.0

        self.layer_sensitivity = sensitivity_scores
        return sensitivity_scores

    def assign_bitwidths(self, sensitivity_thresholds: Optional[Tuple[float, float]] = None) -> Dict[str, int]:
        """
        Assign bitwidths to layers based on sensitivity scores.

        Args:
            sensitivity_thresholds: (low_threshold, high_threshold) percentiles
                                   If None, uses (33rd, 67th) percentiles

        Returns:
            layer_bits: Dictionary of {layer_name: assigned_bits}
        """
        if not self.layer_sensitivity:
            raise ValueError("Must compute sensitivity first using compute_layer_sensitivity()")

        sensitivities = list(self.layer_sensitivity.values())

        if sensitivity_thresholds is None:
            # Use percentiles: low 33%, medium 33%, high 33%
            low_threshold = torch.quantile(torch.tensor(sensitivities), 0.33).item()
            high_threshold = torch.quantile(torch.tensor(sensitivities), 0.67).item()
        else:
            low_threshold, high_threshold = sensitivity_thresholds

        print(f"\nSensitivity thresholds: low={low_threshold:.2e}, high={high_threshold:.2e}")

        # Assign bits based on sensitivity
        layer_bits = {}
        for layer_name, sensitivity in self.layer_sensitivity.items():
            if sensitivity <= low_threshold:
                # Low sensitivity -> use lowest bits
                bits = self.bit_candidates[0]
            elif sensitivity <= high_threshold:
                # Medium sensitivity -> use medium bits
                bits = self.bit_candidates[1] if len(self.bit_candidates) > 1 else self.bit_candidates[0]
            else:
                # High sensitivity -> use highest bits
                bits = self.bit_candidates[-1]

            layer_bits[layer_name] = bits

        self.layer_bits = layer_bits

        # Print summary
        bit_distribution = {}
        for bits in self.bit_candidates:
            bit_distribution[bits] = sum(1 for b in layer_bits.values() if b == bits)

        print("\nBit allocation summary:")
        for bits, count in sorted(bit_distribution.items()):
            print(f"  {bits}-bit: {count} layers")

        return layer_bits

    def apply_mixed_precision(self, activation_bits: int = 8) -> nn.Module:
        """
        Apply mixed-precision quantization to the model based on computed bit assignments.

        Args:
            activation_bits: Bits for activation quantization (uniform across layers)

        Returns:
            quantized_model: Model with mixed-precision quantization applied
        """
        if not self.layer_bits:
            raise ValueError("Must assign bitwidths first using assign_bitwidths()")

        print("\nApplying mixed-precision quantization...")

        layers = self._get_quantizable_layers()

        for layer_name, layer in layers.items():
            if layer_name in self.layer_bits:
                bits = self.layer_bits[layer_name]

                # Quantize weights
                if hasattr(layer, 'weight') and layer.weight is not None:
                    with torch.no_grad():
                        quantizer = SymmetricUniformQuantizer(
                            bits=bits,
                            mode='per_channel',
                            channel_dim=0
                        )
                        layer.weight.data = quantizer(layer.weight.data)

                print(f"  {layer_name}: {bits}-bit")

        print("Mixed-precision quantization applied successfully!")
        return self.model

    def get_bit_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of bit assignments across layers.

        Returns:
            distribution: {bits: count}
        """
        if not self.layer_bits:
            return {}

        distribution = {}
        for bits in self.layer_bits.values():
            distribution[bits] = distribution.get(bits, 0) + 1

        return distribution

    def print_sensitivity_report(self, top_k: int = 10):
        """
        Print a report of the most and least sensitive layers.

        Args:
            top_k: Number of top/bottom layers to show
        """
        if not self.layer_sensitivity:
            print("No sensitivity data available.")
            return

        sorted_layers = sorted(self.layer_sensitivity.items(),
                             key=lambda x: x[1], reverse=True)

        print(f"\n{'='*60}")
        print("SENSITIVITY ANALYSIS REPORT")
        print(f"{'='*60}")

        print(f"\nTop {top_k} Most Sensitive Layers:")
        print(f"{'Layer Name':<40} {'Sensitivity':<15} {'Assigned Bits'}")
        print("-" * 60)
        for layer_name, sensitivity in sorted_layers[:top_k]:
            bits = self.layer_bits.get(layer_name, 'N/A')
            print(f"{layer_name:<40} {sensitivity:<15.2e} {bits}")

        print(f"\nTop {top_k} Least Sensitive Layers:")
        print(f"{'Layer Name':<40} {'Sensitivity':<15} {'Assigned Bits'}")
        print("-" * 60)
        for layer_name, sensitivity in sorted_layers[-top_k:]:
            bits = self.layer_bits.get(layer_name, 'N/A')
            print(f"{layer_name:<40} {sensitivity:<15.2e} {bits}")

        print(f"\n{'='*60}\n")
