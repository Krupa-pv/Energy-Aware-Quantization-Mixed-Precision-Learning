"""
Configuration for Energy-Aware Quantization Experiments
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class QuantizationConfig:
    """Configuration for quantization experiments"""
    
    # Bitwidth options
    bitwidths: List[int] = field(default_factory=lambda: [4, 6, 8])
    full_precision_bits: int = 32
    
    # Mixed-precision thresholds (percentiles)
    high_sensitivity_percentile: float = 75.0  # Top 25% → 8 bits
    low_sensitivity_percentile: float = 25.0   # Bottom 25% → 4 bits
    # Middle 50% → 6 bits
    
    # Calibration
    num_calibration_samples: int = 256
    calibration_batch_size: int = 32


@dataclass
class EnergyConfig:
    """Configuration for energy modeling"""
    
    # Energy scaling factors (relative units)
    # E_MAC(b) ∝ b²
    # E_DRAM(b) ∝ b
    
    # Reference energy at 8-bit (arbitrary units, for relative comparison)
    e_mac_8bit: float = 1.0      # MAC energy at 8-bit
    e_dram_8bit: float = 1.0     # DRAM access energy at 8-bit
    
    # Scaling: E_MAC scales with b², E_DRAM scales with b
    def get_mac_energy(self, bits: int) -> float:
        """Get MAC energy for given bitwidth (scales as b²)"""
        return self.e_mac_8bit * (bits / 8) ** 2
    
    def get_dram_energy(self, bits: int) -> float:
        """Get DRAM energy for given bitwidth (scales as b)"""
        return self.e_dram_8bit * (bits / 8)


@dataclass
class ModelConfig:
    """Configuration for model experiments"""
    
    # ResNet-18 config
    resnet_dataset: str = "cifar100"
    resnet_num_classes: int = 100
    resnet_input_size: int = 32
    
    # DeiT-Tiny config
    deit_dataset: str = "imagenet"
    deit_num_classes: int = 1000
    deit_input_size: int = 224
    
    # Paths
    data_root: str = "./data"
    results_dir: str = "./results"
    figures_dir: str = "./figures"


@dataclass 
class ExperimentConfig:
    """Master configuration combining all configs"""
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Device
    device: str = "cuda"  # Will fallback to CPU if CUDA unavailable
    
    # Random seed for reproducibility
    seed: int = 42


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()


