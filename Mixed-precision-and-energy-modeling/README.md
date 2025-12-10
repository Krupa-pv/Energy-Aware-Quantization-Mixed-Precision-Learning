# Energy-Aware Quantization and Mixed-Precision Learning

This project implements post-training quantization (PTQ) and mixed-precision inference for energy-efficient deep learning, focusing on ResNet-18 and DeiT-Tiny models.

## 🎯 Project Overview

**Goal:** Explore lightweight, training-free approaches for improving energy efficiency without significantly reducing model accuracy.

**Key Components:**
- **Sensitivity Analysis**: Measure how each layer responds to quantization
- **Mixed-Precision Assignment**: Assign different bit widths (4/6/8-bit) based on sensitivity
- **Energy Modeling**: Analytical model for estimating inference energy consumption
- **Visualization**: Pareto curves, sensitivity distributions, energy breakdowns

## 🔗 Integration with Teammates' Work

This module uses the trained models from our teammates:
- **ResNet-18**: Salma's CIFAR-100 trained model from `Resnet_Cifar100_PTQ/checkpoints/resnet18_cifar100_trained.pth`
- **DeiT-Tiny**: Krupa's ImageNet pretrained model from timm (same as `DeiT_RA_PTQ`)

The accuracy values in results come from their actual experiments.

## 📁 Project Structure

```
397final/
├── config.py                 # Configuration dataclasses
├── run_experiment.py         # Main experiment runner
├── requirements.txt          # Python dependencies
├── PROJECT_PROPOSAL.md       # Original project proposal
├── README.md                 # This file
│
├── src/
│   ├── models/
│   │   └── model_loader.py   # Load ResNet-18 and DeiT-Tiny
│   │
│   ├── quantization/
│   │   ├── quantize.py       # Quantization utilities
│   │   ├── sensitivity.py    # Layer sensitivity measurement
│   │   └── mixed_precision.py # Bitwidth assignment
│   │
│   ├── energy/
│   │   └── energy_model.py   # Analytical energy estimation
│   │
│   ├── visualization/
│   │   └── plots.py          # Pareto curves, plots
│   │
│   └── utils/
│       └── data_loader.py    # Data loading utilities
│
├── data/                     # Downloaded datasets
├── results/                  # Experiment outputs
└── figures/                  # Generated plots
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Run ResNet-18 experiments (CIFAR-100)
python run_experiment.py --model resnet

# Run DeiT-Tiny experiments (ImageNet)
python run_experiment.py --model deit

# Run all experiments
python run_experiment.py --model all

# Use CPU if no GPU available
python run_experiment.py --model resnet --device cpu
```

### 3. View Results

Results are saved to `./results/`:
- `results/resnet/` - ResNet-18 results and plots
- `results/deit/` - DeiT-Tiny results and plots

## 📊 Key Algorithms

### Sensitivity Measurement

```python
S_l = ||y_l^full - y_l^quant||_2
```

Where `y_l^full` is the full-precision output and `y_l^quant` is the quantized output.

### Mixed-Precision Assignment

| Sensitivity Level | Bitwidth | Assignment Rule |
|-------------------|----------|-----------------|
| High (top 25%)    | 8 bits   | Preserve accuracy |
| Moderate (middle) | 6 bits   | Balance |
| Low (bottom 25%)  | 4 bits   | Maximize compression |

### Energy Model

```
E = Σ_l N^(l)_MAC × E_MAC(b_l) + Σ_l N^(l)_mem × E_DRAM(b_l)
```

Where:
- `E_MAC(b) ∝ b²` (MAC energy scales quadratically)
- `E_DRAM(b) ∝ b` (Memory energy scales linearly)

## 📈 Expected Outputs

1. **Pareto Frontier Plot**: Energy vs Accuracy trade-off
2. **Sensitivity Distribution**: Histogram of layer sensitivities
3. **Bitwidth Distribution**: Pie chart of assigned bit widths
4. **Energy Breakdown**: Comparison across configurations
5. **Full Report Figure**: 4-panel comprehensive analysis

## 🔧 Configuration

Edit `config.py` to customize:

```python
@dataclass
class QuantizationConfig:
    bitwidths: List[int] = [4, 6, 8]
    high_sensitivity_percentile: float = 75.0
    low_sensitivity_percentile: float = 25.0
    num_calibration_samples: int = 256
```

## 👥 Team

- **Salma Bhar** (sxb1283) - PTQ for ResNet-18
- **Krupa Venkatesan** (kxv178) - PTQ for DeiT-Tiny + ranking-aware module
- **Wiam Skakri** (wxs428) - Mixed-precision assignment + energy modeling

## 📚 References

1. Z. Liu et al., "Post-Training Quantization for Vision Transformer," ICML 2021. [arXiv:2106.14156](https://arxiv.org/abs/2106.14156)

2. N. Spingarn et al., "Energy Awareness in Low Precision Neural Networks," ICLR 2023. [arXiv:2202.02783](https://arxiv.org/abs/2202.02783)


