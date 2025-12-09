# Vision Transformer Quantization

Implementation of post-training quantization for vision transformers and CNNs.

## Overview

Quantization techniques for reducing model size and improving inference speed:
- Vision transformers with attention-aware quantization
- CNNs with per-channel quantization
- Energy modeling
- Mixed-precision allocation

Based on papers from NeurIPS 2022 and SysML 2020.

## Project Structure

```
src/
├── quantization/      # quantization implementations
├── models/            # resnet and vit models
├── energy/            # energy estimation
└── utils/             # data loading and evaluation

experiments/           # experiment scripts
finetune_and_quantize.py  # main training script
```

## Features

- Symmetric uniform quantization (4/6/8-bit)
- Per-tensor and per-channel modes
- Ranking-aware quantization for transformers
- Sensitivity-based mixed-precision
- Energy modeling: E = Σ N_MAC * E_MAC(b²) + Σ N_mem * E_DRAM(b)

## Models

- ResNet-18 on CIFAR-10 (95.18% accuracy baseline)
- DeiT-Tiny on ImageNet (5.7M parameters, 85.34% baseline on 10K images)

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 2.0+

## Usage

Run DeiT-Tiny quantization experiments:
```bash
python run_deit_experiments.py
```

Run complete analysis with sensitivity and ranking-aware comparison:
```bash
python complete_deit_analysis.py
```

Generate visualization plots:
```bash
python create_pareto_plots.py
```

## Results

### ResNet-18 (CIFAR-10)
- Baseline: 95.18% accuracy
- 8-bit: 95.23% accuracy (0.05% improvement), 86% energy reduction
- 6-bit: 95.13% accuracy (0.05% drop), 90% energy reduction
- 4-bit: 93.13% accuracy (2.05% drop), 94% energy reduction

### DeiT-Tiny (ImageNet-1K, 10K test images)
**Standard PTQ:**
- Baseline: 85.34% top-1 accuracy
- 8-bit: 85.10% (0.24% drop)
- 6-bit: 85.01% (0.33% drop)
- 4-bit: 83.04% (2.30% drop)

**Ranking-Aware PTQ:**
- 8-bit: 85.18% (0.16% drop, +0.08% vs standard)
- 6-bit: 85.11% (0.23% drop, +0.10% vs standard)
- 4-bit: 74.90% (10.44% drop, -8.14% vs standard)

**Key Finding:** Ranking-aware quantization improves accuracy at 8/6-bit by preserving attention score ordering, but hurts performance at very low precision (4-bit).

### Sensitivity Analysis (49 DeiT layers)
- Most sensitive: blocks.8.mlp.fc2 (0.28)
- Least sensitive: blocks.7.attn.proj (0.06)
- Sensitivity ratio: 4.7x
- Finding: MLP layers require higher precision than attention projections

## Technical Details

### Quantization
Uses symmetric uniform quantization:
```
q = clamp(round(x / s), -2^(b-1), 2^(b-1) - 1)
```
where s = max(|x|) / (2^(b-1) - 1)

### Energy Model
- MAC operations: energy scales as b²
- Memory access: energy scales as b
- Theoretical reduction validated through inference measurements

## References

- Post-Training Quantization for Vision Transformer, Liu et al., NeurIPS 2021
- Energy Awareness in Low Precision Neural Networks, Eliezer et al., ICLR 2023
