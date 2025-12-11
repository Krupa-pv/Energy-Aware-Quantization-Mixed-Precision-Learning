# ResNet-18 on CIFAR-100 with Uniform Symmetric PTQ

## Overview
This project implements training-free Post-Training Quantization (PTQ) for ResNet-18 trained on CIFAR-100. </br>
The goal is to evaluate the impact of uniform symmetric quantization at 8-bit, 6-bit, and 4-bit precision on model accuracy without any retraining.

This repository includes all code needed for:
- Training FP32 ResNet-18
- Applying uniform symmetric PTQ
- Running calibration
- Evaluating accuracy across multiple bit-widths
- Comparing accuracy drop vs. FP32 baseline

## Train the FP32 ResNet-18 Baseline
On an HPC node, run:
`
sbatch train_resnet.slurm
`
produces:
`
resnet18_cifar100_trained.pth
`

## Run Post-Training Quantization (8/6/4-bit)
We implement uniform symmetric PTQ using runtime forward hooks:
- forward_pre_hook → quantize weights before each forward pass
- forward_hook → quantize activations after each layer

Run PTQ for each precision level on an HPC node:
'''
sbatch run_resnet_ptq.slurm
'''
produces:
'''
resnet18_ptq_4bit.pth
resnet18_ptq_6bit.pth
resnet18_ptq_8bit.pth
'''

## Method Summary (Uniform Symmetric PTQ)
### 1. Symmetric Uniform Quantization
- Values are mapped into integer range: q∈[−2^(b−1), 2^(b−1) - 1]
- Per-channel quantization for weights
- Per-tensor quantization for activations

### 2. Calibration (Training-Free)
- Use ~512 CIFAR-100 training samples
- Estimate activation ranges
- No retraining required
- Produces stable 8-bit and 6-bit performance

### 3. Hook-Based PTQ Implementation
- Weights quantized at runtime using forward_pre_hooks
- Activations quantized using forward_hooks
- Allows true quantization behavior without modifying the model architecture

## Requirements
- Python 3.10+
- PyTorch ≥ 2.0
- torchvision, timm, matplotlib
- CUDA GPU or HPC node recommended
