# Energy-Aware-Quantization-Mixed-Precision-Learning

## Team
- **Salma Bhar (sxb1283)** – Primary Contact  
- **Wiam Skakri (wxs428)**  
- **Krupa Venkatesan (kxv178)**  

## Overview
Deep learning models—especially Vision Transformers—continue to grow in size, increasing compute cost, memory bandwidth, latency, and total energy consumption:

E = N_MAC * E_MAC + N_mem * E_DRAM

This project explores **post-training quantization (PTQ)** and **mixed-precision inference** as lightweight, training-free methods for improving energy efficiency while preserving accuracy.

Models studied:
- **ResNet-18** (CIFAR-100)  
- **DeiT-Tiny** (ImageNet, PTQ only)

---

## Background

### Post-Training Quantization for ViTs (Liu et al., 2021)
- Introduces **ranking loss** to preserve attention structure.
- Proposes **mixed-precision** using nuclear-norm sensitivity.
- Highly relevant to DeiT-Tiny quantization.

### Energy Awareness in Low Precision Networks (Eliezer et al., 2023)
- Shows benefits of **unsigned arithmetic** for power reduction.
- Observes multiplier power dominated by input bitwidth.  
- Introduces **PANN**, enabling controllable energy–accuracy trade-offs.

---

## Technical Approach

### 1. PTQ Implementation
- Per-tensor and per-channel quantization  
- Symmetric uniform quantization  
- Calibration using a small dataset subset  
- Evaluation at **8-bit, 6-bit, 4-bit**

**For ViTs:**  
Ranking-aware quantization to preserve ordering in attention scores.

---

### 2. Mixed-Precision Quantization (Sensitivity-Based)

Layer sensitivity:

S_l = || y_full - y_quant ||_2

Bitwidth assignment:
- **High sensitivity → 8-bit**
- **Moderate sensitivity → 6-bit**
- **Low sensitivity → 4-bit**

Requires a single calibration pass.

---

### 3. Tools & Datasets
- PyTorch  
- timm (DeiT-Tiny pretrained weights)  
- CIFAR-100  
- ImageNet (500–1000 calibration samples)  
- NumPy + Matplotlib for energy modeling + Pareto curves  

---

## Efficiency Metrics

### Accuracy
- CIFAR-100 (ResNet-18)  
- ImageNet validation (DeiT-Tiny PTQ)  

### Energy Modeling

E = Σ_l [ N_MAC(l) * E_MAC(b_l) ] + Σ_l [ N_mem(l) * E_DRAM(b_l) ]

where:
E_MAC(b) ∝ b^2
E_DRAM(b) ∝ b

### Pareto Frontier
We will compare:
- FP32  
- 8-bit PTQ  
- 6-bit PTQ  
- 4-bit PTQ  
- Mixed precision  

---

## Expected Outcomes
- ResNet-18 quantizes cleanly to 8-bit with minimal accuracy loss.  
- DeiT-Tiny requires ranking-aware PTQ for stability.  
- Mixed precision reduces total estimated energy.  
- Pareto curves show accuracy–energy trade-offs.

---

## Evaluation Plan
We compare:
- Per-tensor vs per-channel  
- With vs without ranking-aware ViT quantization  
- Fixed vs mixed precision  

We report:
- Accuracy drop  
- Estimated energy reduction  
- Final Pareto plots  

---

## Timeline

| Day | Task | Member |
|-----|------|--------|
| 1 | Implement PTQ for ResNet-18 | Salma |
| 2 | Calibrate + evaluate ResNet-18 | Salma |
| 3 | DeiT-Tiny PTQ + ranking module | Krupa |
| 4 | Calibration + sensitivity analysis | Krupa |
| 5 | Mixed-precision + energy model | Wiam |
| 6 | Plots + Pareto results | Wiam |
| 7 | Report writing + cleanup | Team |

---

## References
1. **Post-Training Quantization for Vision Transformer**, Liu et al., ICML 2021 — https://arxiv.org/abs/2106.14156  
2. **Energy Awareness in Low Precision Neural Networks**, Eliezer et al., ICLR 2023 — https://arxiv.org/abs/2202.02783  
