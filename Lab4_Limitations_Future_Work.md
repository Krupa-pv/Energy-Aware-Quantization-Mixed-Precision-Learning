# Lab 4: Limitations and Future Work Analysis

**Project:** Energy-Aware Quantization and Mixed-Precision Learning  
**Team:** Salma Bhar (sxb1283), Wiam Skakri (wxs428), Krupa Venkatesan (kxv178)  
**Course:** ECSE 397/600 – Efficient Deep Learning  
**Date:** December 9, 2025

---

## 1. Key Limitations of Our Current Approach

### Limitation 1: Theoretical Energy Model Without Hardware Validation

**Category:** Methodological / Implementation Limitation

Our energy model uses analytical approximations based on literature:
- E_MAC(b) ∝ b² (MAC energy scales quadratically with bitwidth)
- E_DRAM(b) ∝ b (Memory access energy scales linearly)

While theoretically grounded, these coefficients are not calibrated to specific hardware platforms. The relative weighting between MAC and memory energy (we used 1:200 ratio) is an approximation that may not reflect actual behavior on GPUs, TPUs, or edge accelerators. **We report up to 88% theoretical energy savings at 4-bit, but these numbers are not validated through real hardware measurements.**

### Limitation 2: Evaluation Dataset Scale and Statistical Rigor

**Category:** Dataset / Experimental Limitation

- **ResNet-18 on CIFAR-100:** Our baseline achieves only 61.44% accuracy, which is below typical reported accuracies (~75-78%) for well-tuned ResNet-18 on CIFAR-100. This suggests suboptimal training hyperparameters or insufficient training epochs.
  
- **DeiT-Tiny on ImageNet:** We evaluated on a 10,000-image subset rather than the full 50,000-image validation set due to compute constraints. While still statistically meaningful, this limits direct comparison with published benchmarks.

- **No repeated trials:** We did not perform multiple runs with different random seeds to report confidence intervals or standard deviations on our accuracy measurements.

### Limitation 3: Limited Baseline Comparisons and Ablations

**Category:** Methodological Limitation

Our project lacks comparisons with state-of-the-art quantization methods:
- **Missing baselines:** No comparison with recent PTQ methods like GPTQ, AWQ, SmoothQuant, or OmniQuant that have shown strong results on transformers.
- **Ranking-aware quantization:** While implemented, the ranking-aware technique for ViT attention was not systematically compared against standard PTQ across all bitwidths in a controlled ablation.
- **Mixed-precision accuracy:** The mixed-precision configuration's accuracy (85.05% for DeiT) is estimated based on bitwidth distribution rather than directly measured through end-to-end inference.

### Limitation 4: Severe Accuracy Degradation at 4-bit

**Category:** Model / Implementation Limitation

At 4-bit precision:
- **ResNet-18:** Accuracy drops from 61.44% to 38.04% (−23.4 percentage points)
- **DeiT-Tiny:** Accuracy drops from 85.34% to 83.04% (−2.3 percentage points)

The severe degradation for ResNet-18 at 4-bit suggests our simple symmetric uniform quantization is insufficient for aggressive compression. More sophisticated techniques (learned step sizes, asymmetric quantization, or outlier handling) are needed.

---

## 2. Proposed Future Improvements

### For Limitation 1: Hardware-Validated Energy Measurements

| Improvement | Description |
|-------------|-------------|
| **Edge deployment** | Deploy quantized models on actual hardware (NVIDIA Jetson, Raspberry Pi, or mobile NPUs) and measure real power consumption using tools like NVIDIA's `tegrastats` or external power meters. |
| **Profiling tools** | Use PyTorch's built-in profiler or NVIDIA Nsight to measure actual inference latency and memory bandwidth utilization. |
| **Hardware-specific coefficients** | Calibrate our energy model coefficients using measured data from target platforms rather than literature approximations. |

### For Limitation 2: Expanded and Rigorous Evaluation

| Improvement | Description |
|-------------|-------------|
| **Full ImageNet validation** | Evaluate DeiT-Tiny on the complete 50K ImageNet validation set for proper benchmarking. |
| **Improved ResNet training** | Retrain ResNet-18 with better hyperparameters (learning rate scheduling, data augmentation, longer training) to achieve competitive baseline accuracy. |
| **Statistical significance** | Run experiments 3-5 times with different seeds and report mean ± standard deviation. |
| **Additional datasets** | Extend evaluation to CIFAR-10, ImageNet-100, or domain-specific datasets (medical imaging, autonomous driving). |

### For Limitation 3: Comprehensive Baseline Comparisons

| Improvement | Description |
|-------------|-------------|
| **SOTA PTQ methods** | Implement or integrate comparisons with GPTQ, AWQ, SmoothQuant, and BRECQ for fair benchmarking. |
| **Quantization-aware training (QAT)** | Compare PTQ results against QAT to quantify the accuracy gap recoverable through fine-tuning. |
| **Systematic ablations** | Conduct controlled experiments isolating the effect of: (1) per-tensor vs. per-channel quantization, (2) ranking-aware vs. standard quantization, (3) different sensitivity thresholds for mixed-precision. |
| **Direct mixed-precision evaluation** | Actually run inference with the mixed-precision model rather than estimating accuracy from bitwidth distributions. |

### For Limitation 4: Advanced Quantization Techniques

| Improvement | Description |
|-------------|-------------|
| **Learned step sizes (LSQ)** | Replace fixed quantization scales with learnable parameters optimized during calibration. |
| **Asymmetric quantization** | Use asymmetric ranges for activations (especially after ReLU) to better utilize the quantization bins. |
| **Outlier handling** | Implement techniques like SmoothQuant's activation smoothing or GPTQ's outlier-aware quantization for sensitive layers. |
| **Knowledge distillation** | Use the full-precision model as a teacher to recover accuracy in heavily quantized models. |

---

## 3. Reflection on Project Progress

Implementing the basic PTQ pipeline was **easier than expected**—PyTorch's quantization utilities and the modular design of `timm` models made weight quantization straightforward. However, **integrating ranking-aware quantization deeply into the attention mechanism** proved more challenging than anticipated, requiring careful hook placement and understanding of the transformer forward pass.

If starting again, we would **prioritize hardware deployment earlier** in the project timeline. Our theoretical energy model, while informative, lacks the validation that actual measurements would provide. We would also **invest more time in hyperparameter tuning** for the ResNet-18 baseline to ensure a competitive starting point.

A **surprising observation** was that DeiT-Tiny proved remarkably robust to quantization—maintaining 85.01% accuracy even at 6-bit—while ResNet-18 degraded severely at 4-bit. This suggests that transformer architectures may have inherent properties (perhaps the softmax normalization in attention or layer normalization) that provide quantization resilience, which warrants further investigation.

---

## Summary Table

| Limitation | Category | Proposed Improvement |
|------------|----------|---------------------|
| Theoretical energy model | Methodological | Hardware deployment + real power measurements |
| Limited evaluation scale | Dataset/Experimental | Full ImageNet + multiple seeds + better baselines |
| Missing SOTA comparisons | Methodological | Compare with GPTQ, AWQ, SmoothQuant; add ablations |
| 4-bit accuracy collapse | Model/Implementation | LSQ, asymmetric quantization, outlier handling |

---

*This document addresses the requirements of Lab 4 for ECSE 397/600: Efficient Deep Learning.*

