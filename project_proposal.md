# Project Proposal: Energy-Aware Quantization and Mixed-Precision Learning

## Title and Team Information

**Title:** Energy-Aware Quantization and Mixed-Precision Learning

**Team:** Salma Bhar (sxb1283), Wiam Skakri (wxs428), Krupa Venkatesan (kxv178)

**Primary contact:** Salma Bhar (sxb1283)

---

## Motivation and Problem Statement

As deep learning models keep getting bigger (especially Vision Transformers), they also get more expensive to run. This creates problems for both edge devices and large-scale servers because they need more:

- Compute cost
- Memory bandwidth
- Latency
- Total energy usage, which can be analytically modeled as:

$$E = N_{MAC} \cdot E_{MAC} + N_{mem} \cdot E_{DRAM}$$

This is particularly problematic in real-world applications like mobile vision, robotics, autonomous systems, and large-scale inference services, where energy efficiency directly impacts operating cost and feasibility.

**Goal:** To explore post-training quantization (PTQ) and mixed-precision inference as lightweight, training-free approaches for improving energy efficiency without significantly reducing model accuracy.

We will focus on:

- **ResNet-18** (CNN baseline)
- **DeiT-Tiny** (lightweight Vision Transformer)

and analyze the energy-accuracy trade-off produced by different precision assignments.

---

## Background and Related Work

### Paper 1: Post-Training Quantization for Vision Transformer [1]

**Relationship to Project:** This paper is highly relevant as it specifically addresses quantizing Vision Transformers (ViTs), a key model type required by the project. The project requires implementing PTQ schemes for ViT models.

**Key Insight 1 - Ranking Loss:** To preserve the functionality of the complex Multi-Head Self Attention (MSA) mechanism, the paper introduces a ranking loss into the conventional quantization objective. This is essential because the attention mechanism does not exist in CNNs, making conventional CNN quantization methods unsuitable for ViTs.

**Key Insight 2 - Mixed-Precision:** The work explores mixed-precision quantization by analyzing feature diversity using the nuclear norm of the attention map and output feature, assigning more bits to more sensitive layers. This directly informs the project requirement for implementing and evaluating mixed-precision schemes.

### Paper 2: Energy awareness in low precision neural networks [2]

**Relationship to Project:** This paper relates to the project's energy-aware focus and analytical modeling requirements. It aligns with the goal of analytically modeling and evaluating total energy consumption $E = N_{MAC} \cdot E_{MAC} + N_{mem} \cdot E_{DRAM}$.

**Key Insight 1 - Unsigned Arithmetic:** It identifies that a major portion of dynamic power is caused by the use of signed integers (two's complement), especially due to high bit toggling at the input of large accumulators. Converting a pre-trained network to unsigned arithmetic can cut power consumption significantly without degrading accuracy.

**Key Insight 2 - Multiplier Power and PANN:** It observes that the multiplier's power is dominated by the larger bit width of its two inputs (Observation 2). PANN (Power-Aware Neural Network) addresses this by approximating multiplication via repeated additions (leveraging the ability to quantize weights drastically while maintaining accuracy) and allows for seamless traversal of the power-accuracy trade-off by tuning the number of additions R.

---

## Technical Approach

### i) Model Architectures

We will evaluate:

- **ResNet-18** on CIFAR-100
- **DeiT-Tiny** pretrained on ImageNet (evaluated via PTQ only)

Both models are lightweight enough for rapid experimentation.

### ii) Planned Modifications and Strategies

#### A. Post-Training Quantization (PTQ)

We will implement:

- Per-tensor and per-channel quantization
- Symmetric uniform quantization for weights/activations
- Calibration using a small dataset subset
- Evaluation at 8-bit, 6-bit, and 4-bit precision

For Vision Transformers, we include:

**Ranking-Aware Quantization (Simplified from [1]):** a lightweight module that preserves relative ordering in the attention scores to reduce sensitivity during quantization.

#### B. Mixed-Precision Quantization (Sensitivity-Based)

Instead of computationally expensive nuclear norm metrics, we use:

$$S_l = ||y_l^{full} - y_l^{quant}||_2$$

Layer bitwidth assignment:

- High-sensitivity → 8 bits
- Moderate-sensitivity → 6 bits
- Low-sensitivity → 4 bits

This requires no training and can be computed in a single calibration pass.

### iii) Tools, Frameworks, and Datasets

- **PyTorch** (quantization utilities + custom modules)
- **timm** for pretrained DeiT-Tiny weights
- **CIFAR-100** for ResNet-18 experiments
- **ImageNet** (500-1000 calibration images) for DeiT-Tiny PTQ
- **NumPy + Matplotlib** for energy modeling and Pareto plots

---

## Efficiency Aspect and Metrics

### Why This Project Is Efficient

- Quantization lowers the bitwidth of weights/activations → fewer memory transfers
- Mixed precision allocates higher bitwidth only where necessary
- Energy modeling quantifies the impact on hardware efficiency

### Efficiency Metrics

We will measure:

#### A. Accuracy

Top-1 accuracy on:

- CIFAR-100 (ResNet-18)
- ImageNet validation (DeiT-Tiny, via PTQ only)

#### B. Compute and Memory Efficiency

Using the analytical energy model:

$$E = \sum_l N^{(l)}_{MAC} \cdot E_{MAC}(b_l) + \sum_l N^{(l)}_{mem} \cdot E_{DRAM}(b_l)$$

Where:

- $N_{MAC}$ is the multiply-accumulate count per layer
- $E_{MAC}(b) \propto b^2$
- $E_{DRAM}(b) \propto b$

#### C. Pareto Frontier

We will plot energy vs. accuracy for:

- Full precision
- 8-bit PTQ
- 6-bit PTQ
- 4-bit PTQ
- Mixed precision

---

## Expected Results and Evaluation Plan

### Expected Outcomes

- ResNet-18 will quantize to 8-bit with an accuracy drop
- DeiT-Tiny will quantize stably only with the ranking-aware modification
- Mixed precision will show a reduction in estimated energy
- Pareto curves will reveal optimal precision assignments

### Evaluation Criteria

We will:

- Compare quantized models against FP32 baselines
- Run ablations:
  - Per-tensor vs per-channel quantization
  - With vs without ranking-aware ViT quantization
  - Fixed vs mixed precision
- Report:
  - Accuracy drop
  - Energy reduction
  - Pareto curves

**Success** = energy reduction with minimal accuracy loss.

---

## Project Timeline and Division of Work

| Day   | Task                                              | Assigned To |
|-------|---------------------------------------------------|-------------|
| Day 1 | Implement PTQ for ResNet-18 (8/6/4-bit)           | Salma       |
| Day 2 | Calibrate and evaluate ResNet-18                  | Salma       |
| Day 3 | Implement DeiT-Tiny PTQ + ranking-aware module    | Krupa       |
| Day 4 | Calibrate DeiT-Tiny and measure sensitivity per layer | Krupa   |
| Day 5 | Implement mixed-precision assignment + energy model | Wiam      |
| Day 6 | Generate plots, Pareto curves, ablation results   | Wiam        |
| Day 7 | Final report writing, figures, cleanup            | Whole team  |

---

## References

[1] Z. Liu, Y. Wang, K. Han, S. Ma, W. Gao, "Post-Training Quantization for Vision Transformer," ICML, 2021. Available at: https://arxiv.org/abs/2106.14156

[2] N. Spingarn Eliezer, R. Banner, E. Hoffer, H. Ben-Yaakov, T. Michaeli, "Energy Awareness in Low Precision Neural Networks," ICLR, 2023. Available at: https://arxiv.org/abs/2202.02783

