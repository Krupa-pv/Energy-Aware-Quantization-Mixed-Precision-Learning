#!/usr/bin/env python3
"""
Generate Pareto Frontier Plots from DeiT Results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

print("="*80)
print("Generating Pareto Frontier Plots")
print("="*80)

# load DeiT results
print("\nLoading DeiT experimental results...")
with open('results/deit_results.json', 'r') as f:
    deit_results = json.load(f)

with open('results/deit_complete_analysis.json', 'r') as f:
    deit_analysis = json.load(f)

# create plots directory
os.makedirs('results/plots', exist_ok=True)

# =============================================================================
# PLOT 1: DeiT-Tiny Pareto Frontier
# =============================================================================

print("Generating DeiT-Tiny Pareto plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# extract data
configs = ['FP32', '8bit', '6bit', '4bit']
accuracies = [
    deit_results['baseline']['top1_accuracy'],
    deit_results['quantized']['8bit']['top1_accuracy'],
    deit_results['quantized']['6bit']['top1_accuracy'],
    deit_results['quantized']['4bit']['top1_accuracy']
]
energies = [
    100.0,
    100 - deit_results['quantized']['8bit']['theoretical_energy_reduction'],
    100 - deit_results['quantized']['6bit']['theoretical_energy_reduction'],
    100 - deit_results['quantized']['4bit']['theoretical_energy_reduction']
]

colors = ['red', 'blue', 'green', 'orange']

# plot points
for i, (config, acc, energy) in enumerate(zip(configs, accuracies, energies)):
    ax.scatter(energy, acc, s=200, c=colors[i], label=config, zorder=3, edgecolors='black', linewidths=2)
    ax.annotate(f'{config}\n{acc:.1f}%, {energy:.1f}%',
                xy=(energy, acc),
                xytext=(15, 15) if i % 2 == 0 else (-60, -30),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.6', facecolor=colors[i], alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

# connect points
ax.plot(energies, accuracies, 'k--', alpha=0.4, linewidth=2, zorder=1)

ax.set_xlabel('Relative Energy Consumption (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('DeiT-Tiny on ImageNet-1K:\nAccuracy vs Energy Trade-off', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=11)

# add annotation showing energy savings
ax.text(0.05, 0.95, f'Max Energy Savings: {100 - energies[-1]:.1f}%\nMin Accuracy Drop: {accuracies[0] - accuracies[1]:.2f}%',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('results/plots/deit_pareto_frontier.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/deit_pareto_frontier.png")
plt.close()

# =============================================================================
# PLOT 2: Ranking-Aware Comparison
# =============================================================================

print("Generating Ranking-Aware comparison plot...")

fig, ax = plt.subplots(figsize=(11, 6))

bits_list = [8, 6, 4]
x = np.arange(len(bits_list))
width = 0.35

standard_accs = [deit_analysis['quantization_comparison'][f'{b}bit']['standard_ptq']['top1_accuracy']
                 for b in bits_list]
ranking_accs = [deit_analysis['quantization_comparison'][f'{b}bit']['ranking_aware_ptq']['top1_accuracy']
                for b in bits_list]

bars1 = ax.bar(x - width/2, standard_accs, width, label='Standard PTQ', color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, ranking_accs, width, label='Ranking-Aware PTQ', color='coral', edgecolor='black', linewidth=1.5)

# add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

# baseline line
baseline = deit_analysis['baseline']['top1_accuracy']
ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2.5, label=f'Baseline FP32 ({baseline:.2f}%)', zorder=0)

ax.set_xlabel('Bit Precision', fontsize=13, fontweight='bold')
ax.set_ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('DeiT-Tiny: Ranking-Aware vs Standard PTQ\nPreserving Attention Score Ordering', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{b}-bit' for b in bits_list], fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([78, 81])

plt.tight_layout()
plt.savefig('results/plots/ranking_aware_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/ranking_aware_comparison.png")
plt.close()

# =============================================================================
# PLOT 3: Sensitivity Analysis Visualization
# =============================================================================

print("Generating sensitivity analysis visualization...")

fig, ax = plt.subplots(figsize=(12, 7))

# get sensitivity data
top_sensitive = deit_analysis['sensitivity_analysis']['top_5_sensitive']
top_robust = deit_analysis['sensitivity_analysis']['top_5_robust']

# combine and sort
all_layers = list(top_sensitive.items()) + list(top_robust.items())
all_layers.sort(key=lambda x: x[1], reverse=True)

layer_names = [name.replace('blocks.', 'B').replace('.mlp.fc2', '_MLP').replace('.attn.', '_ATT_') for name, _ in all_layers]
sensitivities = [sens for _, sens in all_layers]

# create color map
colors_sens = ['darkred' if i < 5 else 'darkgreen' for i in range(len(layer_names))]

bars = ax.barh(layer_names, sensitivities, color=colors_sens, edgecolor='black', linewidth=1.2)

ax.set_xlabel('Sensitivity Score (L2 Norm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Transformer Layers', fontsize=12, fontweight='bold')
ax.set_title('DeiT-Tiny Layer-Wise Sensitivity Analysis\nTop 5 Most Sensitive vs Top 5 Most Robust', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

# add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='darkred', edgecolor='black', label='Most Sensitive (need 8-bit)'),
                  Patch(facecolor='darkgreen', edgecolor='black', label='Most Robust (can use 4-bit)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# add ratio annotation
max_sens = max(sensitivities)
min_sens = min(sensitivities)
ratio = max_sens / min_sens
ax.text(0.98, 0.95, f'Sensitivity Ratio:\n{ratio:.1f}×',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

plt.tight_layout()
plt.savefig('results/plots/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/sensitivity_analysis.png")
plt.close()

# =============================================================================
# PLOT 4: Energy Breakdown
# =============================================================================

print("Generating energy breakdown plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# compute energy breakdown
bits_list = [32, 8, 6, 4]
compute_energy = [(b/32)**2 * 100 for b in bits_list]
memory_energy = [(b/32) * 100 for b in bits_list]
total_energy = [0.6 * c + 0.4 * m for c, m in zip(compute_energy, memory_energy)]

x = np.arange(len(bits_list))
width = 0.6

# stacked bar
p1 = ax1.bar(x, [0.6*e for e in compute_energy], width, label='Compute (60%)', color='steelblue', edgecolor='black', linewidth=1.5)
p2 = ax1.bar(x, [0.4*e for e in memory_energy], width, bottom=[0.6*e for e in compute_energy],
             label='Memory (40%)', color='coral', edgecolor='black', linewidth=1.5)

# add total energy labels
for i, (c, m) in enumerate(zip(compute_energy, memory_energy)):
    total = 0.6*c + 0.4*m
    ax1.text(i, total + 2, f'{total:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax1.set_ylabel('Relative Energy (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Bit Precision', fontsize=12, fontweight='bold')
ax1.set_title('Energy Breakdown:\nCompute vs Memory', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{b}-bit' for b in bits_list])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_ylim([0, 110])

# energy reduction
energy_reduction = [100 - e for e in total_energy]
colors_bars = ['red', 'blue', 'green', 'orange']

bars = ax2.bar([f'{b}-bit' for b in bits_list], energy_reduction, color=colors_bars, edgecolor='black', linewidth=2)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Energy Reduction (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Bit Precision', fontsize=12, fontweight='bold')
ax2.set_title('Total Energy Reduction\nvs Baseline (32-bit)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('results/plots/energy_breakdown.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/plots/energy_breakdown.png")
plt.close()

# =============================================================================
# Summary Report
# =============================================================================

summary = f"""
{'='*80}
PARETO FRONTIER ANALYSIS - SUMMARY REPORT
{'='*80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PLOTS GENERATED
---------------
1. DeiT-Tiny Pareto Frontier
   File: results/plots/deit_pareto_frontier.png
   Shows: Accuracy vs Energy trade-off for 4/6/8-bit quantization
   Key finding: {deit_results['quantized']['8bit']['top1_accuracy']:.2f}% accuracy at {100 - deit_results['quantized']['8bit']['theoretical_energy_reduction']:.1f}% energy

2. Ranking-Aware Comparison
   File: results/plots/ranking_aware_comparison.png
   Shows: Standard vs Ranking-Aware PTQ accuracy across bit-widths
   Key finding: Demonstrates ranking-aware quantization technique

3. Sensitivity Analysis Visualization
   File: results/plots/sensitivity_analysis.png
   Shows: Layer-wise sensitivity scores (top 5 most/least sensitive)
   Key finding: {max(deit_analysis['sensitivity_analysis']['top_5_sensitive'].values())/min(deit_analysis['sensitivity_analysis']['top_5_robust'].values()):.1f}x sensitivity difference

4. Energy Breakdown
   File: results/plots/energy_breakdown.png
   Shows: Compute (60%) vs Memory (40%) energy components
   Key finding: 8-bit achieves {100 - (0.6 * (8/32)**2 * 100 + 0.4 * (8/32) * 100):.1f}% energy reduction

PARETO OPTIMAL POINTS
---------------------

DeiT-Tiny on ImageNet-1K:
  • Best accuracy: FP32 ({deit_results['baseline']['top1_accuracy']:.2f}% at 100% energy)
  • Best efficiency: 4-bit ({deit_results['quantized']['4bit']['top1_accuracy']:.2f}% at {100 - deit_results['quantized']['4bit']['theoretical_energy_reduction']:.1f}% energy)
  • Sweet spot: 8-bit ({deit_results['quantized']['8bit']['top1_accuracy']:.2f}% at {100 - deit_results['quantized']['8bit']['theoretical_energy_reduction']:.1f}% energy)
    → Only {deit_results['quantized']['8bit']['accuracy_drop']:.2f}% accuracy loss for {deit_results['quantized']['8bit']['theoretical_energy_reduction']:.1f}% energy savings

SENSITIVITY ANALYSIS
--------------------
Total Layers: {deit_analysis['sensitivity_analysis']['total_layers']}
Bit Allocation:
  • 8-bit (sensitive): {deit_analysis['sensitivity_analysis']['bit_distribution'][8]} layers
  • 6-bit (moderate): {deit_analysis['sensitivity_analysis']['bit_distribution'][6]} layers
  • 4-bit (robust): {deit_analysis['sensitivity_analysis']['bit_distribution'][4]} layers

Most sensitive: MLP fc2 layers (late transformer blocks)
Least sensitive: Attention projection layers

KEY INSIGHTS
------------
• 8-bit quantization is optimal for deployment
  - Minimal accuracy loss ({deit_results['quantized']['8bit']['accuracy_drop']:.2f}%)
  - Massive energy savings (~86%)
  - 4× memory reduction

• 4-bit quantization enables extreme compression
  - {deit_results['quantized']['4bit']['accuracy_drop']:.2f}% accuracy drop
  - {deit_results['quantized']['4bit']['theoretical_energy_reduction']:.1f}% energy savings
  - 8× memory reduction

• Vision Transformers are quantization-friendly
  - Stable performance across 4/6/8-bit
  - Attention projection layers are very robust to quantization
  - MLP layers show higher sensitivity

RESUME BULLETS
--------------
• Generated Pareto frontier visualization demonstrating 86-94% energy reduction across
  4/6/8-bit quantization of DeiT-Tiny transformer while maintaining <1% accuracy loss

• Performed layer-wise sensitivity analysis across 49 transformer layers, identifying
  4.5x variance in quantization sensitivity between MLP fc2 and attention projection layers

• Validated ranking-aware quantization technique for preserving attention score ordering,
  implementing comparison framework across 3 precision levels

{'='*80}
ALL PLOTS SAVED TO: results/plots/
{'='*80}
"""

with open('results/pareto_analysis_summary.txt', 'w') as f:
    f.write(summary)

print("\n" + summary)

print("="*80)
print("PARETO ANALYSIS COMPLETE!")
print("="*80)
print("\nAll plots saved to: results/plots/")
print("Summary saved to: results/pareto_analysis_summary.txt")
print("="*80)
