"""
Visualization Module
====================
Generate plots for energy-accuracy trade-offs, Pareto frontiers,
and sensitivity analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color scheme
COLORS = {
    'fp32': '#2E86AB',      # Blue
    '8bit': '#A23B72',      # Magenta
    '6bit': '#F18F01',      # Orange
    '4bit': '#C73E1D',      # Red
    'mixed': '#3A7D44',     # Green
    'pareto': '#2E86AB',    # Blue for Pareto line
}


def plot_pareto_frontier(
    results: Dict[str, Dict[str, float]],
    title: str = "Energy-Accuracy Pareto Frontier",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7)
) -> plt.Figure:
    """
    Plot Pareto frontier of energy vs accuracy.
    
    Args:
        results: Dictionary with structure:
                {config_name: {'accuracy': float, 'energy': float}}
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    configs = []
    accuracies = []
    energies = []
    
    for name, data in results.items():
        configs.append(name)
        accuracies.append(data['accuracy'])
        energies.append(data['energy'])
    
    # Normalize energy for better visualization
    max_energy = max(energies)
    energies_norm = [e / max_energy for e in energies]
    
    # Plot points
    for i, (config, acc, energy) in enumerate(zip(configs, accuracies, energies_norm)):
        color = COLORS.get(config.lower().replace('-', '').replace(' ', ''), '#666666')
        ax.scatter(energy, acc, s=200, c=color, label=config, zorder=5, edgecolors='white', linewidth=2)
        
        # Add label
        offset = (0.02, 0.01)
        ax.annotate(config, (energy + offset[0], acc + offset[1]), fontsize=10, fontweight='bold')
    
    # Find and plot Pareto frontier
    points = list(zip(energies_norm, accuracies))
    pareto_points = _find_pareto_frontier(points)
    
    if len(pareto_points) > 1:
        pareto_points_sorted = sorted(pareto_points, key=lambda x: x[0])
        px, py = zip(*pareto_points_sorted)
        ax.plot(px, py, '--', color=COLORS['pareto'], linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    # Styling
    ax.set_xlabel('Relative Energy (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc='lower right', fontsize=10)
    
    # Add annotation for optimal region
    ax.annotate('Optimal Region\n(Low Energy, High Accuracy)', 
                xy=(0.1, max(accuracies) * 0.95),
                fontsize=9, fontstyle='italic', color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Pareto plot to {save_path}")
    
    return fig


def _find_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Find Pareto-optimal points (minimize x, maximize y).
    """
    pareto = []
    for p in points:
        is_dominated = False
        for q in points:
            # q dominates p if q has lower/equal energy AND higher/equal accuracy
            # with at least one strict inequality
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                is_dominated = True
                break
        if not is_dominated:
            pareto.append(p)
    return pareto


def plot_sensitivity_distribution(
    sensitivities: Dict[str, float],
    title: str = "Layer Sensitivity Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot distribution of layer sensitivities.
    
    Args:
        sensitivities: Dict mapping layer name to sensitivity value
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart of sensitivities
    ax1 = axes[0]
    names = list(sensitivities.keys())
    values = list(sensitivities.values())
    
    # Shorten names for display
    short_names = [n.split('.')[-1] if '.' in n else n for n in names]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
    sorted_indices = np.argsort(values)[::-1]
    
    y_pos = np.arange(len(names))
    ax1.barh(y_pos, [values[i] for i in sorted_indices], color=[colors[i] for i in sorted_indices])
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([short_names[i] for i in sorted_indices], fontsize=8)
    ax1.set_xlabel('Sensitivity (L2 norm)')
    ax1.set_title('Layer Sensitivity Ranking')
    ax1.invert_yaxis()
    
    # Histogram
    ax2 = axes[1]
    ax2.hist(values, bins=20, color=COLORS['pareto'], edgecolor='white', alpha=0.7)
    ax2.axvline(np.percentile(values, 25), color=COLORS['4bit'], linestyle='--', label='25th percentile (4-bit)')
    ax2.axvline(np.percentile(values, 75), color=COLORS['8bit'], linestyle='--', label='75th percentile (8-bit)')
    ax2.set_xlabel('Sensitivity')
    ax2.set_ylabel('Count')
    ax2.set_title('Sensitivity Distribution')
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_bitwidth_distribution(
    assignments: Dict[str, int],
    title: str = "Mixed-Precision Bitwidth Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot distribution of assigned bitwidths.
    
    Args:
        assignments: Dict mapping layer name to bitwidth
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count bitwidths
    counts = {4: 0, 6: 0, 8: 0}
    for bits in assignments.values():
        if bits in counts:
            counts[bits] += 1
    
    # Pie chart
    labels = [f'{b}-bit\n({counts[b]} layers)' for b in [4, 6, 8]]
    sizes = [counts[4], counts[6], counts[8]]
    colors = [COLORS['4bit'], COLORS['6bit'], COLORS['8bit']]
    explode = (0.05, 0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        colors=colors,
        explode=explode,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend with energy implications
    legend_labels = [
        '4-bit: Max compression, lowest energy',
        '6-bit: Balanced',
        '8-bit: Best accuracy, higher energy'
    ]
    ax.legend(wedges, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.15))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_energy_breakdown(
    energy_results: Dict[str, Dict],
    title: str = "Energy Consumption Breakdown",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot energy breakdown by configuration.
    
    Args:
        energy_results: Dict with energy data per configuration
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    configs = list(energy_results.keys())
    mac_energies = [energy_results[c].get('mac_energy', 0) for c in configs]
    mem_energies = [energy_results[c].get('memory_energy', 0) for c in configs]
    
    # Normalize
    max_total = max(m + e for m, e in zip(mac_energies, mem_energies))
    mac_norm = [m / max_total for m in mac_energies]
    mem_norm = [m / max_total for m in mem_energies]
    
    x = np.arange(len(configs))
    width = 0.6
    
    bars1 = ax.bar(x, mac_norm, width, label='MAC Energy', color=COLORS['8bit'])
    bars2 = ax.bar(x, mem_norm, width, bottom=mac_norm, label='Memory Energy', color=COLORS['6bit'])
    
    ax.set_ylabel('Relative Energy')
    ax.set_xlabel('Configuration')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    
    # Add percentage labels on bars
    for i, (m, e) in enumerate(zip(mac_norm, mem_norm)):
        total = m + e
        ax.annotate(f'{total:.0%}', (i, total + 0.02), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_accuracy_vs_bitwidth(
    results: Dict[str, Tuple[int, float]],
    title: str = "Accuracy vs Average Bitwidth",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot accuracy against average bitwidth.
    
    Args:
        results: Dict mapping config name to (avg_bitwidth, accuracy)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, (bitwidth, accuracy) in results.items():
        color = COLORS.get(name.lower().replace('-', '').replace(' ', ''), '#666666')
        ax.scatter(bitwidth, accuracy, s=150, c=color, label=name, zorder=5)
        ax.annotate(name, (bitwidth + 0.1, accuracy), fontsize=9)
    
    ax.set_xlabel('Average Bitwidth', fontsize=12)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_full_report_figure(
    pareto_data: Dict[str, Dict[str, float]],
    sensitivities: Dict[str, float],
    bitwidth_assignments: Dict[str, int],
    energy_breakdown: Dict[str, Dict],
    model_name: str = "Model",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive 4-panel report figure.
    
    Args:
        pareto_data: Data for Pareto plot
        sensitivities: Layer sensitivities
        bitwidth_assignments: Assigned bitwidths
        energy_breakdown: Energy data per config
        model_name: Name of the model
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Panel 1: Pareto Frontier
    ax1 = fig.add_subplot(2, 2, 1)
    _plot_pareto_on_axis(ax1, pareto_data)
    ax1.set_title('Energy-Accuracy Pareto Frontier', fontweight='bold')
    
    # Panel 2: Sensitivity Distribution
    ax2 = fig.add_subplot(2, 2, 2)
    values = list(sensitivities.values())
    ax2.hist(values, bins=15, color=COLORS['pareto'], edgecolor='white', alpha=0.7)
    ax2.axvline(np.percentile(values, 25), color=COLORS['4bit'], linestyle='--', label='4-bit threshold')
    ax2.axvline(np.percentile(values, 75), color=COLORS['8bit'], linestyle='--', label='8-bit threshold')
    ax2.set_xlabel('Sensitivity')
    ax2.set_ylabel('Layer Count')
    ax2.set_title('Sensitivity Distribution', fontweight='bold')
    ax2.legend()
    
    # Panel 3: Bitwidth Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    counts = {4: 0, 6: 0, 8: 0}
    for bits in bitwidth_assignments.values():
        counts[bits] = counts.get(bits, 0) + 1
    ax3.bar(['4-bit', '6-bit', '8-bit'], [counts[4], counts[6], counts[8]], 
            color=[COLORS['4bit'], COLORS['6bit'], COLORS['8bit']])
    ax3.set_ylabel('Number of Layers')
    ax3.set_title('Mixed-Precision Assignment', fontweight='bold')
    
    # Panel 4: Energy Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    configs = list(energy_breakdown.keys())
    totals = [energy_breakdown[c].get('total_energy', 0) for c in configs]
    max_total = max(totals) if totals else 1
    totals_norm = [t / max_total for t in totals]
    
    bars = ax4.bar(configs, totals_norm, color=[COLORS.get(c.lower(), '#666') for c in configs])
    ax4.set_ylabel('Relative Energy')
    ax4.set_title('Energy Comparison', fontweight='bold')
    ax4.set_xticklabels(configs, rotation=45, ha='right')
    
    # Add percentage labels
    for bar, val in zip(bars, totals_norm):
        ax4.annotate(f'{val:.0%}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle(f'{model_name} - Quantization Analysis Report', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved report figure to {save_path}")
    
    return fig


def _plot_pareto_on_axis(ax, results: Dict[str, Dict[str, float]]):
    """Helper to plot Pareto on a given axis."""
    for name, data in results.items():
        color = COLORS.get(name.lower().replace('-', '').replace(' ', ''), '#666666')
        ax.scatter(data['energy'], data['accuracy'], s=100, c=color, label=name, zorder=5)
    
    ax.set_xlabel('Relative Energy')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)


