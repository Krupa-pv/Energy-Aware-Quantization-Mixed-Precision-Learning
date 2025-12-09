import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from tqdm import tqdm


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader,
                     device: str = 'cuda', verbose: bool = True) -> Tuple[float, float]:
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        verbose: Whether to show progress bar

    Returns:
        top1_accuracy: Top-1 accuracy percentage
        top5_accuracy: Top-5 accuracy percentage
    """
    model.eval()
    model.to(device)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        iterator = tqdm(data_loader, desc="Evaluating") if verbose else data_loader

        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch
            else:
                inputs, targets = batch, None

            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            else:
                continue

            outputs = model(inputs)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_top1 += predicted.eq(targets).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 += sum([targets[i] in top5_pred[i] for i in range(len(targets))])

    top1_accuracy = 100. * correct_top1 / total
    top5_accuracy = 100. * correct_top5 / total

    if verbose:
        print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")

    return top1_accuracy, top5_accuracy


def evaluate_model_stats(model: nn.Module) -> Dict[str, any]:
    """
    Compute model statistics (parameters, size, etc.).

    Args:
        model: Model to analyze

    Returns:
        stats: Dictionary of model statistics
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        params_count = param.numel()
        total_params += params_count
        if param.requires_grad:
            trainable_params += params_count

    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
    }

    return stats


def print_model_stats(model: nn.Module, model_name: str = "Model"):
    """
    Print formatted model statistics.

    Args:
        model: Model to analyze
        model_name: Name of the model for display
    """
    stats = evaluate_model_stats(model)

    print(f"\n{'='*50}")
    print(f"{model_name} Statistics")
    print(f"{'='*50}")
    print(f"Total Parameters:      {stats['total_params']:,} ({stats['total_params_M']:.2f}M)")
    print(f"Trainable Parameters:  {stats['trainable_params']:,} ({stats['trainable_params_M']:.2f}M)")
    print(f"{'='*50}\n")


def compare_models(baseline_acc: float, quantized_acc: float,
                  baseline_energy: float, quantized_energy: float) -> Dict[str, float]:
    """
    Compare baseline and quantized models.

    Args:
        baseline_acc: Baseline model accuracy
        quantized_acc: Quantized model accuracy
        baseline_energy: Baseline model energy
        quantized_energy: Quantized model energy

    Returns:
        comparison: Dictionary with comparison metrics
    """
    acc_drop = baseline_acc - quantized_acc
    acc_retention = (quantized_acc / baseline_acc) * 100 if baseline_acc > 0 else 0

    energy_reduction = ((baseline_energy - quantized_energy) / baseline_energy) * 100 if baseline_energy > 0 else 0
    energy_ratio = quantized_energy / baseline_energy if baseline_energy > 0 else 1.0

    comparison = {
        'accuracy_drop': acc_drop,
        'accuracy_retention': acc_retention,
        'energy_reduction_percentage': energy_reduction,
        'energy_ratio': energy_ratio,
    }

    return comparison


def print_comparison(baseline_acc: float, quantized_acc: float,
                    baseline_energy: float, quantized_energy: float,
                    config_name: str = "Configuration"):
    """
    Print formatted comparison between baseline and quantized models.

    Args:
        baseline_acc: Baseline accuracy
        quantized_acc: Quantized accuracy
        baseline_energy: Baseline energy
        quantized_energy: Quantized energy
        config_name: Name of quantization configuration
    """
    comparison = compare_models(baseline_acc, quantized_acc, baseline_energy, quantized_energy)

    print(f"\n{'='*60}")
    print(f"Comparison: {config_name}")
    print(f"{'='*60}")
    print(f"Accuracy:")
    print(f"  Baseline:          {baseline_acc:.2f}%")
    print(f"  Quantized:         {quantized_acc:.2f}%")
    print(f"  Drop:              {comparison['accuracy_drop']:.2f}%")
    print(f"  Retention:         {comparison['accuracy_retention']:.1f}%")
    print(f"\nEnergy:")
    print(f"  Baseline:          {baseline_energy:.2e}")
    print(f"  Quantized:         {quantized_energy:.2e}")
    print(f"  Reduction:         {comparison['energy_reduction_percentage']:.1f}%")
    print(f"  Ratio:             {comparison['energy_ratio']:.3f}x")
    print(f"{'='*60}\n")
