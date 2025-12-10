#!/bin/bash
#SBATCH --job-name=energy_quant
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=24:00:00
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=============================================="
echo "Energy-Aware Quantization Experiment"
echo "=============================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Load required modules (GCCcore provides libffi.so.8)
module load GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

# Activate your virtual environment
source ~/myenv/bin/activate

# Change to the experiment directory
cd ~/final/Mixed-precision-and-energy-modeling
echo "Working directory: $(pwd)"

echo "Using Python environment:"
which python
python --version
echo ""

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Verify key packages
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
echo ""

# Verify Salma's ResNet checkpoint exists (in ~/final/Resnet_Cifar100_PTQ/)
RESNET_CKPT="$HOME/final/Resnet_Cifar100_PTQ/checkpoints/resnet18_cifar100_trained.pth"
if [ -f "$RESNET_CKPT" ]; then
    echo "✓ Found Salma's ResNet checkpoint: $RESNET_CKPT"
else
    echo "⚠ WARNING: Salma's ResNet checkpoint NOT found at: $RESNET_CKPT"
    echo "  ResNet experiment will use random weights!"
fi
echo ""

# Run the experiments
echo ""
echo "Running Energy-Aware Quantization Experiments..."
echo "=============================================="

# Run ResNet-18 experiment
echo ""
echo "[1/2] Running ResNet-18 on CIFAR-100..."
python run_experiment.py --model resnet --device cuda

# Run DeiT-Tiny experiment
echo ""
echo "[2/2] Running DeiT-Tiny on ImageNet..."
python run_experiment.py --model deit --device cuda

echo ""
echo "=============================================="
echo "Job finished at: $(date)"
echo "=============================================="
echo ""
echo "Output files:"
ls -la results/resnet/ 2>/dev/null || echo "ResNet results not found"
echo ""
ls -la results/deit/ 2>/dev/null || echo "DeiT results not found"
