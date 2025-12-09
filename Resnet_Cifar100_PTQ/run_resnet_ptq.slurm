#!/bin/bash -l
#SBATCH -J proj_2
#SBATCH -A ecse600
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

cd /home/sxb1283/proj_2

source ~/quant_env_311/bin/activate

BITWIDTHS=("8" "6" "4")

for B in "${BITWIDTHS[@]}"; do
    echo "[Running] PTQ for ${B}-bit"
    python run_ptq_resnet18.py --bitwidth $B
done
