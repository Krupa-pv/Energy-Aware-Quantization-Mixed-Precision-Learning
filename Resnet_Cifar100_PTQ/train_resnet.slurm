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

module load Python/3.11.3-GCCcore-12.3.0

source ~/quant_env_311/bin/activate

python train_resnet18_cifar100.py
