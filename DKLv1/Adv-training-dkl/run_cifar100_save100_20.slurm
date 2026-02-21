#!/bin/bash
#SBATCH --job-name=dkl_cifar100_s6
#SBATCH --partition=DGXH200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

echo "========== JOB START =========="
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

source ~/.bashrc
conda activate base

PROJECT_DIR=/home/tong.li003/DKL/DKLv1/Adv-training-dkl
cd $PROJECT_DIR

echo "Now in: $(pwd)"

export PYTHONPATH=$(pwd):$PYTHONPATH

python train_dkl_cifar100_save100_20.py \
  --arch WideResNet34_10 \
  --data CIFAR100 \
  --train_budget 'high' \
  --mark cifar100_dkl_a5b20t4_s6 \
  --epsilon 8 \
  --lr 0.2 \
  --beta 20.0 \
  --alpha 5.0 \
  --T 4.0 \
  --epochs 200 \
  --save-start 100 \
  --save-freq 20 \
  --seed 0

echo "End time: $(date)"
