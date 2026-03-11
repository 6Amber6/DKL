#!/bin/bash
#SBATCH --job-name=eval_gated_s6
#SBATCH --partition=DGXH200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
set -o pipefail

INIT_DIR=$(pwd)
mkdir -p $INIT_DIR/logs

echo "========== JOB START =========="
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

source ~/.bashrc
conda activate base

PROJECT_ROOT=/home/tong.li003/DKL
ADV_DIR=$PROJECT_ROOT/DKLv1/Adv-training-dkl
cd $ADV_DIR/auto_attacks
mkdir -p logs

export PYTHONPATH=$PROJECT_ROOT:$ADV_DIR:$PYTHONPATH

echo "Now in: $(pwd)"
echo "========== Eval Gated Fusion (epoch 200) =========="

python -m cifar10.eval.gated \
  --checkpoint ../workdir/cifar10_gated_a4b20t4_s6/fusion-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_gated_a4b20t4_s6_epoch200.log

echo "End time: $(date)"
