#!/bin/bash
#SBATCH --job-name=eval_soft_routing_s6
#SBATCH --partition=DGXA100
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
echo "========== Eval Soft Routing Fusion (epoch 200) =========="

python -m cifar10.eval.soft_routing \
  --checkpoint ../workdir/cifar10_soft_routing_a4b20t4_s6/fusion-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_soft_routing_a4b20t4_s6_epoch200.log

echo "End time: $(date)"
