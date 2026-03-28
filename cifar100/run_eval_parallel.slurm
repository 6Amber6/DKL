#!/bin/bash
#SBATCH --job-name=eval_cifar100_parallel_s6
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
export PYTHONUNBUFFERED=1

echo "Now in: $(pwd)"
echo "========== Eval Parallel Fusion CIFAR-100 (best) =========="

stdbuf -oL python -u -m cifar100.eval.parallel \
  --checkpoint ../workdir/cifar100_parallel_a4b20t4_s6/fusion-best.pt \
  --data-dir $ADV_DIR/../data \
  --preprocess '01' \
  --sub-depth 34 \
  --sub-widen 10 \
  --log_path logs/cifar100_parallel_a4b20t4_s6_best.log

echo "========== Eval Parallel Fusion CIFAR-100 (last) =========="

stdbuf -oL python -u -m cifar100.eval.parallel \
  --checkpoint ../workdir/cifar100_parallel_a4b20t4_s6/fusion-last.pt \
  --data-dir $ADV_DIR/../data \
  --preprocess '01' \
  --sub-depth 34 \
  --sub-widen 10 \
  --log_path logs/cifar100_parallel_a4b20t4_s6_last.log

echo "End time: $(date)"
