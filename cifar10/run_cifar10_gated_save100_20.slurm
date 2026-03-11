#!/bin/bash
#SBATCH --job-name=gated_cifar10_s6
#SBATCH --partition=DGXA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
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
cd $PROJECT_ROOT

echo "Now in: $(pwd)"

export PYTHONPATH=$PROJECT_ROOT:$ADV_DIR:$PYTHONPATH

python cifar10/train/gated.py \
  --epochs-sub 100 \
  --epochs-fusion 200 \
  --batch-size 128 \
  --lr 0.1 \
  --epsilon 0.031372549 \
  --alpha 4.0 \
  --beta 20.0 \
  --T 4.0 \
  --train_budget high \
  --model-dir $ADV_DIR/workdir \
  --mark cifar10_gated_a4b20t4_s6 \
  --data-path $ADV_DIR/../data \
  --awp-gamma 0.005 \
  --awp-warmup 10 \
  --save-start 100 \
  --save-freq 20 \
  --seed 0

echo "End time: $(date)"
