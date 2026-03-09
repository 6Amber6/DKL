#!/bin/bash
#SBATCH --job-name=eval_cifar10_swa
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

echo "========== JOB START =========="
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"

source ~/.bashrc
conda activate base

PROJECT_DIR=/home/tong.li003/DKL/DKLv1/Adv-training-dkl
cd $PROJECT_DIR/auto_attacks
export PYTHONPATH=$PROJECT_DIR/auto_attacks:$PROJECT_DIR:$PYTHONPATH

echo "Now in: $(pwd)"

python eval.py --arch WideResNet34 \
  --checkpoint ../workdir/cifar10_dkl_a4b20t4/ours-model-epoch-SWA0.91200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path "logs/cifar10_dkl_a4b20t4_swa.log"

echo "End time: $(date)"
