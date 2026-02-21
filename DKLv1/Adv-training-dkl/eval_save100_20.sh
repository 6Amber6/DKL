#!/bin/bash
# 评估 save100_20 训练的 CIFAR-10 / CIFAR-100（epoch 200 checkpoint）
# 在 Adv-training-dkl 目录下执行: bash eval_save100_20.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR/auto_attacks"

mkdir -p logs

echo "========== Eval CIFAR-10 (epoch 200) =========="
python eval.py --arch WideResNet34 \
  --checkpoint ../workdir/cifar10_dkl_a4b20t4_s6/ours-model-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_dkl_a4b20t4_s6_epoch200.log

echo "========== Eval CIFAR-100 (epoch 200) =========="
python eval.py --arch WideResNet34 \
  --checkpoint ../workdir/cifar100_dkl_a5b20t4_s6/ours-model-epoch200.pt \
  --data CIFAR100 \
  --preprocess '01' \
  --log_path logs/cifar100_dkl_a5b20t4_s6_epoch200.log

echo "========== Done =========="
