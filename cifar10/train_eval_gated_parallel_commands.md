# Gated / Parallel 训练与评估命令（对齐 DKL save100_20）

## 训练

### Gated Fusion
```bash
# 提交 slurm 任务（在 DKL 根目录或 cifar10 目录下）
sbatch cifar10/run_cifar10_gated_save100_20.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL
export PYTHONPATH=/home/tong.li003/DKL:/home/tong.li003/DKL/DKLv1/Adv-training-dkl:$PYTHONPATH

python cifar10/train/gated.py \
  --epochs-sub 100 \
  --epochs-fusion 200 \
  --batch-size 128 \
  --lr 0.1 \
  --alpha 4.0 \
  --beta 20.0 \
  --T 4.0 \
  --train_budget high \
  --model-dir ./DKLv1/Adv-training-dkl/workdir \
  --mark cifar10_gated_a4b20t4_s6 \
  --data-path ./DKLv1/Adv-training-dkl/../data \
  --seed 0
```

### Parallel Fusion
```bash
sbatch cifar10/run_cifar10_parallel_save100_20.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL
export PYTHONPATH=/home/tong.li003/DKL:/home/tong.li003/DKL/DKLv1/Adv-training-dkl:$PYTHONPATH

python cifar10/train/parallel.py \
  --epochs-sub 100 \
  --epochs-fusion 200 \
  --batch-size 128 \
  --lr 0.1 \
  --alpha 4.0 \
  --beta 20.0 \
  --T 4.0 \
  --train_budget high \
  --model-dir ./DKLv1/Adv-training-dkl/workdir \
  --mark cifar10_parallel_a4b20t4_s6 \
  --data-path ./DKLv1/Adv-training-dkl/../data \
  --seed 0
```

## 评估

与 DKL 评估流程一致：在 `auto_attacks` 目录下执行，使用 `preprocess '01'`。

### Gated Fusion
```bash
# 提交 slurm 任务
sbatch cifar10/run_eval_gated.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL/DKLv1/Adv-training-dkl/auto_attacks
export PYTHONPATH=/home/tong.li003/DKL:$PYTHONPATH

python -m cifar10.eval.gated \
  --checkpoint ../workdir/cifar10_gated_a4b20t4_s6/fusion-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_gated_a4b20t4_s6_epoch200.log
```

或使用脚本：
```bash
bash cifar10/eval_gated.sh
```

### Parallel Fusion
```bash
# 提交 slurm 任务
sbatch cifar10/run_eval_parallel.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL/DKLv1/Adv-training-dkl/auto_attacks
export PYTHONPATH=/home/tong.li003/DKL:$PYTHONPATH

python -m cifar10.eval.parallel \
  --checkpoint ../workdir/cifar10_parallel_a4b20t4_s6/fusion-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_parallel_a4b20t4_s6_epoch200.log
```

或使用脚本：
```bash
bash cifar10/eval_parallel.sh
```

## 对照：DKL 原版命令

```bash
# 训练
sbatch DKLv1/Adv-training-dkl/run_cifar10_save100_20.slurm

# 评估
cd /home/tong.li003/DKL/DKLv1/Adv-training-dkl/auto_attacks
python eval.py --arch WideResNet34 \
  --checkpoint ../workdir/cifar10_dkl_a4b20t4_s6/ours-model-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_dkl_a4b20t4_s6_epoch200.log
```

## 模型保存路径

| 方法 | 保存目录 | checkpoint 文件名 |
|------|----------|-------------------|
| DKL | workdir/cifar10_dkl_a4b20t4_s6/ | ours-model-epoch200.pt |
| Gated | workdir/cifar10_gated_a4b20t4_s6/ | fusion-epoch200.pt |
| Parallel | workdir/cifar10_parallel_a4b20t4_s6/ | fusion-epoch200.pt |

## 保存策略（save100_20）

Gated 与 Parallel 默认从 epoch 100 起每 20 epoch 保存一次（100, 120, 140, 160, 180, 200），共 6 个 fusion checkpoint，与 DKL 一致。可通过 `--save-start` 和 `--save-freq` 调整。
