# Gated / Parallel / Soft Routing / Soft Routing Conf 训练与评估命令（对齐 DKL save100_20）

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

### Soft Routing Fusion
```bash
sbatch cifar10/run_cifar10_soft_routing_save100_20.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL
export PYTHONPATH=/home/tong.li003/DKL:/home/tong.li003/DKL/DKLv1/Adv-training-dkl:$PYTHONPATH

python cifar10/train/soft_routing.py \
  --epochs-sub 100 \
  --epochs-fusion 200 \
  --batch-size 128 \
  --lr 0.1 \
  --alpha 4.0 \
  --beta 20.0 \
  --T 4.0 \
  --train_budget high \
  --model-dir ./DKLv1/Adv-training-dkl/workdir \
  --mark cifar10_soft_routing_a4b20t4_s6 \
  --data-path ./DKLv1/Adv-training-dkl/../data \
  --save-start 100 \
  --save-freq 20 \
  --seed 0
```

### Soft Routing Confidence Fusion
```bash
sbatch cifar10/run_cifar10_soft_routing_conf_save100_20.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL
export PYTHONPATH=/home/tong.li003/DKL:/home/tong.li003/DKL/DKLv1/Adv-training-dkl:$PYTHONPATH

python cifar10/train/soft_routing_conf.py \
  --epochs-sub 100 \
  --epochs-fusion 200 \
  --batch-size 128 \
  --lr 0.1 \
  --alpha 4.0 \
  --beta 20.0 \
  --T 4.0 \
  --train_budget high \
  --model-dir ./DKLv1/Adv-training-dkl/workdir \
  --mark cifar10_soft_routing_conf_a4b20t4_s6 \
  --data-path ./DKLv1/Adv-training-dkl/../data \
  --save-start 100 \
  --save-freq 20 \
  --route-T 1.0 \
  --route-margin 0.0 \
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

### Soft Routing Fusion
```bash
# 提交 slurm 任务
sbatch cifar10/run_eval_soft_routing.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL/DKLv1/Adv-training-dkl/auto_attacks
export PYTHONPATH=/home/tong.li003/DKL:$PYTHONPATH

python -m cifar10.eval.soft_routing \
  --checkpoint ../workdir/cifar10_soft_routing_a4b20t4_s6/fusion-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_soft_routing_a4b20t4_s6_epoch200.log
```

或使用脚本：
```bash
bash cifar10/eval_soft_routing.sh
```

### Soft Routing Confidence Fusion
```bash
# 提交 slurm 任务
sbatch cifar10/run_eval_soft_routing_conf.slurm
```

或直接运行：
```bash
cd /home/tong.li003/DKL/DKLv1/Adv-training-dkl/auto_attacks
export PYTHONPATH=/home/tong.li003/DKL:$PYTHONPATH

python -m cifar10.eval.soft_routing_conf \
  --checkpoint ../workdir/cifar10_soft_routing_conf_a4b20t4_s6/fusion-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_soft_routing_conf_a4b20t4_s6_epoch200.log
```

或使用脚本：
```bash
bash cifar10/eval_soft_routing_conf.sh
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
| Soft Routing | workdir/cifar10_soft_routing_a4b20t4_s6/ | fusion-epoch200.pt |
| Soft Routing Conf | workdir/cifar10_soft_routing_conf_a4b20t4_s6/ | fusion-epoch200.pt |

## 保存策略（save100_20）

Gated、Parallel、Soft Routing、Soft Routing Conf 默认从 epoch 100 起每 20 epoch 保存一次（100, 120, 140, 160, 180, 200），共 6 个 fusion checkpoint，与 DKL 一致。可通过 `--save-start` 和 `--save-freq` 调整。

## Soft Routing 说明

- **Stage 1**：m4 为 4+1 类（vehicle+unknown），m6 为 6+1 类（animal+unknown），使用 CIFARUnknownDataset 在全数据上训练
- **Stage 2**：软路由融合，DKL loss + aux_ce_loss

### Soft Routing（原版）
- **SoftRoutingFusion**：路由参数 `--route-a`, `--route-b`, `--route-T`, `--route-margin`（默认 1.0, 0.5, 1.0, 0.0）
- 权重：s4 = a*conf4 + b*unk6, s6 = a*conf6 + b*unk4

### Soft Routing Confidence
- **SoftRoutingConfidenceFusion**：路由参数 `--route-T`, `--route-margin`（默认 1.0, 0.0）
- 权重：w = softmax([c4, c6]/T)，其中 c4=1-P(unk4), c6=1-P(unk6)，直接按专家置信度加权

### 训练策略（与原代码一致）
- **两阶段预热**：Stage 1 子模型 CE → Stage 2 CE warmup 10 epochs
- **BN 冻结**：warmup 后 freeze_bn，epoch 40 后 unfreeze_bn
- **aux 损失防遗忘**：aux_ce_loss(logits4, logits6, y, lut4, lut6)
- **双层 LR 调度**：MultiStepLR(milestones=[epochs//2, epochs*0.75], gamma=0.1) × backbone_lr_ratio(r1=0.15, r2=0.5, r3=0.35)
- **resume 续训**：完整 checkpoint（m4, m6, model, optimizer, scheduler, ema_shadow）
