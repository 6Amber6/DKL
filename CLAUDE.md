# DKL - Decoupled Kullback-Leibler Divergence Loss

## Project Overview

DKL (Decoupled Kullback-Leibler) 是一个 NeurIPS 2024 论文实现，核心贡献是证明 KL 散度等价于 weighted MSE + Cross-Entropy 的解耦形式，并在此基础上提出改进（IKL/GKL），应用于对抗训练、知识蒸馏、半监督学习等任务。

核心损失函数:
```
DKL: L_robust = β * wMSE(Δ_nat, Δ_adv) + α * SCE(P_nat || P_adv)
其中 Δ_ij = logits[i] - logits[j], weight_ij = p_nat[i] * p_nat[j]
```

仓库同时包含 CIFAR-10 上的多模型融合实验（cifar10/），探索并行拼接、门控、软路由等融合策略与 DKL loss 的结合。

## Project Structure

```
DKL/
├── README.md                              # 论文介绍与引用
├── DKLv1/                                 # 第一版 (IKL loss)
│   ├── Adv-training-dkl/                  #   对抗训练 (CIFAR-10/100)
│   │   ├── train_dkl_cifar10.py           #     CIFAR-10 DKL 训练
│   │   ├── train_dkl_cifar100.py          #     CIFAR-100 DKL 训练
│   │   ├── train_trades_cifar.py          #     TRADES baseline
│   │   ├── utils_awp.py                   #     AWP (Adversarial Weight Perturbation)
│   │   ├── autoaug.py                     #     CIFAR10Policy, Cutout
│   │   ├── dataset/cifar.py               #     自定义数据集 (双 augmentation)
│   │   ├── models/wideresnet.py           #     WideResNet 定义
│   │   ├── auto_attacks/eval.py           #     AutoAttack 评估
│   │   ├── swa.py                         #     Stochastic Weight Averaging
│   │   └── utils/                         #     Bar, Logger, AverageMeter
│   ├── KD-dkl/                            #   知识蒸馏 (基于 mdistiller)
│   ├── Imbalanced_KD-dkl/                 #   不平衡知识蒸馏
│   └── Semi-supervised-learning-dkl/      #   半监督学习
├── DKLv2/                                 # 第二版 (GKL loss, 改进版)
│   ├── Adv-training-dkl/                  #   同 v1 结构，使用 GKL 改进
│   ├── KD-dkl/                            #   GKL 知识蒸馏
│   └── Imbalanced_KD-dkl/                 #   GKL 不平衡 KD
└── cifar10/                               # CIFAR-10 多模型融合实验
    ├── model/                             #   模型定义
    │   ├── wideresnet_update.py           #     WideResNet backbone
    │   ├── parallel_wrn.py                #     ParallelFusionWRN, GatedFusionWRN, WRNWithEmbedding
    │   └── soft_routing_wrn.py            #     SoftRoutingFusion, SoftRoutingConfidenceFusion
    ├── train/                             #   训练脚本
    │   ├── parallel.py                    #     拼接融合 + DKL
    │   ├── gated.py                       #     门控融合 + DKL
    │   ├── soft_routing.py                #     软路由 (a*(1-unk) + b*unk_other)
    │   └── soft_routing_conf.py           #     置信路由 (1-unk)
    ├── eval/                              #   AutoAttack 评估
    │   ├── parallel.py, gated.py          #     各方法的 AA eval
    │   └── soft_routing.py, soft_routing_conf.py
    └── *.slurm                            #   SLURM 任务脚本
```

## Dependencies

- Python 3.8.13
- PyTorch 1.8.1+cu111
- NumPy 1.23.1
- torchvision
- autoattack (`pip install autoattack`)
- Open-sourced code from [DKD](https://github.com/megvii-research/mdistiller) (用于 KD)

## Key Hyperparameters

### DKL/GKL 对抗训练 (DKLv1/v2)

| 参数 | CIFAR-10 | CIFAR-100 |
|------|----------|-----------|
| arch | WideResNet34 | WideResNet34 |
| epsilon | 8/255 | 8/255 |
| alpha (SCE weight) | 4.0 | 4.0 |
| beta (wMSE weight) | 20.0 | 20.0 |
| gamma | 1.0 | 1.0 |
| T (temperature) | 4.0 | 4.0 |
| lr | 0.1 | 0.1 |
| epochs | 200 | 200 |
| weight_decay | 5e-4 | 5e-4 |
| AWP gamma | 0.005 | 0.005 |
| AWP warmup | 10 | 10 |
| batch_size | 128 | 128 |

### CIFAR-10 融合实验 (cifar10/)

| 参数 | 值 |
|------|-----|
| Stage 1 (submodel) epochs | 100 |
| Stage 2 (fusion) epochs | 200 |
| alpha / beta / T | 4.0 / 20.0 / 4.0 |
| AWP gamma / warmup | 0.005 / 10 |
| Cutout | n_holes=1, length=8 |
| EMA decay | 0.999 |
| aux_weight | 0.02 |
| save_start / save_freq | 100 / 20 |

## Common Commands

### DKL/GKL 对抗训练 (单模型)

```bash
cd DKLv2/Adv-training-dkl

# CIFAR-10 训练
python train_dkl_cifar10.py \
  --arch WideResNet34 --data CIFAR10 \
  --epochs 200 --batch-size 128 --lr 0.1 \
  --epsilon 8 --num-steps 10 --step-size 2 \
  --alpha 4.0 --beta 20.0 --T 4.0 \
  --awp-gamma 0.005 --awp-warmup 10 \
  --aug basic --train_budget low \
  --model-dir ./workdir --mark cifar10_dkl \
  --data-path ../data

# AutoAttack 评估
python auto_attacks/eval.py \
  --arch WideResNet34 --data CIFAR10 \
  --checkpoint ./workdir/cifar10_dkl/model_best.pt \
  --data_dir ../data --preprocess 01 --epsilon 0.03137
```

注意: DKL 训练脚本中 `--epsilon` 和 `--step-size` 使用整数 (pixel 值)，内部除以 255。

### CIFAR-10 融合实验

```bash
cd /path/to/DKL
export PYTHONPATH=$PWD:$PWD/DKLv1/Adv-training-dkl

# 并行拼接融合训练
python cifar10/train/parallel.py \
  --epochs-sub 100 --epochs-fusion 200 \
  --batch-size 128 --lr 0.1 \
  --epsilon 0.031372549 \
  --alpha 4.0 --beta 20.0 \
  --awp-gamma 0.005 --awp-warmup 10 \
  --model-dir ./workdir --mark cifar10_parallel \
  --data-path ./data

# AutoAttack 评估
python cifar10/eval/parallel.py \
  --checkpoint ./workdir/cifar10_parallel/fusion-epoch200.pt \
  --data-dir ./data --preprocess 01
```

## DKL vs TRADES 区别

| 特性 | TRADES | DKL |
|------|--------|-----|
| 鲁棒 loss | β * KL(f(x), f(x')) | β * wMSE + α * SCE |
| 样本权重 | 无 | 基于 class prior 的 pairwise weight |
| 温度 | 无 | T=4.0 |
| AWP | 可选 | 默认启用 |
| 数据输入 | raw [0,1] 或 normalized | raw [0,1] |
| epsilon 传参 | 浮点 (0.031) | 整数 (8)，内部 /255 |
| 训练 budget | 固定 PGD 步数 | 渐进式 (low/high budget) |

## CIFAR-10 融合方法变体

项目在 cifar10/ 下探索了 4 种融合策略，均使用 DKL loss:

1. **并行拼接 (Parallel)**: `concat(e4, e6)` → `Linear(1280, 10)`，梯度流畅通
2. **门控融合 (Gated)**: `sigmoid(gate) * e4 + (1-sigmoid(gate)) * e6`，sigmoid 可能导致梯度遮蔽
3. **软路由 (Soft Routing)**: `score_i = a*(1-unk_i) + b*unk_j`，双重 softmax 路由
4. **置信路由 (Confidence)**: `w = softmax([1-unk4, 1-unk6] / T)`，简化路由

类别划分: Vehicle [0,1,8,9] (4类) + Animal [2,3,4,5,6,7] (6类)

## Checkpoint Formats

### DKLv1/v2 对抗训练

```python
# 训练保存
torch.save(model.state_dict(), path)  # 或包含 'state_dict' key
```

### CIFAR-10 融合实验

```python
# 完整 checkpoint (checkpoint-last.pt)
{
    'm4_state_dict': ...,       # Vehicle submodel
    'm6_state_dict': ...,       # Animal submodel
    'model_state_dict': ...,    # Fusion model
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'ema_shadow': {...},
    'fusion_epoch': int
}

# 评估 checkpoint (fusion-epoch{N}.pt)
model.state_dict()  # 纯 state_dict
```

## Coding Conventions

- 数据输入为 raw [0,1] 像素值（ToTensor only，不做 mean/std 归一化）
- AutoAttack 评估使用 `--preprocess 01`
- DKL 训练脚本 epsilon/step_size 用整数 (8, 2)，内部 /255; cifar10/ 下用浮点
- 训练遵循: Stage 1 (CE pretrain) → Stage 2 (DKL + AWP fusion)
- 渐进式攻击强度: epsilon 从 0 线性增长到目标值; PGD 步数从 2 增长到 5 (high budget)
- 使用 DKLv1/Adv-training-dkl/ 下的 utils (Bar, AverageMeter) 和 utils_awp
- PYTHONPATH 需包含 DKL 根目录和 DKLv1/Adv-training-dkl
