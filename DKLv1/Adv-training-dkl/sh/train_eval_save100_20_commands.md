# 从 epoch 100 起每 20 epoch 保存一次（共 6 个 checkpoint）

- 保存的 epoch：100, 120, 140, 160, 180, 200
- 不跑 SWA，直接评估上述任意 checkpoint
- 训练脚本：`train_dkl_cifar10_save100_20.py` / `train_dkl_cifar100_save100_20.py`（不修改原 train_dkl_cifar10.py / train_dkl_cifar100.py）

---

## CIFAR-10（复现设置：high budget, lr=0.2, alpha=4, beta=20, T=4）

### 训练
```bash
cd /path/to/Adv-training-dkl
export PYTHONPATH=$(pwd):$PYTHONPATH

python train_dkl_cifar10_save100_20.py \
  --arch WideResNet34_10 \
  --data CIFAR10 \
  --train_budget 'high' \
  --mark cifar10_dkl_a4b20t4_s6 \
  --epsilon 8 \
  --lr 0.2 \
  --beta 20.0 \
  --alpha 4.0 \
  --T 4.0 \
  --epochs 200 \
  --save-start 100 \
  --save-freq 20 \
  --seed 0
```

### 评估（6 个 checkpoint 任选，示例用 epoch 200）
```bash
cd auto_attacks

# 最后一份（推荐）
python eval.py --arch WideResNet34 \
  --checkpoint ../workdir/cifar10_dkl_a4b20t4_s6/ours-model-epoch200.pt \
  --data CIFAR10 \
  --preprocess '01' \
  --log_path logs/cifar10_dkl_a4b20t4_s6_epoch200.log

# 其他可选：--checkpoint ../workdir/cifar10_dkl_a4b20t4_s6/ours-model-epoch100.pt (120,140,160,180 同理)
```

---

## CIFAR-100（复现设置：high budget, lr=0.2, alpha=5, beta=20, T=4）

### 训练
```bash
cd /path/to/Adv-training-dkl
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
```

### 评估（示例用 epoch 200）
```bash
cd auto_attacks

python eval.py --arch WideResNet34 \
  --checkpoint ../workdir/cifar100_dkl_a5b20t4_s6/ours-model-epoch200.pt \
  --data CIFAR100 \
  --preprocess '01' \
  --log_path logs/cifar100_dkl_a5b20t4_s6_epoch200.log
```

---

## 磁盘占用（约）

- 每实验 6 个 model + 6 个 optimizer：约 6 × (176+176) MB ≈ **2.1 GB / 实验**
- CIFAR-10 + CIFAR-100 合计约 **4.2 GB**
