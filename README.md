# Parallel Fusion Adversarial Training with DKL Loss

Improving adversarial robustness through **parallel sub-network fusion** on top of the [Decoupled Kullback-Leibler (DKL) Divergence Loss](https://arxiv.org/pdf/2305.13948) (NeurIPS 2024).

## Motivation

Standard adversarial training (e.g., TRADES, DKL) relies on a single WideResNet to learn both natural accuracy and adversarial robustness simultaneously. We hypothesize that splitting the representation space across semantically meaningful sub-networks and fusing their embeddings can yield richer, more robust features.

**Key Insight:** By grouping CIFAR classes into semantically coherent meta-groups (based on coarse superclass hierarchy) and training dedicated sub-networks per group, each sub-network specializes on a subset of visual concepts. The fused embedding captures complementary information that a single network cannot, improving both clean accuracy and adversarial robustness.

## Method

1. **Stage 1 (Sub-network Pre-training):** Train separate WideResNet sub-networks on semantically grouped class subsets using cross-entropy loss.
2. **Stage 2 (Fusion + Adversarial Training):** Concatenate sub-network embeddings, add a linear fusion head for full classification, and fine-tune end-to-end with DKL adversarial training (+ AWP, progressive epsilon warmup, EMA).

Architecture: `WRN sub-networks -> Embedding Concatenation -> FC -> 10/100 classes`

## Results

### CIFAR-10 (epsilon = 8/255)

| Model | Architecture | Clean Acc | AutoAttack |
|-------|-------------|-----------|------------|
| DKL baseline | WRN-34-10 | 84.30 | 56.35 |
| **Ours** | **WRN-34-10 x 2 + FC** | **86.58** | **57.15** |

> Clean **+2.28**, AutoAttack **+0.80** (evaluated at epoch 180)

### CIFAR-100 (epsilon = 8/255)

| Model | Architecture | Clean Acc | AutoAttack |
|-------|-------------|-----------|------------|
| DKL baseline | WRN-34-10 | 65.18 | 31.22 |
| **Ours** | **WRN-34-10 x 4 + FC** | **69.36** | **32.54** |

> Clean **+4.18**, AutoAttack **+1.32** (evaluated at epoch 200)

## Project Structure

```
cifar10/
  model/          # Fusion architectures (parallel, gated, soft-routing)
  train/          # Training scripts for 4 fusion methods
  eval/           # AutoAttack evaluation
cifar100/
  model/          # Parallel fusion WRN for 100 classes
  train/          # Parallel fusion training
  eval/           # AutoAttack evaluation
DKLv1/
  Adv-training-dkl/   # DKL baseline reproduction
```

### CIFAR-10: Four Fusion Methods

| Method | Description |
|--------|-------------|
| **Parallel** | Two WRN sub-networks, embedding concatenation + FC |
| **Gated** | Learned sigmoid gating weights for fusion |
| **Soft Routing** | Asymmetric weighting based on unknown score |
| **Confidence Routing** | Simplified routing based on prediction confidence |

### CIFAR-100: Parallel Fusion

Four meta-groups by coarse superclass (textured organic / smooth organic / rigid man-made / large structures), each with a dedicated WRN-34-10 sub-network, fused via embedding concatenation + FC.

## Training

```bash
# CIFAR-10 Parallel Fusion
cd DKL
PYTHONPATH=.:DKLv1/Adv-training-dkl python cifar10/train/parallel.py \
  --epochs-sub 100 --epochs-fusion 200 \
  --epsilon 0.031372549 --alpha 4.0 --beta 20.0 --T 4.0 \
  --train_budget high --sub-depth 34 --sub-widen 10 \
  --awp-gamma 0.005 --awp-warmup 10

# CIFAR-100 Parallel Fusion
PYTHONPATH=.:DKLv1/Adv-training-dkl python cifar100/train/parallel.py \
  --epochs-sub 100 --epochs-fusion 200 \
  --epsilon 0.031372549 --alpha 4.0 --beta 20.0 --T 4.0 \
  --train_budget high --sub-depth 34 --sub-widen 10 \
  --awp-gamma 0.005 --awp-warmup 10
```

## Evaluation

```bash
# AutoAttack (CIFAR-10)
PYTHONPATH=.:DKLv1/Adv-training-dkl python cifar10/eval/parallel.py \
  --model-path <checkpoint>

# AutoAttack (CIFAR-100)
PYTHONPATH=.:DKLv1/Adv-training-dkl python cifar100/eval/parallel.py \
  --model-path <checkpoint>
```

## DKL Baseline Reproduction

Based on the [DKL](https://github.com/jiequancui/DKL) codebase (NeurIPS 2024). See `DKLv1/Adv-training-dkl/` for standalone baseline training scripts.

```bash
cd DKLv1/Adv-training-dkl
python train_dkl_cifar10.py --arch WideResNet34_10 --train_budget high \
  --epsilon 8 --alpha 4.0 --beta 20.0 --T 4.0 --lr 0.2 --epochs 200
```

## Citation

```bibtex
@article{cui2023decoupled,
  title={Decoupled Kullback-Leibler Divergence Loss},
  author={Cui, Jiequan and Tian, Zhuotao and Zhong, Zhisheng and Qi, Xiaojuan and Yu, Bei and Zhang, Hanwang},
  journal={arXiv preprint arXiv:2305.13948},
  year={2023}
}
```
