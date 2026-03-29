"""
DKL-based Parallel WRN Training for CIFAR-100.
Architecture: 4-group submodels (by coarse superclass) with fusion for 100-class.

Loss: L = L_nat + L_robust + L_aux.
  L_nat = CE(logits_nat, y).
  L_robust = DKL: beta * wMSE(delta_nat - delta_adv) + alpha * SCE(p_nat || p_adv).
  L_aux = 0.02 * sum(CE on each group head for its classes).

Aligned with DKL baseline (train_dkl_cifar100.py):
  - raw [0,1] input (no mean/std normalize)
  - DKL loss with sample_weight (pairwise class prior)
  - progressive epsilon warmup + progressive PGD steps
  - AWP (Adversarial Weight Perturbation)
  - cosine annealing LR (scheduler starts after warmup)
  - WEIGHT update with temperature T
"""
from __future__ import print_function
import os
import sys
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets

# Add DKL root for cifar100.model, then Adv-training-dkl for utils/utils_awp
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'DKLv1', 'Adv-training-dkl'))
from cifar100.model.parallel_wrn import ParallelFusionWRN100
from cifar10.model.parallel_wrn import WRNWithEmbedding
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP
from autoaug import Cutout


# =========================================================
# CIFAR-100 fine -> coarse mapping (standard 20 superclasses)
# =========================================================
CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]

# =========================================================
# Meta-groups by coarse class IDs
# =========================================================
COARSE_GROUPS = {
    "textured_organic": [7, 15, 16, 8, 13],
    "smooth_organic": [1, 0, 11, 12, 14],
    "rigid_manmade": [6, 5, 3, 18, 9],
    "large_structures": [10, 17, 19, 2, 4],
}
GROUP_ORDER = ["textured_organic", "smooth_organic", "rigid_manmade", "large_structures"]

NUM_CLASSES = 100


def build_fine_classes_for_group(group_coarse):
    return sorted([i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse])


GROUP_FINE = [build_fine_classes_for_group(COARSE_GROUPS[name]) for name in GROUP_ORDER]


# =========================================================
# EMA
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])


# =========================================================
# Dataset wrapper for fine labels
# =========================================================
class CIFARFineSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_fine_classes):
        self.base = base_dataset
        self.map = {c: i for i, c in enumerate(keep_fine_classes)}
        targets = torch.tensor(self.base.targets)
        self.idx = (
            (targets.unsqueeze(1) == torch.tensor(keep_fine_classes))
            .any(dim=1).nonzero(as_tuple=False).view(-1)
        )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.base[self.idx[i]]
        return x, self.map[int(y)]


# =========================================================
# BN freeze/unfreeze
# =========================================================
def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

def unfreeze_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
            for p in m.parameters():
                p.requires_grad = True


# =========================================================
# DKL loss (aligned with baseline)
# =========================================================
def dkl_loss(logits_nat, logits_adv, weight, alpha, beta):
    num_classes = logits_nat.size(1)
    delta_n = logits_nat.view(-1, num_classes, 1) - logits_nat.view(-1, 1, num_classes)
    delta_a = logits_adv.view(-1, num_classes, 1) - logits_adv.view(-1, 1, num_classes)
    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * weight).sum() / logits_nat.size(0)
    loss_sce = -(F.softmax(logits_nat, dim=1).detach() * F.log_softmax(logits_adv, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce


# =========================================================
# Aux loss (per-group sub-logits)
# =========================================================
def aux_ce_loss(aux_outputs, y, device, weight=0.02):
    loss = 0.0
    for group_idx, logits_g in enumerate(aux_outputs):
        if logits_g is None:
            continue
        fine_classes = torch.tensor(GROUP_FINE[group_idx], device=device)
        mask = torch.isin(y, fine_classes)
        if mask.any():
            loss += F.cross_entropy(logits_g[mask], torch.searchsorted(fine_classes, y[mask]))
    return weight * loss


# =========================================================
# Adversarial perturbation (DKL style, aligned with baseline)
# =========================================================
def perturb_input(model, x_natural, step_size, epsilon, perturb_steps, weight, device):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_nat = model(x_natural)
            logits_adv = model(x_adv)
            loss_kl = dkl_loss(logits_nat, logits_adv, weight, 1.0, 1.0)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


# =========================================================
# Backbone LR ratio (smooth linear ramp)
# =========================================================
def backbone_lr_ratio(epoch, total_epochs, r1=0.15, r2=0.5, r3=0.35):
    p1 = int(total_epochs * 0.4)
    p2 = int(total_epochs * 0.7)
    if epoch <= p1:
        return r1 + (r2 - r1) * (epoch - 1) / max(p1 - 1, 1)
    if epoch <= p2:
        return r2
    return r2 + (r3 - r2) * (epoch - p2) / max(total_epochs - p2, 1)


# =========================================================
# Arguments
# =========================================================
parser = argparse.ArgumentParser(description='DKL Parallel WRN CIFAR-100')
parser.add_argument('--epochs-sub', type=int, default=100)
parser.add_argument('--epochs-fusion', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--epsilon', type=float, default=8/255)
parser.add_argument('--alpha', type=float, default=4.0)
parser.add_argument('--beta', type=float, default=20.0)
parser.add_argument('--T', type=float, default=4.0)
parser.add_argument('--train_budget', type=str, default='low', choices=['low', 'high'])
parser.add_argument('--sub-depth', type=int, default=34, choices=[28, 34])
parser.add_argument('--sub-widen', type=int, default=10, choices=[4, 8, 10])
parser.add_argument('--model-dir', default='./workdir')
parser.add_argument('--mark', type=str, default='cifar100_parallel')
parser.add_argument('--data-path', type=str, default='../data')
parser.add_argument('--awp-gamma', type=float, default=0.005)
parser.add_argument('--awp-warmup', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume', default='auto', help='path to checkpoint or "auto"')
parser.add_argument('--use-ema', action='store_true', default=True)
parser.add_argument('--no-ema', action='store_false', dest='use_ema')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model_dir = os.path.join(args.model_dir, args.mark)
os.makedirs(model_dir, exist_ok=True)

# =========================================================
# Data: raw [0,1] (aligned with DKL baseline, no mean/std)
# =========================================================
transform_sub = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=1, magnitude=5),
    transforms.ToTensor(),
])
transform_fusion = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=5),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
])
transform_test = transforms.Compose([transforms.ToTensor()])

base_train_sub = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_sub)
base_train_fusion = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_fusion)
testset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)

group_loaders = []
for fine_classes in GROUP_FINE:
    loader = torch.utils.data.DataLoader(
        CIFARFineSubset(base_train_sub, fine_classes),
        batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    group_loaders.append(loader)

train_loader_100 = torch.utils.data.DataLoader(
    base_train_fusion, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

group_test_loaders = []
for fine_classes in GROUP_FINE:
    group_test_loaders.append(torch.utils.data.DataLoader(
        CIFARFineSubset(testset, fine_classes),
        batch_size=200, shuffle=False, num_workers=4
    ))


# =========================================================
# Train CE epoch (Stage 1)
# =========================================================
def train_ce_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


# =========================================================
# Train DKL epoch (Stage 2)
# =========================================================
def train_dkl_epoch(model, raw_model, loader, optimizer, epoch, awp_adversary, ema, weight, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    bar = Bar('Processing', max=len(loader))

    weight = weight if weight is not None else torch.ones(NUM_CLASSES, NUM_CLASSES, device=device) / NUM_CLASSES
    WEIGHT = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    epoch_scale = args.epochs_fusion / 200.0
    varepsilon = args.epsilon * (epoch / args.epochs_fusion)

    if args.train_budget == 'low':
        step_size = varepsilon
        iters_attack = 2
    else:
        if epoch <= int(50 * epoch_scale):
            step_size, iters_attack = varepsilon, 2
        elif epoch <= int(100 * epoch_scale):
            step_size, iters_attack = 2 * varepsilon / 3, 3
        elif epoch <= int(150 * epoch_scale):
            step_size, iters_attack = varepsilon / 2, 4
        else:
            step_size, iters_attack = varepsilon / 2, 5

    for batch_idx, (data, target) in enumerate(loader):
        x_natural, target = data.to(device), target.to(device)
        with torch.no_grad():
            onehot = F.one_hot(target, NUM_CLASSES).float()
            s_n = onehot @ weight
            sample_weight = s_n.view(-1, NUM_CLASSES, 1) @ s_n.view(-1, 1, NUM_CLASSES)

        x_adv = perturb_input(model, x_natural, step_size, varepsilon, iters_attack, sample_weight, device)
        model.train()

        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(x_adv, x_natural, target, sample_weight, args.alpha, args.beta)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        aux_outputs, logits_nat = model(x_natural, return_aux=True)
        logits_adv = model(x_adv)

        with torch.no_grad():
            WEIGHT = WEIGHT + (onehot.t() @ F.softmax(logits_nat.clone().detach() / args.T, dim=-1))

        loss_robust = dkl_loss(logits_nat, logits_adv, sample_weight, args.alpha, args.beta)
        loss_aux = aux_ce_loss(aux_outputs, target, device)
        loss_natural = F.cross_entropy(logits_nat, target)
        loss = loss_natural + loss_robust + loss_aux

        prec1, _ = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if ema is not None:
            ema.update(raw_model)

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        bar.suffix = '({batch}/{size}) Loss:{loss:.4f} top1:{top1:.2f}'.format(
            batch=batch_idx + 1, size=len(loader), loss=losses.avg, top1=top1.avg)
        bar.next()
    bar.finish()

    row_sum = WEIGHT.sum(dim=1, keepdim=True).clamp_min(1e-12)
    WEIGHT = WEIGHT / row_sum
    return losses.avg, top1.avg, WEIGHT


# =========================================================
# Test
# =========================================================
def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# =========================================================
# PGD-40 evaluation (raw [0,1] input)
# =========================================================
def eval_pgd40(model, loader, epsilon, device, num_steps=40, step_size=0.003):
    model.eval()
    clean_correct, robust_correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            clean_correct += (model(x).argmax(1) == y).sum().item()
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(num_steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = x.detach() + torch.clamp(x_adv - x.detach(), -epsilon, epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        with torch.no_grad():
            robust_correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return clean_correct / total, robust_correct / total


# =========================================================
# Main
# =========================================================
def main():
    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = os.path.join(model_dir, 'checkpoint-last.pt')

    resume_loaded = False
    warmup_start = 1
    start_epoch_dkl = 11
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f'[INFO] Resuming from {resume_path}')
            resume_loaded = True
            fusion_epoch = checkpoint.get('fusion_epoch', 0)
            if fusion_epoch < 10:
                warmup_start = fusion_epoch + 1
                start_epoch_dkl = 11
            else:
                warmup_start = 11
                start_epoch_dkl = fusion_epoch + 1

    # Stage 1: Submodels
    group_names = GROUP_ORDER
    submodels = []
    stage1_files = [os.path.join(model_dir, f'wrn_{name}_final.pt') for name in group_names]
    stage1_done = resume_loaded or all(os.path.isfile(f) for f in stage1_files)

    if stage1_done and not resume_loaded:
        print('==== Stage 1: Loading saved submodels ====')
        for i, name in enumerate(group_names):
            m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                                 num_classes=len(GROUP_FINE[i])).to(device)
            m.load_state_dict(torch.load(stage1_files[i], map_location=device))
            submodels.append(m)
            print(f'  Loaded {name} ({len(GROUP_FINE[i])} classes)')
    elif not resume_loaded:
        print('==== Stage 1: Submodels (CE) ====')
        for i, loader in enumerate(group_loaders):
            m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                                 num_classes=len(GROUP_FINE[i])).to(device)
            opt = optim.SGD(m.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            sched = CosineAnnealingLR(opt, T_max=args.epochs_sub, eta_min=0.01)
            for ep in range(1, args.epochs_sub + 1):
                _, a = train_ce_epoch(m, loader, opt, device)
                sched.step()
                t_acc = test(m, group_test_loaders[i], device)
                print(f'[Sub][{ep}] {group_names[i]} train={a*100:.2f}% test={t_acc*100:.2f}%, lr={opt.param_groups[0]["lr"]:.6f}')
            torch.save(m.state_dict(), stage1_files[i])
            submodels.append(m)
            print(f'  Saved {group_names[i]} -> {stage1_files[i]}')
    else:
        # resume_loaded: create placeholder submodels, will be loaded from checkpoint
        for i in range(len(group_names)):
            m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                                 num_classes=len(GROUP_FINE[i])).to(device)
            submodels.append(m)

    # Stage 2: Fusion + DKL
    fusion = ParallelFusionWRN100(submodels, num_classes=NUM_CLASSES, freeze_backbone=False).to(device)
    for p in fusion.parameters():
        p.requires_grad = True

    # Multi-GPU DataParallel
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        fusion = nn.DataParallel(fusion)
        print(f'[INFO] Using {n_gpus} GPUs via DataParallel')
    raw_model = fusion.module if hasattr(fusion, 'module') else fusion

    params = [{'params': raw_model.fc.parameters(), 'lr': args.lr}]
    for m in raw_model.submodels:
        params.append({'params': m.parameters(), 'lr': args.lr})
    optimizer = optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_fusion - 10, eta_min=0)
    ema = EMA(raw_model, decay=0.999) if args.use_ema else None

    if resume_loaded and checkpoint is not None:
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            ckpt_opt = checkpoint['optimizer_state_dict']
            n_cur, n_ckpt = len(optimizer.param_groups), len(ckpt_opt.get('param_groups', []))
            if n_cur == n_ckpt:
                optimizer.load_state_dict(ckpt_opt)
                print(f'[RESUME] optimizer loaded')
            else:
                print(f'[RESUME] optimizer param_groups mismatch (cur={n_cur}, ckpt={n_ckpt}), skip')
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'[RESUME] scheduler loaded')
        if args.use_ema and ema is not None and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']

    # Proxy for AWP
    proxy_subs = [WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                                    num_classes=len(GROUP_FINE[i])).to(device)
                  for i in range(len(group_names))]
    proxy_fusion = ParallelFusionWRN100(proxy_subs, num_classes=NUM_CLASSES, freeze_backbone=False).to(device)
    for p in proxy_fusion.parameters():
        p.requires_grad = True
    if n_gpus > 1:
        proxy_fusion = nn.DataParallel(proxy_fusion)
    proxy_optim = optim.SGD(proxy_fusion.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=fusion, proxy=proxy_fusion, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    # CE warmup (10 epochs)
    print('==== CE warmup (10 epochs) ====')
    for ep in range(warmup_start, 11):
        fusion.train()
        for x, y in train_loader_100:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = fusion(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(raw_model)
        print(f'[Warmup] Epoch {ep}/10')
        ckpt = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_shadow': ema.shadow if ema is not None else {},
            'fusion_epoch': ep,
        }
        torch.save(ckpt, os.path.join(model_dir, 'checkpoint-last.pt'))

    freeze_bn(fusion)

    bn_unfrozen = start_epoch_dkl > 40
    if bn_unfrozen:
        unfreeze_bn(fusion)
        print('[INFO] Resumed past epoch 40, BN already unfrozen')

    # DKL training
    print('==== DKL training ====')
    best_robust = 0.0
    if resume_loaded and checkpoint is not None:
        best_robust = checkpoint.get('best_robust', 0.0)
    weight = None
    for ep in range(start_epoch_dkl, args.epochs_fusion + 1):
        ratio = backbone_lr_ratio(ep, args.epochs_fusion)
        fusion_lr = optimizer.param_groups[0]['lr']
        for g in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[g]['lr'] = fusion_lr * ratio

        loss_avg, acc_avg, weight = train_dkl_epoch(
            fusion, raw_model, train_loader_100, optimizer, ep, awp_adversary, ema, weight, device
        )
        scheduler.step()

        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')

        if ema is not None:
            ema.apply_to(raw_model)
        val_acc = test(fusion, test_loader, device)

        # PGD-40 every 10 epochs and last epoch
        if ep % 10 == 0 or ep == args.epochs_fusion:
            clean_acc, robust_acc = eval_pgd40(fusion, test_loader, args.epsilon, device)
            print(f'[DKL][{ep}/{args.epochs_fusion}] loss={loss_avg:.4f} acc={acc_avg:.2f}% val={val_acc*100:.2f}% | PGD-40: clean={clean_acc*100:.2f}% robust={robust_acc*100:.2f}%')
            if robust_acc > best_robust:
                best_robust = robust_acc
                torch.save(raw_model.state_dict(), os.path.join(model_dir, 'fusion-best.pt'))
                print(f'[DKL][{ep}] New best PGD-40={robust_acc*100:.2f}%, saved fusion-best.pt')
        else:
            print(f'[DKL][{ep}/{args.epochs_fusion}] loss={loss_avg:.4f} acc={acc_avg:.2f}% val={val_acc*100:.2f}%')

        if ema is not None:
            ema.restore(raw_model)
        fusion.train()

        # Save last checkpoint (for resume)
        ckpt = {
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_shadow': ema.shadow if ema is not None else {},
            'fusion_epoch': ep,
            'best_robust': best_robust,
        }
        torch.save(ckpt, os.path.join(model_dir, 'checkpoint-last.pt'))

    # Save final model with EMA weights
    if ema is not None:
        ema.apply_to(raw_model)
    torch.save(raw_model.state_dict(), os.path.join(model_dir, 'fusion-last.pt'))
    if ema is not None:
        ema.restore(raw_model)


if __name__ == '__main__':
    main()
