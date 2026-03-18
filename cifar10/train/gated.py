"""
DKL-based Gated Fusion WRN Training for CIFAR-10.
Baseline: DKL. Architecture: 4-class + 6-class submodels, gated fusion (alpha*e4 + (1-alpha)*e6) -> 10-class.

Loss: L = L_nat + L_robust + L_aux.
  L_nat = CE(logits_nat, y).
  L_robust = DKL: beta * wMSE + alpha * SCE, with sample weight.
  L_aux = 0.02 * (CE on 4-class head for vehicle + CE on 6-class head for animal).
"""
from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'DKLv1', 'Adv-training-dkl'))
from cifar10.model.parallel_wrn import WRNWithEmbedding, GatedFusionWRN
from utils import Bar, AverageMeter, accuracy
from utils_awp import TradesAWP
from autoaug import Cutout

VEHICLE_CLASSES = [0, 1, 8, 9]
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]


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


class CIFARSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, keep_classes):
        self.base = base_dataset
        self.map = {c: i for i, c in enumerate(keep_classes)}
        targets = torch.tensor(self.base.targets)
        self.idx = (
            (targets.unsqueeze(1) == torch.tensor(keep_classes))
            .any(dim=1).nonzero(as_tuple=False).view(-1)
        )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, y = self.base[self.idx[i]]
        return x, self.map[y]


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


def dkl_loss(logits_nat, logits_adv, weight, alpha, beta):
    num_classes = logits_nat.size(1)
    delta_n = logits_nat.view(-1, num_classes, 1) - logits_nat.view(-1, 1, num_classes)
    delta_a = logits_adv.view(-1, num_classes, 1) - logits_adv.view(-1, 1, num_classes)
    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * weight).sum() / logits_nat.size(0)
    loss_sce = -(F.softmax(logits_nat, dim=1).detach() * F.log_softmax(logits_adv, dim=-1)).sum(1).mean()
    return beta * loss_mse + alpha * loss_sce


def aux_ce_loss(logits4, logits6, y, device, weight=0.02):
    loss = 0.0
    if logits4 is not None:
        mask4 = torch.isin(y, torch.tensor(VEHICLE_CLASSES, device=device))
        if mask4.any():
            remap4 = torch.tensor(VEHICLE_CLASSES, device=device)
            loss += F.cross_entropy(logits4[mask4], torch.searchsorted(remap4, y[mask4]))
    if logits6 is not None:
        mask6 = torch.isin(y, torch.tensor(ANIMAL_CLASSES, device=device))
        if mask6.any():
            remap6 = torch.tensor(ANIMAL_CLASSES, device=device)
            loss += F.cross_entropy(logits6[mask6], torch.searchsorted(remap6, y[mask6]))
    return weight * loss


def perturb_input(model, x_natural, step_size, epsilon, perturb_steps, weight, device):
    model.eval()
    with torch.no_grad():
        logits_nat = model(x_natural).detach()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=device).detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits_adv = model(x_adv)
            loss_kl = dkl_loss(logits_nat, logits_adv, weight, 1.0, 0.0)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def backbone_lr_ratio(epoch, total_epochs, r1=0.2, r2=0.2, r3=0.2):
    """Gated: fixed ratio 0.2 (same as TRADES gated)."""
    return r1


parser = argparse.ArgumentParser(description='DKL Gated Fusion WRN CIFAR-10')
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
parser.add_argument('--train_budget', type=str, default='high', choices=['low', 'high'])
parser.add_argument('--model-dir', default='./workdir')
parser.add_argument('--mark', type=str, default='gated')
parser.add_argument('--data-path', type=str, default='../data')
parser.add_argument('--awp-gamma', type=float, default=0.005)
parser.add_argument('--awp-warmup', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--resume', default='auto', help='path to checkpoint or "auto" for model_dir/checkpoint-last.pt')
parser.add_argument('--use-ema', action='store_true', default=True, help='use EMA of weights (default: True)')
parser.add_argument('--no-ema', action='store_false', dest='use_ema', help='disable EMA (e.g. for DKL-style SWA later)')
parser.add_argument('--save-start', type=int, default=100, help='first epoch to save fusion (then every save-freq)')
parser.add_argument('--save-freq', type=int, default=20, help='save fusion every N epochs from save-start')
args = parser.parse_args()

NUM_CLASSES = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model_dir = os.path.join(args.model_dir, args.mark)
os.makedirs(model_dir, exist_ok=True)

# Stage 1: basic aug for submodels
transform_sub = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# Stage 2: basic + Cutout for fusion
transform_fusion = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=8),
])

transform_test = transforms.Compose([transforms.ToTensor()])

base_train_sub = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_sub)
base_train_fusion = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_fusion)
testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)

train_loader_4 = torch.utils.data.DataLoader(
    CIFARSubset(base_train_sub, VEHICLE_CLASSES),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
train_loader_6 = torch.utils.data.DataLoader(
    CIFARSubset(base_train_sub, ANIMAL_CLASSES),
    batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
train_loader_10 = torch.utils.data.DataLoader(
    base_train_fusion, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)


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


def train_dkl_epoch(model, loader, optimizer, epoch, awp_adversary, ema, weight, device):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    bar = Bar('Processing', max=len(loader))

    weight = weight if weight is not None else torch.ones(NUM_CLASSES, NUM_CLASSES, device=device) / NUM_CLASSES
    WEIGHT = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=device)
    epoch_scale = args.epochs_fusion / 200.0
    varepsilon = args.epsilon * (epoch / args.epochs_fusion)

    if args.train_budget == 'low':
        step_size, iters_attack = varepsilon, 2
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
        out4, out6, logits_nat = model(x_natural, return_aux=True)
        logits_adv = model(x_adv)

        with torch.no_grad():
            WEIGHT = WEIGHT + (onehot.t() @ F.softmax(logits_nat.clone().detach() / args.T, dim=-1))

        loss_robust = dkl_loss(logits_nat, logits_adv, sample_weight, args.alpha, args.beta)
        loss_aux = aux_ce_loss(out4, out6, target, device)
        loss_natural = F.cross_entropy(logits_nat, target)
        loss = loss_natural + loss_robust + loss_aux

        prec1, _ = accuracy(logits_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        bar.suffix = '({batch}/{size}) Loss:{loss:.4f} top1:{top1:.2f}'.format(
            batch=batch_idx + 1, size=len(loader), loss=losses.avg, top1=top1.avg)
        bar.next()
    bar.finish()

    row_sum = WEIGHT.sum(dim=1, keepdim=True).clamp_min(1e-12)
    WEIGHT = WEIGHT / row_sum
    return losses.avg, top1.avg, WEIGHT


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


def main():
    resume_path = args.resume
    if resume_path == 'auto':
        resume_path = os.path.join(model_dir, 'checkpoint-last.pt')

    resume_loaded = False
    warmup_start = 1
    start_epoch_dkl = 11  # DKL ep 11~200 = 190 epochs (10 warmup + 190 DKL = 200 total)
    checkpoint = None
    if resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f'[INFO] Resuming from {resume_path}')
            resume_loaded = True
            fusion_epoch = checkpoint.get('fusion_epoch', checkpoint.get('stage2_epoch', 0))
            if fusion_epoch < 10:
                warmup_start = fusion_epoch + 1
                start_epoch_dkl = 11
            else:
                warmup_start = 11
                start_epoch_dkl = fusion_epoch + 1

    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)

    if resume_loaded and checkpoint is not None:
        m4.load_state_dict(checkpoint['m4_state_dict'])
        m6.load_state_dict(checkpoint['m6_state_dict'])

    if not resume_loaded:
        print('==== Stage 1: Submodels (CE) ====')
        opt4 = optim.SGD(m4.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        opt6 = optim.SGD(m6.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        for ep in range(1, args.epochs_sub + 1):
            l4, a4 = train_ce_epoch(m4, train_loader_4, opt4, device)
            l6, a6 = train_ce_epoch(m6, train_loader_6, opt6, device)
            print(f'[Sub][{ep}] 4c acc={a4*100:.2f}% | 6c acc={a6*100:.2f}%')
        torch.save(m4.state_dict(), os.path.join(model_dir, 'wrn4_final.pt'))
        torch.save(m6.state_dict(), os.path.join(model_dir, 'wrn6_final.pt'))

    fusion = GatedFusionWRN(m4, m6).to(device)
    for p in fusion.parameters():
        p.requires_grad = True

    params = [
        {'params': fusion.fc.parameters(), 'lr': args.lr},
        {'params': fusion.m4.parameters(), 'lr': args.lr},
        {'params': fusion.m6.parameters(), 'lr': args.lr},
        {'params': fusion.gate.parameters(), 'lr': args.lr},
    ]
    optimizer = optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_fusion, eta_min=0)
    ema = EMA(fusion, decay=0.999) if args.use_ema else None

    if resume_loaded and checkpoint is not None:
        fusion.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            ckpt_opt = checkpoint['optimizer_state_dict']
            n_cur, n_ckpt = len(optimizer.param_groups), len(ckpt_opt.get('param_groups', []))
            if n_cur == n_ckpt:
                optimizer.load_state_dict(ckpt_opt)
                print(f'[RESUME] optimizer loaded, lr={[pg["lr"] for pg in optimizer.param_groups]}')
            else:
                print(f'[RESUME] optimizer param_groups mismatch (cur={n_cur}, ckpt={n_ckpt}), not loading')
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'[RESUME] scheduler loaded')
        if args.use_ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']

    proxy_m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4).to(device)
    proxy_m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6).to(device)
    proxy_fusion = GatedFusionWRN(proxy_m4, proxy_m6).to(device)
    proxy_optim = optim.SGD(proxy_fusion.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=fusion, proxy=proxy_fusion, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    print('==== CE warmup (10 epochs) ====')
    for ep in range(warmup_start, 11):
        fusion.train()
        for x, y in train_loader_10:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = fusion(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(fusion)
        scheduler.step()
        print(f'[Warmup] Epoch {ep}/10')
        ckpt = {
            'm4_state_dict': m4.state_dict(),
            'm6_state_dict': m6.state_dict(),
            'model_state_dict': fusion.state_dict(),
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

    print('==== DKL training ====')
    weight = None
    for ep in range(start_epoch_dkl, args.epochs_fusion + 1):
        ratio = backbone_lr_ratio(ep, args.epochs_fusion)
        fusion_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[1]['lr'] = fusion_lr * ratio
        optimizer.param_groups[2]['lr'] = fusion_lr * ratio
        optimizer.param_groups[3]['lr'] = fusion_lr

        loss_avg, acc_avg, weight = train_dkl_epoch(
            fusion, train_loader_10, optimizer, ep, awp_adversary, ema, weight, device
        )
        scheduler.step()

        if not bn_unfrozen and ep >= 40:
            unfreeze_bn(fusion)
            bn_unfrozen = True
            print(f'[INFO] Unfroze BN at epoch {ep}')

        if ema is not None:
            ema.apply_to(fusion)
        val_acc = test(fusion, test_loader, device)
        if ema is not None:
            ema.restore(fusion)
        fusion.train()
        print(f'[DKL][{ep}/{args.epochs_fusion}] loss={loss_avg:.4f} acc={acc_avg:.2f}% val={val_acc*100:.2f}%')

        ckpt = {
            'm4_state_dict': m4.state_dict(),
            'm6_state_dict': m6.state_dict(),
            'model_state_dict': fusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'ema_shadow': ema.shadow if ema is not None else {},
            'fusion_epoch': ep,
        }
        torch.save(ckpt, os.path.join(model_dir, 'checkpoint-last.pt'))

        if ep >= args.save_start and (ep - args.save_start) % args.save_freq == 0:
            if ep == args.epochs_fusion and ema is not None:
                for n, p in fusion.named_parameters():
                    if n in ema.shadow:
                        p.data.copy_(ema.shadow[n])
            torch.save(fusion.state_dict(), os.path.join(model_dir, f'fusion-epoch{ep}.pt'))


if __name__ == '__main__':
    main()
