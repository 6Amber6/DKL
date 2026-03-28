"""
AutoAttack evaluation for DKL Parallel Fusion model on CIFAR-100.
Aligned with DKL eval protocol: raw [0,1] input, preprocess '01'.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from cifar10.model.parallel_wrn import WRNWithEmbedding
from cifar100.model.parallel_wrn import ParallelFusionWRN100

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================
# CIFAR-100 fine -> coarse mapping + groups (must match train)
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
COARSE_GROUPS = {
    "textured_organic": [7, 15, 16, 8, 13],
    "smooth_organic": [1, 0, 11, 12, 14],
    "rigid_manmade": [6, 5, 3, 18, 9],
    "large_structures": [10, 17, 19, 2, 4],
}
GROUP_ORDER = ["textured_organic", "smooth_organic", "rigid_manmade", "large_structures"]


def build_fine_classes_for_group(group_coarse):
    return sorted([i for i, c in enumerate(CIFAR100_FINE_TO_COARSE) if c in group_coarse])


GROUP_FINE = [build_fine_classes_for_group(COARSE_GROUPS[name]) for name in GROUP_ORDER]


def filter_state_dict(state_dict):
    from collections import OrderedDict
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module.0.' in k:
            new_state_dict[k[10:]] = v
        elif 'module.' in k:
            new_state_dict[k.replace('module.', '')] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoAttack eval for DKL Parallel Fusion CIFAR-100')
    parser.add_argument('--checkpoint', type=str, required=True, help='fusion-epoch{N}.pt')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--preprocess', type=str, default='01', choices=['01', 'meanstd', '+-1'])
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--log_path', type=str, default='./log_cifar100.txt')
    parser.add_argument('--version', type=str, default='standard', choices=['standard', 'custom'])
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--sub-depth', type=int, default=28, choices=[28, 34])
    parser.add_argument('--sub-widen', type=int, default=8, choices=[4, 8, 10])
    parser.add_argument('--individual', action='store_true')
    args = parser.parse_args()

    if args.preprocess == '01':
        mean, std = (0, 0, 0), (1, 1, 1)
    elif args.preprocess == 'meanstd':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # Build model (must match training architecture)
    submodels = []
    for i, name in enumerate(GROUP_ORDER):
        m = WRNWithEmbedding(depth=args.sub_depth, widen_factor=args.sub_widen,
                             num_classes=len(GROUP_FINE[i]))
        submodels.append(m)
    net = ParallelFusionWRN100(submodels, num_classes=100, freeze_backbone=False)

    ckpt = filter_state_dict(torch.load(args.checkpoint, map_location=device))
    net.load_state_dict(ckpt, strict=True)
    model = nn.Sequential(Normalize(mean=mean, std=std), net)
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform, download=True)
    test_loader = data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path)

    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    x_test = torch.cat([x for x, _ in test_loader], 0)
    y_test = torch.cat([y for _, y in test_loader], 0)

    if not args.individual:
        adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.batch_size)
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({'adv_complete': adv_complete},
                   os.path.join(args.save_dir, 'aa_{}_n{}_eps_{:.5f}.pth'.format(args.version, args.n_ex, args.epsilon)))
    else:
        adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=args.batch_size)

    print(f'Evaluation complete. Log: {args.log_path}')
