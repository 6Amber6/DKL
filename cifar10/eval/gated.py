"""
Evaluation for DKL Gated Fusion model on CIFAR-10.
Aligns with DKL original auto_attacks/eval.py: AutoAttack, preprocess '01', same args.
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
from cifar10.model.parallel_wrn import WRNWithEmbedding, GatedFusionWRN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def filter_state_dict(state_dict):
    """Align with DKL eval.py: handle state_dict/model_state_dict, strip module/module.0, skip sub_block."""
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
    parser = argparse.ArgumentParser(description='Eval DKL Gated Fusion, same protocol as DKL eval.py')
    parser.add_argument('--checkpoint', type=str, required=True, help='fusion-epoch{N}.pt')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10'])
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--preprocess', type=str, default='01', choices=['01', 'meanstd', '+-1'])
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--log_path', type=str, default='./log.txt')
    parser.add_argument('--version', type=str, default='standard', choices=['standard', 'custom'])
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    parser.add_argument('--individual', action='store_true', help='run attacks individually')
    args = parser.parse_args()

    if args.preprocess == '01':
        mean, std = (0, 0, 0), (1, 1, 1)
    elif args.preprocess == 'meanstd':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    m4 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=4)
    m6 = WRNWithEmbedding(depth=34, widen_factor=10, num_classes=6)
    net = GatedFusionWRN(m4, m6)

    ckpt = filter_state_dict(torch.load(args.checkpoint, map_location=device))
    net.load_state_dict(ckpt, strict=True)
    model = nn.Sequential(Normalize(mean=mean, std=std), net)
    model.to(device)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    testset = getattr(datasets, args.data)(root=args.data_dir, train=False, transform=transform, download=True)
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
