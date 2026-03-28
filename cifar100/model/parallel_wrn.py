"""
CIFAR-100 Parallel Fusion WRN model.
Reuses WRNWithEmbedding from cifar10; defines ParallelFusionWRN100 for N-group fusion.
"""
import torch
import torch.nn as nn

import os, sys
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from cifar10.model.parallel_wrn import WRNWithEmbedding


class ParallelFusionWRN100(nn.Module):
    def __init__(self, submodels, num_classes=100, freeze_backbone=False):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        if freeze_backbone:
            for m in self.submodels:
                for p in m.parameters():
                    p.requires_grad = False
        emb_dim = sum(m.nChannels for m in self.submodels)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_aux=False):
        embs = []
        aux_outputs = []
        for m in self.submodels:
            emb, out = m(x, return_embedding=True)
            embs.append(emb)
            aux_outputs.append(out)
        combined = torch.cat(embs, dim=1)
        final_out = self.fc(combined)
        if return_aux:
            return aux_outputs, final_out
        return final_out
