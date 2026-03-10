"""
Parallel WRN: 4-class (vehicle) + 6-class (animal) submodels with fusion for 10-class CIFAR-10.
Structure: m4(x) -> emb4, logits4; m6(x) -> emb6, logits6; concat(emb4, emb6) -> fc -> 10-class. return_aux returns (logits4, logits6, logits10).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wideresnet_update import WideResNet


class WRNWithEmbedding(WideResNet):
    def __init__(self, depth=34, widen_factor=10, num_classes=4):
        super().__init__(depth=depth, widen_factor=widen_factor, num_classes=num_classes)

    def forward(self, x, return_embedding=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        emb = out.view(out.size(0), -1)
        logits = self.fc(emb)
        if return_embedding:
            return emb, logits
        return logits


class ParallelFusionWRN(nn.Module):
    def __init__(self, model4, model6):
        super().__init__()
        self.m4 = model4
        self.m6 = model6

        for p in self.m4.parameters():
            p.requires_grad = False
        for p in self.m6.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(640 * 2, 10)

    def forward(self, x, return_aux=False):
        e4, out4 = self.m4(x, return_embedding=True)
        e6, out6 = self.m6(x, return_embedding=True)
        emb = torch.cat([e4, e6], dim=1)
        out = self.fc(emb)
        if return_aux:
            return out4, out6, out
        return out


class GatedFusionWRN(nn.Module):
    """Gated fusion: emb = alpha*e4 + (1-alpha)*e6, fc(emb)->10. alpha = sigmoid(gate(concat(e4,e6)))."""
    def __init__(self, m4, m6, emb_dim=640, hidden_dim=256):
        super().__init__()
        self.m4 = m4
        self.m6 = m6
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(emb_dim, 10)

    def forward(self, x, return_aux=False):
        e4, out4 = self.m4(x, return_embedding=True)
        e6, out6 = self.m6(x, return_embedding=True)
        gate_in = torch.cat([e4, e6], dim=1)
        alpha = torch.sigmoid(self.gate(gate_in))
        emb = alpha * e4 + (1 - alpha) * e6
        out = self.fc(emb)
        if return_aux:
            return out4, out6, out
        return out
