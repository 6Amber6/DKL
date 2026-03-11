"""
Soft Routing WRN: 4-class (vehicle) + 6-class (animal) submodels with unknown class.
Each expert: K known classes + 1 unknown. Soft routing fusion for 10-class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .parallel_wrn import WRNWithEmbedding

VEHICLE_CLASSES = [0, 1, 8, 9]
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]


def map_expert_to_10(logits_local, owned_classes, num_classes=10, margin=0.0):
    """
    logits_local: [B, K+1]  (last column = unknown)
    """
    B = logits_local.size(0)
    K = len(owned_classes)
    known = logits_local[:, :K]
    unk = logits_local[:, K:K+1]
    logits_10 = (unk - margin).expand(B, num_classes).clone()
    for i, c in enumerate(owned_classes):
        logits_10[:, c] = known[:, i]
    return logits_10


def soft_routing_fusion(logits4_local, logits6_local, a=1.0, b=0.5, T=1.0, margin=0.0):
    """
    logits4_local: [B,5]  (4 classes + unk)
    logits6_local: [B,7]  (6 classes + unk)
    """
    p4 = F.softmax(logits4_local, dim=1)
    p6 = F.softmax(logits6_local, dim=1)
    unk4 = p4[:, -1:]
    unk6 = p6[:, -1:]
    conf4 = 1 - unk4
    conf6 = 1 - unk6
    s4 = a * conf4 + b * unk6
    s6 = a * conf6 + b * unk4
    s = torch.cat([s4, s6], dim=1)
    w = F.softmax(s / T, dim=1)
    w4, w6 = w[:, :1], w[:, 1:]
    logits4_10 = map_expert_to_10(logits4_local, VEHICLE_CLASSES, margin=margin)
    logits6_10 = map_expert_to_10(logits6_local, ANIMAL_CLASSES, margin=margin)
    final_logits = w4 * logits4_10 + w6 * logits6_10
    return final_logits, w4, w6


class SoftRoutingFusion(nn.Module):
    def __init__(self, m4, m6, a=1.0, b=0.5, T=1.0, margin=0.0):
        super().__init__()
        self.m4 = m4
        self.m6 = m6
        self.a = a
        self.b = b
        self.T = T
        self.margin = margin

    def forward(self, x, return_aux=False):
        out4 = self.m4(x)
        out6 = self.m6(x)
        logits4 = out4[0] if isinstance(out4, (tuple, list)) else out4
        logits6 = out6[0] if isinstance(out6, (tuple, list)) else out6
        final_logits, _, _ = soft_routing_fusion(
            logits4, logits6, a=self.a, b=self.b, T=self.T, margin=self.margin
        )
        if return_aux:
            return logits4, logits6, final_logits
        return final_logits


def soft_routing_fusion_conf(logits4_local, logits6_local, T=1.0, margin=0.0):
    """
    Confidence-based routing: w = softmax([c4, c6]/T), c4=1-unk4, c6=1-unk6.
    logits4_local: [B,5]  (4 classes + unk)
    logits6_local: [B,7]  (6 classes + unk)
    """
    p4 = F.softmax(logits4_local, dim=1)
    p6 = F.softmax(logits6_local, dim=1)
    unk4 = p4[:, -1:]
    unk6 = p6[:, -1:]
    c4 = 1 - unk4
    c6 = 1 - unk6
    s = torch.cat([c4, c6], dim=1)
    w = F.softmax(s / T, dim=1)
    w4, w6 = w[:, :1], w[:, 1:]
    logits4_10 = map_expert_to_10(logits4_local, VEHICLE_CLASSES, margin=margin)
    logits6_10 = map_expert_to_10(logits6_local, ANIMAL_CLASSES, margin=margin)
    final_logits = w4 * logits4_10 + w6 * logits6_10
    return final_logits, w4, w6


class SoftRoutingConfidenceFusion(nn.Module):
    """Confidence-based soft routing: weight by 1 - P(unknown) per expert."""
    def __init__(self, m4, m6, T=1.0, margin=0.0):
        super().__init__()
        self.m4 = m4
        self.m6 = m6
        self.T = T
        self.margin = margin

    def forward(self, x, return_aux=False):
        out4 = self.m4(x)
        out6 = self.m6(x)
        logits4 = out4[0] if isinstance(out4, (tuple, list)) else out4
        logits6 = out6[0] if isinstance(out6, (tuple, list)) else out6
        final_logits, _, _ = soft_routing_fusion_conf(
            logits4, logits6, T=self.T, margin=self.margin
        )
        if return_aux:
            return logits4, logits6, final_logits
        return final_logits
