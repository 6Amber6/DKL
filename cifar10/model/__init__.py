"""Models for our parallel method (separate from DKL original code)."""
from .wideresnet_update import WideResNet, BasicBlock, NetworkBlock
from .parallel_wrn import WRNWithEmbedding, ParallelFusionWRN, GatedFusionWRN
from .soft_routing_wrn import SoftRoutingFusion, SoftRoutingConfidenceFusion
