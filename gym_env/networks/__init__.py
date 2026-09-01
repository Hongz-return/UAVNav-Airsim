"""
Mamba-based networks for temporal RL.
Architecture: CNN (shared) -> Mamba (independent) -> MLP Head
"""
from .mamba_extractor import MambaExtractor, TimeDistributedCNN, MambaBlock
from .mamba_policy import MambaSACPolicy, MambaActor, MambaCritic
from .stacked_dict_extractor import FrameStackedDictExtractor

__all__ = [
    # Feature extractor (simple version)
    "MambaExtractor",
    "TimeDistributedCNN",
    "MambaBlock",
    "FrameStackedDictExtractor",
    # Full policy (independent Actor/Critic Mamba)
    "MambaSACPolicy",
    "MambaActor",
    "MambaCritic",
]
