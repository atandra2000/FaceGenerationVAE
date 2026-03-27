"""Shared utilities — FaceGenerationVAE"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Running average over streamed values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def denorm(tensor: torch.Tensor) -> torch.Tensor:
    """Clip pixel values to [0, 1] (VAE decoder uses Sigmoid, so already normalised)."""
    return tensor.clamp(0.0, 1.0)
