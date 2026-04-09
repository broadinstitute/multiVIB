"""
Utility functions shared across multiVIB modules.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(m: nn.Module) -> None:
    """Kaiming-uniform init for Linear layers; normal init for BatchNorm."""
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def crossover_augmentation(x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    CrossOver augmentation for single-cell gene expression data.

    Randomly swaps ``alpha`` fraction of genes between each cell and a
    uniformly chosen other cell in the same batch.

    Args:
        x:     Gene expression batch, shape ``(batch_size, num_genes)``.
        alpha: Fraction of genes to swap (e.g. 0.1 → 10%).

    Returns:
        Augmented batch with the same shape as ``x``.
    """
    batch_size, num_genes = x.shape
    shuffled = torch.randperm(batch_size, device=x.device)
    x_random = x[shuffled]
    swap_mask = torch.rand((batch_size, num_genes), device=x.device) < alpha
    x_aug = x.clone()
    x_aug[swap_mask] = x_random[swap_mask]
    return x_aug


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One-hot encode a 1-D tensor of category indices."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.float()


def scale_by_batch(x: np.ndarray, batch_label: np.ndarray) -> np.ndarray:
    """
    Z-score normalise ``x`` independently within each batch.

    Args:
        x:           Data array, shape ``(n_cells, n_features)``.
        batch_label: Batch assignment per cell, shape ``(n_cells,)``.

    Returns:
        Scaled array with the same shape as ``x``.
    """
    scaled_x = np.zeros_like(x)
    for b in np.unique(batch_label):
        mask = batch_label == b
        scaled_x[mask] = StandardScaler().fit_transform(x[mask])
    return scaled_x
