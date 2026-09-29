"""
Utility functions shared across multiVIB modules.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(m: nn.Module) -> None:
    """Kaiming-uniform init for Linear layers; normal init for BatchNorm.

    Bias initialisation is skipped when ``m.bias`` is ``None`` (e.g. the
    bias-free skip-projection in :class:`~multivib.layers.ResidualBlock`).
    """
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
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


def latent_mixup(z: torch.Tensor, alpha: float = 0.4) -> tuple:
    """
    Beta-distributed Mixup regularisation in latent space.

    Pairs each cell with a randomly chosen other cell in the same batch and
    interpolates their latent representations::

        z_mix = λ · z + (1 − λ) · z[perm]    λ ~ Beta(α, α),  λ ≥ 0.5

    Clamping ``λ ≥ 0.5`` ensures the mixed representation is majority-owned
    by the original cell, preventing severe identity loss.  The interpolation
    encourages the encoder to produce a convex, well-structured latent space:
    points on the line segment between any two valid cells should also be
    valid (i.e. close to one of the two cell types rather than landing in
    empty space).

    Unlike input-space CrossOver, Mixup never produces feature values outside
    the observed data range and cannot destroy marker-gene signal, because the
    mixing happens after the encoder has already computed representations.

    Args:
        z:     Latent embeddings, shape ``(batch_size, n_latent)``.
        alpha: Beta distribution concentration parameter.
               ``alpha = 1.0`` → Uniform(0.5, 1.0) after clamping;
               ``alpha → 0``   → near-identity (effectively no mixing).

    Returns:
        z_mix: Mixed embeddings, same shape as ``z``.
        lam:   The mixing coefficient used (Python float, in [0.5, 1.0]).
    """
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)              # majority from the original cell
    perm = torch.randperm(z.size(0), device=z.device)
    z_mix = lam * z + (1.0 - lam) * z[perm]
    return z_mix, lam


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


# ---------------------------------------------------------------------------
# KL annealing
# ---------------------------------------------------------------------------

def kl_annealing_weight(
    epoch: int,
    total_epochs: int,
    schedule: str = "cyclical",
    n_cycles: int = 4,
    ratio: float = 0.5,
) -> float:
    """
    Compute a ``[0, 1]`` multiplier for the KL loss weight at a given epoch.

    Static KL weighting can cause posterior collapse (weight too high too
    early) or an under-regularised latent (weight never allowed to reach its
    target value). Annealing ramps the KL weight up over training instead.

    Args:
        epoch:        Current epoch (0-indexed).
        total_epochs: Total number of training epochs.
        schedule:     ``"constant"`` (no annealing, always 1.0),
                      ``"monotonic"`` (single linear ramp to 1.0), or
                      ``"cyclical"`` (repeated ramps, Fu et al. 2019
                      "Cyclical Annealing Schedule").
        n_cycles:     Number of ramp cycles (``schedule="cyclical"`` only).
        ratio:        Fraction of each cycle/schedule spent ramping up
                      before holding at 1.0.

    Returns:
        KL weight multiplier in ``[0, 1]``.
    """
    if schedule == "constant" or total_epochs <= 1:
        return 1.0

    if schedule == "monotonic":
        anneal_epochs = max(1, int(total_epochs * ratio))
        return float(min(1.0, epoch / anneal_epochs))

    if schedule == "cyclical":
        period = total_epochs / n_cycles
        pos_in_cycle = (epoch % period) / period
        return float(min(1.0, pos_in_cycle / ratio))

    raise ValueError(f"Unknown KL annealing schedule: {schedule!r}")


# ---------------------------------------------------------------------------
# EMA teacher (for stabilising graph-regularisation targets)
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential-moving-average "teacher" copy of a model.

    Used to provide a stable target (e.g. the RNA KNN graph in
    :class:`~multivib.losses.GraphNeighborhoodReg`) instead of the
    still-training student encoder's own noisy output, following the
    mean-teacher pattern (Tarvainen & Valpola, 2017).

    Args:
        model: Model to shadow. A detached ``deepcopy`` is kept in eval mode.
        decay: EMA decay rate; higher → slower-moving, more stable teacher.
    """

    def __init__(self, model: nn.Module, decay: float = 0.996) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters/buffers towards ``model``'s current state."""
        for s_p, p in zip(self.shadow.parameters(), model.parameters()):
            s_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s_b, b in zip(self.shadow.buffers(), model.buffers()):
            s_b.copy_(b)

    def to(self, device) -> "EMA":
        self.shadow.to(device)
        return self

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            return self.shadow(*args, **kwargs)
