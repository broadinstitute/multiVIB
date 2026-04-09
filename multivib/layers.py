"""
Neural-network building blocks used by multiVIB models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Custom linear layers
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """
    Linear layer whose effective weight is element-wise multiplied by a
    fixed binary (or soft) mask.

    The mask can encode prior biological knowledge — e.g. a gene-programme
    membership matrix — so that only biologically plausible connections are
    active.

    Args:
        in_features:       Input dimensionality.
        out_features:      Output dimensionality.
        bias:              Whether to add a learnable bias.
        mask_init_value:   Scalar used to fill the initial mask (default 1.0,
                           meaning all connections are open).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init_value: float = 1.0,
    ) -> None:
        super().__init__(in_features, out_features, bias)
        initial_mask = torch.full((out_features, in_features), mask_init_value)
        self.register_buffer("mask", initial_mask)

    def set_mask(self, mask: torch.Tensor) -> None:
        """Replace the buffer with *mask* (shape must match)."""
        if self.mask.shape != mask.shape:
            raise ValueError(
                f"Mask shape mismatch. Expected {self.mask.shape}, got {mask.shape}"
            )
        self.mask.data = mask.data.to(self.mask.device, self.mask.dtype)

    def get_masked_weight(self) -> torch.Tensor:
        return self.weight * self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.get_masked_weight(), self.bias)


class LoRALinear(nn.Module):
    """
    Low-rank adaptation layer: ``output = BN(B(A(x)) + bias)``.

    Parameterises the translator as a low-rank matrix product
    ``W ≈ B · A`` with an optional batch-normalisation step.

    Args:
        in_dim:   Input dimensionality.
        out_dim:  Output dimensionality.
        rank:     Inner rank *r* (``r ≪ min(in_dim, out_dim)``).
        dropout:  Dropout rate (currently unused — reserved for future use).
        use_bias: Unused; kept for API symmetry.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int,
        dropout: float = 0.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        self.lora_a = nn.Linear(in_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_dim, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_dim)

        bias = torch.zeros(out_dim)
        self.register_buffer("bias", bias)

        nn.init.kaiming_uniform_(self.lora_a.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.lora_b.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.batchnorm.weight, 1.0, 0.02)
        nn.init.zeros_(self.batchnorm.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lora_b(self.lora_a(x)) + self.bias
        return self.batchnorm(out)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VariationalEncoder(nn.Module):
    """
    Two-layer MLP variational encoder.

    Maps input features to a diagonal-Gaussian posterior
    ``q(z | x) = N(μ(x), σ²(x))``.

    Args:
        n_input:  Dimensionality of the input features.
        n_hidden: Width of each hidden layer.
        n_latent: Dimensionality of the latent space.
        var_eps:  Small constant added to the variance for numerical stability.
    """

    def __init__(
        self,
        n_input: int = 2000,
        n_hidden: int = 128,
        n_latent: int = 10,
        var_eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.var_eps = var_eps

        self.encoder = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(n_hidden, n_hidden),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(n_hidden),
            nn.LeakyReLU(0.1),
        )
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        self.var_encoder = nn.Linear(n_hidden, n_latent)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            dist:   Diagonal Normal posterior distribution.
            latent: Reparameterised sample from ``dist``.
        """
        q = self.encoder(x)
        qm = self.mean_encoder(q)
        qv = torch.exp(self.var_encoder(q)) + self.var_eps
        dist = Normal(qm, qv.sqrt())
        latent = dist.rsample()
        return dist, latent


# ---------------------------------------------------------------------------
# Auxiliary classifier
# ---------------------------------------------------------------------------

class CellTypeClassifier(nn.Module):
    """
    Two-hidden-layer MLP cell-type classifier.

    Args:
        input_dim:   Dimensionality of the latent embedding input.
        num_classes: Number of cell-type classes to predict.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
