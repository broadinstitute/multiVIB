"""
Neural-network building blocks used by multiVIB models.
"""

import math

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
# Encoder building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Post-activation residual block for the variational encoder.

    Applies ``Linear(in_dim, out_dim) → BN → SiLU → Dropout →
    Linear(out_dim, out_dim) → BN``, then adds a skip connection and a
    final ``SiLU``.  When ``in_dim != out_dim`` the skip is a bias-free
    linear projection; otherwise it is an identity shortcut.

    Args:
        in_dim:  Input dimensionality.
        out_dim: Output dimensionality.
        dropout: Dropout probability inside the block.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class VariationalEncoder(nn.Module):
    """
    Variational encoder with progressive narrowing and a residual block.

    Architecture::

        x → Linear(n_input, n_hidden) → Dropout(0.2) → BN → SiLU   [input proj]
          → ResidualBlock(n_hidden, n_hidden // 2)                   [narrowing + residual]
          → mean_encoder / var_encoder  →  Normal(μ, σ)

    The three-stage progressive narrowing
    (``n_input → n_hidden → n_hidden // 2 → n_latent``) softens the
    dimensionality reduction compared to the previous two-layer flat design.
    The residual skip connection stabilises gradient flow and ``SiLU``
    replaces ``LeakyReLU`` throughout for smoother gradients.

    Args:
        n_input:  Dimensionality of the input features.
        n_hidden: Width of the first hidden layer; the residual block maps
                  to ``n_hidden // 2``.
        n_latent: Dimensionality of the latent space.
        var_eps:  Small constant added to the variance for numerical stability.
        dropout:  Dropout rate inside the residual block.
    """

    def __init__(
        self,
        n_input: int = 2000,
        n_hidden: int = 256,
        n_latent: int = 10,
        var_eps: float = 1e-4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.var_eps = var_eps

        n_narrow = n_hidden // 2

        # Initial projection: high-dim input → first hidden width
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(n_hidden),
            nn.SiLU(),
        )
        # Residual block with progressive narrowing: n_hidden → n_narrow
        self.res_block = ResidualBlock(n_hidden, n_narrow, dropout=dropout)

        self.mean_encoder = nn.Linear(n_narrow, n_latent)
        self.var_encoder = nn.Linear(n_narrow, n_latent)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            dist:   Diagonal Normal posterior distribution.
            latent: Reparameterised sample from ``dist``.
        """
        h = self.input_proj(x)
        h = self.res_block(h)
        qm = self.mean_encoder(h)
        qv = torch.exp(self.var_encoder(h)) + self.var_eps
        dist = Normal(qm, qv.sqrt())
        latent = dist.rsample()
        return dist, latent


# ---------------------------------------------------------------------------
# GMM prior
# ---------------------------------------------------------------------------

_LOG_2PI = math.log(2.0 * math.pi)


class GMMPrior(nn.Module):
    """
    Learnable Gaussian Mixture Model prior for the latent space.

    Replaces the standard isotropic Gaussian prior ``p(z) = N(0, I)`` with
    a richer mixture::

        p(z) = Σ_k  π_k · N(z ; μ_k, diag(σ_k²))

    where the mixture weights ``π_k``, component means ``μ_k``, and log
    variances ``log σ_k²`` are all jointly optimised with the model.

    KL divergence is estimated via a single Monte Carlo sample — the
    reparameterised ``z`` returned by the encoder — which is unbiased and
    fully differentiable::

        KL(q(z|x) ∥ p(z)) ≈ log q(z|x) − log p(z)  |_{z ~ q}

    This gives the model the capacity to capture multi-modal posterior
    structure (e.g. distinct cell-type clusters in latent space) that a
    unimodal Gaussian prior cannot represent.

    Args:
        n_components: Number of mixture components ``K``.  A sensible
                      starting point is the expected number of cell types.
        n_latent:     Latent-space dimensionality (must match the encoder).
    """

    def __init__(self, n_components: int = 10, n_latent: int = 10) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_latent = n_latent

        # Unnormalised log mixture weights; softmax gives π_k
        self.log_weights = nn.Parameter(torch.zeros(n_components))
        # Component means initialised near the origin with small spread
        self.means = nn.Parameter(torch.randn(n_components, n_latent) * 0.1)
        # Log variances per component and dimension (initialised at 0 → σ²=1)
        self.log_vars = nn.Parameter(torch.zeros(n_components, n_latent))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute ``log p(z) = log Σ_k π_k N(z ; μ_k, σ_k²)``.

        Args:
            z: Latent samples, shape ``(N, n_latent)``.

        Returns:
            Log probability, shape ``(N,)``.
        """
        log_pi = F.log_softmax(self.log_weights, dim=0)  # (K,)

        # Broadcast: z (N, 1, D), means (1, K, D), log_vars (1, K, D)
        z_exp = z.unsqueeze(1)
        mu    = self.means.unsqueeze(0)
        lv    = self.log_vars.unsqueeze(0)

        # Log N(z ; μ_k, σ_k²) per component, summed over latent dims
        log_gauss = -0.5 * (
            lv + (z_exp - mu).pow(2) / lv.exp() + _LOG_2PI
        ).sum(dim=-1)  # (N, K)

        return torch.logsumexp(log_gauss + log_pi.unsqueeze(0), dim=1)  # (N,)

    def kl_divergence(self, dist: Normal, z: torch.Tensor) -> torch.Tensor:
        """
        Monte Carlo KL estimate: ``E_q[log q(z|x) − log p(z)]``.

        Args:
            dist: Posterior ``Normal`` distribution ``q(z|x)``.
            z:    Reparameterised sample from ``dist``, shape ``(N, n_latent)``.

        Returns:
            Per-sample KL estimate, shape ``(N,)``.
        """
        log_q = dist.log_prob(z).sum(dim=1)  # (N,)
        log_p = self.log_prob(z)              # (N,)
        return log_q - log_p


# ---------------------------------------------------------------------------
# Projector with FiLM batch conditioning
# ---------------------------------------------------------------------------

class FiLMProjector(nn.Module):
    """
    Two-layer contrastive projector with Feature-wise Linear Modulation (FiLM)
    for batch-effect conditioning.

    The previous design concatenated batch covariates directly to the latent
    code: ``Linear(n_latent + n_batch, 64)``.  This forces a single linear
    layer to simultaneously undo batch signal and build a good contrastive
    embedding — two conflicting objectives.

    FiLM decouples them into two explicit stages::

        z  →  BN(z)  →  FiLM: h = BN(z) * (1 + γ(b)) + β(b)   [batch removal]
           →  Linear(n_latent, n_proj_hidden) → BN → SiLU        [expansion]
           →  Linear(n_proj_hidden, n_out)                        [projection]

    The FiLM step is analogous to conditional batch normalisation: ``γ(b)``
    and ``β(b)`` are linear functions of the batch covariates that learn
    per-dataset scale and shift corrections before any contrastive objective
    is applied to the representation.  The wider intermediate layer gives the
    subsequent projection more capacity.

    The FiLM linear layers are **zero-initialised** so that conditioning is
    neutral at the start of training (``γ=0, β=0`` → identity transform on
    ``BN(z)``) and is learned progressively as needed.

    Args:
        n_latent:      Latent dimensionality (encoder output).
        n_batch:       Batch covariate dimensionality.
        n_proj_hidden: Width of the intermediate projector layer.
        n_out:         Output (contrastive embedding) dimensionality.
    """

    def __init__(
        self,
        n_latent: int = 10,
        n_batch: int = 1,
        n_proj_hidden: int = 256,
        n_out: int = 64,
    ) -> None:
        super().__init__()

        # FiLM: batch covariates → per-dimension scale and shift
        self.film_bn    = nn.BatchNorm1d(n_latent)
        self.film_gamma = nn.Linear(n_batch, n_latent)  # scale correction γ(b)
        self.film_beta  = nn.Linear(n_batch, n_latent)  # shift correction β(b)

        # Deeper projector: expand then project to contrastive space
        self.proj = nn.Sequential(
            nn.Linear(n_latent, n_proj_hidden),
            nn.BatchNorm1d(n_proj_hidden),
            nn.SiLU(),
            nn.Linear(n_proj_hidden, n_out),
        )

        # FiLM layers start neutral — zero init so batch conditioning is
        # learned progressively.  Models call reset_film_init() again after
        # apply(init_weights) to restore zeros overwritten by Kaiming init.
        self.reset_film_init()

    def reset_film_init(self) -> None:
        """Zero-initialise FiLM layers so conditioning is neutral at training start."""
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, z: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:     Latent codes, shape ``(N, n_latent)``.
            batch: Batch covariates, shape ``(N, n_batch)``.

        Returns:
            Contrastive embeddings, shape ``(N, n_out)``.
        """
        h = self.film_bn(z)
        h = h * (1.0 + self.film_gamma(batch)) + self.film_beta(batch)
        return self.proj(h)


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
