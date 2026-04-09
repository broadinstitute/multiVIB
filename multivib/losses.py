"""
Loss functions and regularisers used during multiVIB training.

Modules
-------
DCL                    Decoupled Contrastive Loss (Wu et al., 2021).
OnlinePrototypeClustering
                       Sinkhorn-balanced prototype clustering for pseudo-labels.
SinkhornOTLoss         Entropy-regularised optimal-transport loss.
OODAlignmentLoss       Combined OOD alignment (OT + prototype clustering).
GraphNeighborhoodReg   Discriminative graph neighbourhood regularisation.
VICRegLoss             Variance-Invariance-Covariance regularisation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Literal

SMALL_NUM = np.log(1e-45)


# ---------------------------------------------------------------------------
# 1. Decoupled Contrastive Loss
# ---------------------------------------------------------------------------

class DCL:
    """
    Decoupled Contrastive Loss (DCL).

    Proposed in *Yeh et al., "Decoupled Contrastive Learning" (2022)*
    (https://arxiv.org/pdf/2110.06848.pdf).

    Unlike standard NT-Xent, DCL removes the positive sample from the
    denominator, preventing the loss from inadvertently suppressing the
    embedding of the anchor itself.

    Args:
        temperature: Softmax temperature τ.  Lower → sharper distribution.
        weight_fn:   Optional callable ``(z1, z2) → weights`` that
                     re-scales each positive pair's loss.
    """

    def __init__(self, temperature: float = 0.1, weight_fn=None) -> None:
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        One-way DCL loss from ``z1`` toward ``z2``.

        Args:
            z1: First embedding, shape ``(N, D)``.
            z2: Second embedding, shape ``(N, D)``.

        Returns:
            Scalar loss.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        cross = torch.mm(z1, z2.t())
        pos_loss = -torch.diag(cross) / self.temperature
        if self.weight_fn is not None:
            pos_loss = pos_loss * self.weight_fn(z1, z2)

        neg_sim = torch.cat([torch.mm(z1, z1.t()), cross], dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        neg_loss = torch.logsumexp(neg_sim + neg_mask * SMALL_NUM, dim=1)

        return (pos_loss + neg_loss).mean()


# ---------------------------------------------------------------------------
# 2. Online Prototype Clustering (Sinkhorn assignment)
# ---------------------------------------------------------------------------

class OnlinePrototypeClustering(nn.Module):
    """
    Maintains *K* prototype vectors and assigns batch samples via a
    Sinkhorn-balanced soft assignment.

    The Sinkhorn step enforces a near-uniform assignment across prototypes,
    preventing mode collapse.  Pseudo-labels (argmax of assignment) can be
    used to stratify any class-aware loss downstream.

    Args:
        n_prototypes:   Number of cluster prototypes *K*.
        latent_dim:     Dimensionality of the latent space.
        sinkhorn_eps:   Entropy regularisation ε.  Larger → softer assignments.
        sinkhorn_iters: Number of Sinkhorn iterations.
        momentum:       EMA momentum for prototype update.
    """

    def __init__(
        self,
        n_prototypes: int = 32,
        latent_dim: int = 64,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 10,
        momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.K = n_prototypes
        self.eps = sinkhorn_eps
        self.n_iters = sinkhorn_iters
        self.momentum = momentum

        protos = F.normalize(torch.randn(n_prototypes, latent_dim), dim=-1)
        self.register_buffer("prototypes", protos)

    @torch.no_grad()
    def _sinkhorn_assignment(self, scores: Tensor) -> Tensor:
        n, K = scores.shape
        Q = (scores / self.eps).exp().T  # (K, n)
        for _ in range(self.n_iters):
            Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
            Q /= K
            Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Q /= n
        Q = (Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8)).T  # (n, K)
        return Q

    @torch.no_grad()
    def assign(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Assign latent codes to prototypes; returns pseudo-labels and soft assignments."""
        z_n = F.normalize(z.detach(), dim=-1)
        scores = z_n @ self.prototypes.T
        Q = self._sinkhorn_assignment(scores)
        return Q.argmax(dim=-1), Q

    @torch.no_grad()
    def update_prototypes(self, z: Tensor, soft_assign: Tensor) -> None:
        """EMA update of prototype vectors using soft assignment weights."""
        z_n = F.normalize(z.detach(), dim=-1)
        weighted_sum = soft_assign.T @ z_n
        weight_sum = soft_assign.sum(dim=0)
        new_protos = weighted_sum / weight_sum.unsqueeze(-1).clamp(min=1e-8)
        updated = (
            self.momentum * self.prototypes
            + (1 - self.momentum) * F.normalize(new_protos, dim=-1)
        )
        self.prototypes = F.normalize(updated, dim=-1)

    def forward(self, z_A: Tensor, z_B: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Assign both modalities to prototypes and update prototype vectors.

        Returns:
            Pseudo-labels for modality A and B respectively.
        """
        labels_A, Q_A = self.assign(z_A)
        labels_B, Q_B = self.assign(z_B)
        z_all = torch.cat([z_A, z_B], dim=0)
        Q_all = torch.cat([Q_A, Q_B], dim=0)
        self.update_prototypes(z_all, Q_all)
        return labels_A, labels_B


# ---------------------------------------------------------------------------
# 3. Entropy-Regularised OT Loss
# ---------------------------------------------------------------------------

class SinkhornOTLoss(nn.Module):
    """
    Entropy-regularised optimal-transport loss.

    Optionally stratified by pseudo-labels from
    :class:`OnlinePrototypeClustering` — when ``labels=None`` it falls
    back to global (unstratified) OT.

    Args:
        eps:    Entropic regularisation ε.
        n_iters: Sinkhorn iterations.
        cost:   ``"sqeuclidean"`` or ``"cosine"``.
    """

    def __init__(
        self,
        eps: float = 0.05,
        n_iters: int = 30,
        cost: str = "sqeuclidean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters
        self.cost = cost

    def _cost_matrix(self, x: Tensor, y: Tensor) -> Tensor:
        if self.cost == "sqeuclidean":
            x2 = (x ** 2).sum(-1, keepdim=True)
            y2 = (y ** 2).sum(-1, keepdim=True).T
            return (x2 + y2 - 2.0 * x @ y.T).clamp(min=0.0)
        elif self.cost == "cosine":
            return 1.0 - F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T
        raise ValueError(f"Unknown cost: {self.cost!r}")

    def _sinkhorn(self, C: Tensor) -> Tensor:
        n, m = C.shape
        device = C.device
        log_a = torch.full((n,), -torch.log(torch.tensor(float(n))), device=device)
        log_b = torch.full((m,), -torch.log(torch.tensor(float(m))), device=device)
        log_u = torch.zeros(n, device=device)
        log_v = torch.zeros(m, device=device)
        M = -C / self.eps
        for _ in range(self.n_iters):
            log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(M + log_u.unsqueeze(1), dim=0)
        log_T = M + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        return (log_T.exp() * C).sum()

    def forward(
        self,
        z_A: Tensor,
        z_B: Tensor,
        labels_A: Optional[Tensor] = None,
        labels_B: Optional[Tensor] = None,
    ) -> Tensor:
        if labels_A is None or labels_B is None:
            return self._sinkhorn(self._cost_matrix(z_A, z_B))

        classes = labels_A.unique()
        costs = []
        for c in classes:
            mA = (labels_A == c).nonzero(as_tuple=True)[0]
            mB = (labels_B == c).nonzero(as_tuple=True)[0]
            if mA.numel() == 0 or mB.numel() == 0:
                continue
            costs.append(self._sinkhorn(self._cost_matrix(z_A[mA], z_B[mB])))

        return torch.stack(costs).mean() if costs else z_A.new_tensor(0.0)


# ---------------------------------------------------------------------------
# 4. Combined Label-Free OOD Alignment Loss
# ---------------------------------------------------------------------------

class OODAlignmentLoss(nn.Module):
    """
    Drop-in replacement for labelled OOD alignment losses.

    Combines prototype clustering (→ pseudo-labels) with a stratified
    Sinkhorn OT loss to align samples from different modalities or batches
    without requiring ground-truth cell-type annotations.

    Args:
        n_prototypes:            Number of latent-space prototypes.
        latent_dim:              Latent-space dimensionality.
        sinkhorn_eps:            OT entropic regularisation ε.
        ot_weight:               Weight of the OT loss term.
        cluster_momentum:        EMA decay for prototype update.
        use_pseudo_labels_for_ot:
            If ``True``, stratify OT by prototype pseudo-labels.
            If ``False``, use global (faster) OT.
    """

    def __init__(
        self,
        n_prototypes: int = 32,
        latent_dim: int = 64,
        sinkhorn_eps: float = 0.05,
        ot_weight: float = 1.0,
        cluster_momentum: float = 0.99,
        use_pseudo_labels_for_ot: bool = True,
    ) -> None:
        super().__init__()
        self.ot_weight = ot_weight
        self.use_pseudo_labels_for_ot = use_pseudo_labels_for_ot

        self.clusterer = OnlinePrototypeClustering(
            n_prototypes=n_prototypes,
            latent_dim=latent_dim,
            sinkhorn_eps=sinkhorn_eps,
            momentum=cluster_momentum,
        )
        self.ot_loss = SinkhornOTLoss(eps=sinkhorn_eps)

    def forward(self, z_A: Tensor, z_B: Tensor) -> Tuple[Tensor, dict]:
        """
        Args:
            z_A: OOD latent codes from modality A, shape ``(N, D)``.
            z_B: OOD latent codes from modality B, shape ``(M, D)``.

        Returns:
            total_loss: Scalar.
            info:       Dict with individual loss values for logging.
        """
        if self.use_pseudo_labels_for_ot:
            labels_A, labels_B = self.clusterer(z_A, z_B)
            ot = self.ot_loss(z_A, z_B, labels_A=labels_A, labels_B=labels_B)
        else:
            ot = self.ot_loss(z_A, z_B)

        total = self.ot_weight * ot
        info = {"ot_loss": ot.item()}
        return total, info


# ---------------------------------------------------------------------------
# 5. Graph Neighbourhood Regularisation
# ---------------------------------------------------------------------------

class GraphNeighborhoodReg(nn.Module):
    """
    Discriminative graph regularisation for unsupervised multi-modal
    integration.

    Uses the RNA KNN graph as an unsupervised oracle.  Four optional loss
    components:

    1. **Adjacency alignment** — align the ATAC soft adjacency to RNA.
    2. **Graph-contrastive** — attraction for KNN pairs, repulsion for
       non-neighbours (hinge loss).
    3. **Laplacian smoothing** — penalise ``tr(Z^T L Z) / N``.
    4. **PPR diffusion alignment** — align to Personalised PageRank
       diffused RNA adjacency (captures transitive cluster structure).

    Args:
        k:                   Number of nearest neighbours for the RNA KNN graph.
        temp:                Temperature for soft adjacency softmax.
        alignment_loss:      ``"mse"`` or ``"kl"`` for component 1.
        contrastive_margin:  Hinge margin γ for repulsion (component 2).
        n_negative_samples:  Non-neighbour negatives sampled per cell.
        ppr_alpha:           Teleport probability for PPR diffusion.
        ppr_iterations:      Power-iteration steps for approximate PPR.
        detach_rna:          Treat RNA graph as a fixed target (no gradients).
        weight_alignment:    Loss weight for component 1.
        weight_contrastive:  Loss weight for component 2.
        weight_laplacian:    Loss weight for component 3.
        weight_diffusion:    Loss weight for component 4 (0 = disabled).
    """

    def __init__(
        self,
        k: int = 15,
        temp: float = 0.1,
        alignment_loss: Literal["mse", "kl"] = "kl",
        contrastive_margin: float = 0.5,
        n_negative_samples: int = 32,
        ppr_alpha: float = 0.2,
        ppr_iterations: int = 5,
        detach_rna: bool = True,
        weight_alignment: float = 1.0,
        weight_contrastive: float = 1.0,
        weight_laplacian: float = 0.5,
        weight_diffusion: float = 0.0,
    ) -> None:
        super().__init__()
        assert alignment_loss in ("mse", "kl")
        self.k = k
        self.temp = temp
        self.alignment_loss = alignment_loss
        self.contrastive_margin = contrastive_margin
        self.n_negative_samples = n_negative_samples
        self.ppr_alpha = ppr_alpha
        self.ppr_iterations = ppr_iterations
        self.detach_rna = detach_rna
        self.weight_alignment = weight_alignment
        self.weight_contrastive = weight_contrastive
        self.weight_laplacian = weight_laplacian
        self.weight_diffusion = weight_diffusion

    # ---- graph utilities ---------------------------------------------------

    def _build_knn_graph(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        N, device = z.size(0), z.device
        k = min(self.k, N - 1)
        sim = z @ z.T
        sim = sim.masked_fill(torch.eye(N, dtype=torch.bool, device=device), float("-inf"))
        _, topk_idx = sim.topk(k, dim=1)
        A_binary = torch.zeros(N, N, dtype=torch.bool, device=device)
        A_binary.scatter_(1, topk_idx, True)
        sim_topk = sim.masked_fill(~A_binary, float("-inf"))
        A_soft = F.softmax(sim_topk / self.temp, dim=1)
        return A_binary, A_soft

    def _normalized_laplacian(self, A_binary: Tensor) -> Tensor:
        A = A_binary.float()
        d = A.sum(dim=1).clamp(min=1.0).pow(-0.5)
        D = torch.diag(d)
        return torch.eye(A.size(0), device=A.device) - D @ A @ D

    def _ppr_diffusion(self, A_soft: Tensor) -> Tensor:
        alpha, S = self.ppr_alpha, A_soft.clone()
        for _ in range(self.ppr_iterations):
            S = (1 - alpha) * (A_soft @ S) + alpha * A_soft
        return S / S.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # ---- loss components ---------------------------------------------------

    def _adjacency_alignment_loss(self, A_atac: Tensor, A_rna: Tensor) -> Tensor:
        if self.alignment_loss == "mse":
            return F.mse_loss(A_atac, A_rna)
        return F.kl_div((A_atac + 1e-8).log(), A_rna, reduction="batchmean")

    def _contrastive_loss(self, z_atac: Tensor, A_rna_binary: Tensor) -> Tensor:
        N, device = z_atac.size(0), z_atac.device
        cos_dist = 1.0 - z_atac @ z_atac.T

        neighbor_dist = cos_dist * A_rna_binary.float()
        n_neighbors = A_rna_binary.float().sum(dim=1).clamp(min=1)
        attr_loss = (neighbor_dist.sum(dim=1) / n_neighbors).mean()

        diag = torch.eye(N, dtype=torch.bool, device=device)
        non_neighbor = ~A_rna_binary & ~diag
        n_neg = min(self.n_negative_samples, non_neighbor.sum(dim=1).min().item())
        n_neg = max(int(n_neg), 1)

        noise = torch.rand(N, N, device=device).masked_fill(~non_neighbor, -1.0)
        _, neg_idx = noise.topk(n_neg, dim=1)
        hinge = F.relu(self.contrastive_margin - cos_dist.gather(1, neg_idx))
        rep_loss = hinge.mean()
        return attr_loss + rep_loss

    def _laplacian_loss(self, z_atac: Tensor, L_rna: Tensor) -> Tensor:
        return torch.trace(z_atac.T @ L_rna @ z_atac) / z_atac.size(0)

    def _diffusion_alignment_loss(self, A_atac: Tensor, A_rna_ppr: Tensor) -> Tensor:
        return F.mse_loss(A_atac, A_rna_ppr)

    # ---- forward -----------------------------------------------------------

    def forward(self, z_rna: Tensor, z_atac: Tensor) -> Tensor:
        """
        Args:
            z_rna:  RNA latent embeddings, shape ``(N, D)``.
            z_atac: ATAC latent embeddings, shape ``(N, D)``.

        Returns:
            Scalar total loss.
        """
        z_rna_in = z_rna.detach() if self.detach_rna else z_rna
        z_rna_n = F.normalize(z_rna_in, dim=1)
        z_atac_n = F.normalize(z_atac, dim=1)

        A_rna_binary, A_rna_soft = self._build_knn_graph(z_rna_n)
        _, A_atac_soft = self._build_knn_graph(z_atac_n)

        total = z_rna.new_zeros(())

        if self.weight_alignment > 0.0:
            total = total + self.weight_alignment * self._adjacency_alignment_loss(
                A_atac_soft, A_rna_soft
            )
        if self.weight_contrastive > 0.0:
            total = total + self.weight_contrastive * self._contrastive_loss(
                z_atac_n, A_rna_binary
            )
        if self.weight_laplacian > 0.0:
            L_rna = self._normalized_laplacian(A_rna_binary)
            total = total + self.weight_laplacian * self._laplacian_loss(z_atac, L_rna)
        if self.weight_diffusion > 0.0:
            A_rna_ppr = self._ppr_diffusion(A_rna_soft.detach())
            total = total + self.weight_diffusion * self._diffusion_alignment_loss(
                A_atac_soft, A_rna_ppr
            )

        return total


# ---------------------------------------------------------------------------
# 6. VICReg Loss
# ---------------------------------------------------------------------------

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance regularisation (Bardes et al., 2022).

    Applied to embeddings from two modalities *independently*:

    - **Variance** term: force per-dimension std above ``gamma`` to prevent
      dead (collapsed) latent dimensions.
    - **Covariance** term: penalise off-diagonal covariance entries to
      prevent redundant / correlated dimensions.

    The invariance term (cross-modal MSE) is handled by the contrastive
    loss and is therefore not included here.

    Args:
        gamma:      Target standard deviation per latent dimension.
        weight_var: Weight for the variance term.
        weight_cov: Weight for the covariance term.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        weight_var: float = 25.0,
        weight_cov: float = 1.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight_var = weight_var
        self.weight_cov = weight_cov

    def _variance_loss(self, z: Tensor) -> Tensor:
        return F.relu(self.gamma - z.std(dim=0)).mean()

    def _covariance_loss(self, z: Tensor) -> Tensor:
        N, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (N - 1)
        diag = torch.eye(D, device=z.device, dtype=torch.bool)
        return cov[~diag].pow(2).sum() / D

    def forward(self, z_rna: Tensor, z_atac: Tensor) -> Tensor:
        var_loss = self._variance_loss(z_rna) + self._variance_loss(z_atac)
        cov_loss = self._covariance_loss(z_rna) + self._covariance_loss(z_atac)
        return self.weight_var * var_loss + self.weight_cov * cov_loss
