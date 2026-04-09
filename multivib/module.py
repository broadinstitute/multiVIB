import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import Normal
from typing import Optional, Tuple, Literal
from torch.utils.data import Dataset, DataLoader


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', 
                                       nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)

class MaskedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, mask_init_value=1.0):
        super().__init__(in_features, out_features, bias)
        mask_shape = (out_features, in_features)
        initial_mask = torch.full(mask_shape, mask_init_value)
        self.register_buffer('mask', initial_mask)
        
    def set_mask(self, mask):
        if self.mask.shape != mask.shape:
            raise ValueError(f"Mask shape mismatch. Expected {self.mask.shape}, got {mask.shape}")
        self.mask.data = mask.data.to(self.mask.device, self.mask.dtype)

    def get_masked_weight(self):
        return self.weight * self.mask

    def forward(self, x):
        # Apply mask to weights before the linear operation
        masked_weight = self.get_masked_weight()
        return F.linear(x, masked_weight, self.bias)
    
class LoRALinear(torch.nn.Module):
    def __init__(self, 
                 in_dim, out_dim, 
                 rank,
                 dropout=0.0, use_bias=False):
        
        super(LoRALinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        
        # LoRA A and B matrices
        self.lora_a = torch.nn.Linear(in_dim, rank, bias=False)
        self.lora_b = torch.nn.Linear(rank, out_dim, bias=False)
        self.batchnorm = torch.nn.BatchNorm1d(out_dim)
        
        bias = torch.zeros(out_dim)
        self.register_buffer('bias', bias)

        # Initialization of LoRA matrices
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, 
                                       mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.lora_b.weight, 
                                       mode='fan_in', nonlinearity='relu')
        
        torch.nn.init.normal_(self.batchnorm.weight, 1.0, 0.02)
        torch.nn.init.zeros_(self.batchnorm.bias)
        
    def forward(self, x):
        
        # LoRA forward pass
        lora_output = self.lora_a(x)
        lora_output = self.lora_b(lora_output)
        lora_output = lora_output + self.bias # * (self.alpha / self.rank) 
        lora_output = self.batchnorm(lora_output)

        return lora_output

class VariationalEncoder(torch.nn.Module):
    def __init__(self, n_input=2000,
                 n_hidden=128, 
                 n_latent=10,
                 var_eps=1e-4):
        super(VariationalEncoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.var_eps = var_eps
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_input, n_hidden),
            torch.nn.Dropout(p=0.2),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
            # torch.nn.ReLU(),
            
            torch.nn.Linear(n_hidden, n_hidden),
            torch.nn.Dropout(p=0.1),
            torch.nn.BatchNorm1d(n_hidden),
            torch.nn.LeakyReLU(0.1),
            # torch.nn.ReLU(),
        )
        self.mean_encoder = torch.nn.Linear(n_hidden, n_latent)
        self.var_encoder = torch.nn.Linear(n_hidden, n_latent)
        
    def forward(self, x):
        q = self.encoder(x)
        qm = self.mean_encoder(q)
        qv = torch.exp(self.var_encoder(q)) + self.var_eps
        dist = Normal(qm, qv.sqrt())
        latent = dist.rsample()
        return dist, latent
    
class CellTypeClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

import numpy as np
SMALL_NUM = np.log(1e-45)
class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.1, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        cross_view_distance = torch.mm(z1, z2.t())
        
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        
        return (positive_loss + negative_loss).mean()

class NeighborhoodAugmenter(torch.nn.Module):
    def __init__(self, mode='swap', mix_ratio=0.8, k_neighbors=3):
        """
        Args:
            mode: 'swap' (discrete feature crossover) or 'mixup' (linear interpolation).
            mix_ratio: The fraction of original genes to keep (0.8 = 80% original, 20% neighbor).
            k_neighbors: Pool of nearest neighbors to randomly sample from. 
                         K>1 adds more variance to the augmentations.
        """
        super().__init__()
        assert mode in ['swap', 'mixup'], "Mode must be 'swap' or 'mixup'"
        self.mode = mode
        self.mix_ratio = mix_ratio
        self.k_neighbors = k_neighbors

    def forward(self, x, latent=None):
        """
        Args:
            x: Log-normalized scRNA-seq batch [Batch, Genes].
            latent: Optional [Batch, Latent_Dim]. If provided, distances are calculated 
                    in the latent space (highly recommended for scRNA-seq). 
                    If None, distances are calculated in the raw gene space.
        Returns:
            x_aug: Augmented gene expression profile.
        """
        device = x.device
        batch_size = x.size(0)

        # Safety check for small batches
        k = min(self.k_neighbors, batch_size - 1)
        if k == 0:
            return x # Cannot do neighborhood mixup with a batch size of 1

        # 1. Determine the space to compute neighbors
        # We detach 'h' so gradients don't flow backward through the neighbor selection process
        h = latent.detach() if latent is not None else x.detach()

        # 2. Compute Pairwise Cosine Similarity
        h_norm = F.normalize(h, dim=1)
        sim = torch.matmul(h_norm, h_norm.T)

        # Mask out self-similarity (a cell cannot be its own neighbor)
        mask_self = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask_self, -9e15)

        # 3. Find top-K neighbors
        # topk_idx shape: [Batch, K]
        _, topk_idx = torch.topk(sim, k=k, dim=1)

        # 4. Randomly pick ONE neighbor from the top-K for each cell
        # This adds stochasticity: View 1 and View 2 will likely use different neighbors
        rand_idx = torch.randint(0, k, (batch_size,), device=device)
        
        # Gather the actual indices of the chosen neighbors
        # (Uses advanced indexing to get one specific neighbor per row)
        neighbor_idx = topk_idx[torch.arange(batch_size), rand_idx]

        # Fetch the gene expression profiles of the chosen neighbors
        x_neighbor = x[neighbor_idx]

        # 5. Apply the Augmentation
        if self.mode == 'swap':
            # Create a boolean mask where `True` means "keep original gene"
            # Probability of keeping original gene = mix_ratio
            keep_mask = (torch.rand(x.shape, device=device) < self.mix_ratio).float()
            
            # Swap genes: (Mask * Original) + (Inverse_Mask * Neighbor)
            x_aug = keep_mask * x + (1.0 - keep_mask) * x_neighbor

        elif self.mode == 'mixup':
            # Linear interpolation (blending)
            x_aug = self.mix_ratio * x + (1.0 - self.mix_ratio) * x_neighbor

        return x_aug
    
    
## extra loss functions
# ---------------------------------------------------------------------------
# 1.  Online Prototype Clustering  (Sinkhorn assignment, no labels needed)
# ---------------------------------------------------------------------------
 
class OnlinePrototypeClustering(nn.Module):
    """
    Maintains K prototype vectors and assigns batch samples to them via a
    Sinkhorn-balanced soft assignment.
 
    The Sinkhorn step enforces a near-uniform assignment across prototypes,
    preventing mode collapse where every sample falls into one cluster.
 
    This produces pseudo-labels (argmax of assignment) that can be fed to
    any class-stratified loss function.
 
    Args
    ----
    n_prototypes : int
        Number of cluster prototypes K.  A good starting range is 10–100.
        Use ~sqrt(expected_n_classes) × 3 if you have a rough estimate.
    latent_dim : int
        Dimensionality of the latent space.
    sinkhorn_eps : float
        Entropy regularisation for the Sinkhorn assignment.  Larger → softer
        (more uniform) assignments.  Typical: 0.05–0.5.
    sinkhorn_iters : int
        Number of Sinkhorn iterations for prototype assignment.
    momentum : float
        EMA momentum for prototype update.  Prototypes are updated as a
        running mean of assigned embeddings, not via gradient descent.
        Lower → faster adaptation (noisier); higher → more stable.
    """
 
    def __init__(
        self,
        n_prototypes: int = 32,
        latent_dim: int = 64,
        sinkhorn_eps: float = 0.05,
        sinkhorn_iters: int = 10,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.K = n_prototypes
        self.eps = sinkhorn_eps
        self.n_iters = sinkhorn_iters
        self.momentum = momentum
 
        # Learnable prototypes (L2-normalised after each update)
        protos = F.normalize(torch.randn(n_prototypes, latent_dim), dim=-1)
        self.register_buffer("prototypes", protos)
 
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sinkhorn_assignment(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute balanced soft assignment Q from raw similarity scores.
 
        Solves:   Q* = argmin_Q <Q, -scores/eps> + H(Q)
                  s.t. Q 1_K = 1/n,  Q^T 1_n = 1/K
 
        where H(Q) is the negative entropy (entropic regulariser).
        The doubly-stochastic Q is computed via Sinkhorn iterations in
        log-space.
 
        Args
        ----
        scores : (n, K) raw dot-product similarities to prototypes
 
        Returns
        -------
        Q : (n, K) soft assignment, rows sum to 1/n
        """
        n, K = scores.shape
        Q = (scores / self.eps).exp().T   # (K, n)  — work in transposed form
 
        for _ in range(self.n_iters):
            # Normalise columns (over K)  →  each sample sums to 1
            Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-8)
            Q /= K
            # Normalise rows (over n)    →  each prototype sums to 1
            Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
            Q /= n
 
        Q = (Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8)).T  # (n, K)
        return Q
 
    # ------------------------------------------------------------------
    @torch.no_grad()
    def assign(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign a batch of latent codes to prototypes.
 
        Args
        ----
        z : (n, d)  L2-normalised latent codes
 
        Returns
        -------
        pseudo_labels : (n,) long  — argmax cluster index per sample
        soft_assign   : (n, K)    — full soft assignment matrix
        """
        z_n = F.normalize(z.detach(), dim=-1)
        scores = z_n @ self.prototypes.T           # (n, K)
        Q = self._sinkhorn_assignment(scores)      # (n, K) balanced
        pseudo_labels = Q.argmax(dim=-1)           # (n,)
        return pseudo_labels, Q
 
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_prototypes(
        self,
        z: torch.Tensor,
        soft_assign: torch.Tensor,
    ) -> None:
        """
        EMA update of prototypes using the soft assignment as weights.
 
        new_proto[k] = normalise( momentum * proto[k]
                                  + (1 - momentum) * weighted_mean(z, Q[:,k]) )
 
        Args
        ----
        z           : (n, d) latent codes (detached)
        soft_assign : (n, K) soft assignment from `assign()`
        """
        z_n = F.normalize(z.detach(), dim=-1)
        # Weighted sum of embeddings per prototype
        # soft_assign.T : (K, n)  @  z_n : (n, d)  →  (K, d)
        weighted_sum = soft_assign.T @ z_n         # (K, d)
        weight_sum   = soft_assign.sum(dim=0)      # (K,)
        new_protos   = weighted_sum / weight_sum.unsqueeze(-1).clamp(min=1e-8)
        # EMA
        updated = (
            self.momentum * self.prototypes
            + (1 - self.momentum) * F.normalize(new_protos, dim=-1)
        )
        self.prototypes = F.normalize(updated, dim=-1)
 
    # ------------------------------------------------------------------
    def forward(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assign OOD latent codes from both modalities to prototypes and
        update prototype vectors.
 
        Args
        ----
        z_A : (n, d) OOD latents from modality A
        z_B : (m, d) OOD latents from modality B
 
        Returns
        -------
        labels_A : (n,) long pseudo-labels for A
        labels_B : (m,) long pseudo-labels for B
        """
        labels_A, Q_A = self.assign(z_A)
        labels_B, Q_B = self.assign(z_B)
 
        # Update prototypes jointly using both modalities
        z_all = torch.cat([z_A, z_B], dim=0)
        Q_all = torch.cat([Q_A, Q_B], dim=0)
        self.update_prototypes(z_all, Q_all)
 
        return labels_A, labels_B
 
# ---------------------------------------------------------------------------
# 2.  Sinkhorn OT Loss  (same as before; stratified by pseudo-labels)
# ---------------------------------------------------------------------------
 
class SinkhornOTLoss(nn.Module):
    """
    Entropy-regularised OT loss, stratified by pseudo-labels from
    OnlinePrototypeClustering.  When labels=None falls back to global OT.
    """
 
    def __init__(
        self,
        eps: float = 0.05,
        n_iters: int = 30,
        cost: str = "sqeuclidean",
    ):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters
        self.cost = cost
 
    def _cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.cost == "sqeuclidean":
            x2 = (x ** 2).sum(-1, keepdim=True)
            y2 = (y ** 2).sum(-1, keepdim=True).T
            return (x2 + y2 - 2.0 * x @ y.T).clamp(min=0.0)
        elif self.cost == "cosine":
            return 1.0 - F.normalize(x, dim=-1) @ F.normalize(y, dim=-1).T
        raise ValueError(f"Unknown cost: {self.cost!r}")
 
    def _sinkhorn(self, C: torch.Tensor) -> torch.Tensor:
        n, m   = C.shape
        device = C.device
        log_a  = torch.full((n,), -torch.log(torch.tensor(float(n))), device=device)
        log_b  = torch.full((m,), -torch.log(torch.tensor(float(m))), device=device)
        log_u  = torch.zeros(n, device=device)
        log_v  = torch.zeros(m, device=device)
        M      = -C / self.eps
 
        for _ in range(self.n_iters):
            log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(M + log_u.unsqueeze(1), dim=0)
 
        log_T = M + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        return (log_T.exp() * C).sum()
 
    def forward(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor,
        labels_A: Optional[torch.Tensor] = None,
        labels_B: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels_A is None or labels_B is None:
            return self._sinkhorn(self._cost_matrix(z_A, z_B))
 
        classes = labels_A.unique()
        costs   = []
        for c in classes:
            mA = (labels_A == c).nonzero(as_tuple=True)[0]
            mB = (labels_B == c).nonzero(as_tuple=True)[0]
            if mA.numel() == 0 or mB.numel() == 0:
                continue
            costs.append(self._sinkhorn(self._cost_matrix(z_A[mA], z_B[mB])))
 
        return torch.stack(costs).mean() if costs else z_A.new_tensor(0.0)
 
# ---------------------------------------------------------------------------
# 3.  Combined Label-Free OOD Alignment Loss
# ---------------------------------------------------------------------------
 
class OODAlignmentLoss(nn.Module):
    """
    Drop-in replacement for the labelled OOD alignment losses.
 
    Combines:
      (a) Prototype clustering  →  pseudo-labels for stratified Sinkhorn OT
      (b) Mutual NN filtering   →  conservative pseudo-pair contrastive loss
 
    Both paths work without any class information.
 
    Args
    ----
    n_prototypes          : int   — number of latent space prototypes
    latent_dim            : int   — latent space dimensionality
    sinkhorn_eps          : float — OT entropic regularisation
    mnn_temperature       : float — temperature for MNN contrastive loss
    mnn_confidence        : float — cosine sim threshold for MNN filtering
    ot_weight             : float — relative weight of OT loss
    pseudo_weight         : float — relative weight of MNN pseudo-pair loss
    cluster_momentum      : float — EMA decay for prototype update
    use_pseudo_labels_for_ot : bool
        If True, pseudo-labels from clustering are used to stratify OT.
        If False, global (unstratified) OT is used — faster but less precise.
    """
 
    def __init__(
        self,
        n_prototypes: int = 32,
        latent_dim: int = 64,
        sinkhorn_eps: float = 0.05,
        ot_weight: float = 1.0,
        # pseudo_weight: float = 1.0,
        cluster_momentum: float = 0.99,
        use_pseudo_labels_for_ot: bool = True,
    ):
        super().__init__()
 
        self.ot_weight = ot_weight
        # self.pseudo_weight = pseudo_weight
        self.use_pseudo_labels_for_ot = use_pseudo_labels_for_ot
 
        self.clusterer  = OnlinePrototypeClustering(
            n_prototypes=n_prototypes,
            latent_dim=latent_dim,
            sinkhorn_eps=sinkhorn_eps,
            momentum=cluster_momentum,
        )
        self.ot_loss    = SinkhornOTLoss(eps=sinkhorn_eps)
 
    # ------------------------------------------------------------------
    def forward(
        self,
        z_A: torch.Tensor,
        z_B: torch.Tensor,
        # alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args
        ----
        z_A   : (n, d) OOD latent codes from modality A
        z_B   : (m, d) OOD latent codes from modality B
        alpha : float  annealing weight for MNN pseudo-pair loss
 
        Returns
        -------
        total_loss : scalar
        info       : dict with individual loss values and diagnostics
        """
 
        # ---- Sinkhorn OT (stratified by pseudo-labels if requested) ----
        if self.use_pseudo_labels_for_ot:
            # ---- Pseudo-label generation via prototype clustering ----
            labels_A, labels_B = self.clusterer(z_A, z_B)
            ot = self.ot_loss(z_A, z_B, labels_A=labels_A, labels_B=labels_B)
        else:
            ot = self.ot_loss(z_A, z_B)
 
        # ---- MNN pseudo-pair loss ----
        
        total = self.ot_weight * ot # + self.pseudo_weight * mnn
 
        info = {
            "ot_loss":    ot.item(),
        }
        return total, info

# ──────────────────────────────────────────────────────────────────────────────
# 4. GRAPH-BASED NEIGHBORHOOD REGULARIZATION
# ──────────────────────────────────────────────────────────────────────────────

class GraphNeighborhoodReg(nn.Module):
    """
    Discriminative graph regularization for unsupervised multi-modal integration.
 
    Args:
        k:
            Number of nearest neighbors for the RNA KNN graph.
        temp:
            Temperature for softmax over neighbor similarities.
            Lower → harder (sparser) neighbor assignment.
        alignment_loss:
            'mse' or 'kl'. Controls how A_atac is aligned to A_rna.
            Set weight_alignment=0.0 to disable entirely.
        contrastive_margin:
            Margin γ for the repulsion hinge: max(0, γ − d(i,j)) for non-neighbors.
            Controls how far non-neighbors need to be pushed. Typical: 0.5–1.0.
        n_negative_samples:
            Number of non-neighbor negatives to sample per cell per step.
            Sampling avoids the O(N²) cost of using all non-neighbors.
        ppr_alpha:
            Teleport probability for PPR diffusion (higher → stays more local).
            Only used when weight_diffusion > 0.
        ppr_iterations:
            Number of power-iteration steps for approximate PPR.
        detach_rna:
            If True, RNA graph is treated as a fixed target — no gradients
            flow through z_rna. Almost always True in practice.
        weight_alignment:
            Loss weight for adjacency alignment (component 1).
        weight_contrastive:
            Loss weight for attraction + repulsion (component 2).
        weight_laplacian:
            Loss weight for Laplacian smoothing (component 3).
        weight_diffusion:
            Loss weight for PPR diffusion alignment (component 4).
            Set to 0.0 to disable PPR (saves compute).
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
    ):
        super().__init__()
        assert alignment_loss in ("mse", "kl")
        self.k                  = k
        self.temp               = temp
        self.alignment_loss     = alignment_loss
        self.contrastive_margin = contrastive_margin
        self.n_negative_samples = n_negative_samples
        self.ppr_alpha          = ppr_alpha
        self.ppr_iterations     = ppr_iterations
        self.detach_rna         = detach_rna
        self.weight_alignment   = weight_alignment
        self.weight_contrastive = weight_contrastive
        self.weight_laplacian   = weight_laplacian
        self.weight_diffusion   = weight_diffusion
 
    # ──────────────────────────────────────────────────────────────────────────
    # Graph construction utilities
    # ──────────────────────────────────────────────────────────────────────────
 
    def _build_knn_graph(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Build a binary KNN adjacency and a soft row-stochastic adjacency.
 
        Args:
            z: [N, D] L2-normalized embeddings
        Returns:
            A_binary: [N, N] bool — True where cell j is a KNN neighbor of i
            A_soft:   [N, N] float — row-stochastic soft adjacency via softmax
        """
        N      = z.size(0)
        device = z.device
        k      = min(self.k, N - 1)
 
        sim      = z @ z.T                                          # [N, N]
        diag     = torch.eye(N, dtype=torch.bool, device=device)
        sim      = sim.masked_fill(diag, float("-inf"))
 
        _, topk_idx = sim.topk(k, dim=1)                           # [N, k]
        A_binary    = torch.zeros(N, N, dtype=torch.bool, device=device)
        A_binary.scatter_(1, topk_idx, True)
 
        # Soft adjacency: softmax over top-k entries, -inf elsewhere
        sim_topk = sim.masked_fill(~A_binary, float("-inf"))
        A_soft   = F.softmax(sim_topk / self.temp, dim=1)          # [N, N]
 
        return A_binary, A_soft
 
    def _normalized_laplacian(self, A_binary: Tensor) -> Tensor:
        """
        Compute the symmetric normalized graph Laplacian from a binary adjacency.
 
            L = I − D^{-1/2} · A · D^{-1/2}
 
        Args:
            A_binary: [N, N] bool adjacency (will be cast to float)
        Returns:
            L: [N, N] normalized Laplacian
        """
        A   = A_binary.float()
        deg = A.sum(dim=1).clamp(min=1.0)                          # [N]
        d   = deg.pow(-0.5)
        D   = torch.diag(d)                                        # [N, N]
        A_norm = D @ A @ D                                         # [N, N]
        L = torch.eye(A.size(0), device=A.device) - A_norm
        return L
 
    def _ppr_diffusion(self, A_soft: Tensor) -> Tensor:
        """
        Approximate Personalized PageRank via power iteration.
 
            PPR = α·(I − (1−α)·A_row)^{-1}
            ≈ iterated: S ← (1−α)·A_row·S + α·I
 
        Starting from S = A_soft, each iteration propagates neighborhood
        information one additional hop. The result is a smoother adjacency
        that captures transitive cluster membership.
 
        Args:
            A_soft: [N, N] row-stochastic soft adjacency
        Returns:
            S: [N, N] diffused adjacency (each row still sums ≈ 1)
        """
        alpha = self.ppr_alpha
        S     = A_soft.clone()
        for _ in range(self.ppr_iterations):
            S = (1 - alpha) * (A_soft @ S) + alpha * A_soft
        # Re-normalize rows for numerical stability
        row_sums = S.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return S / row_sums
 
    # ──────────────────────────────────────────────────────────────────────────
    # Loss components
    # ──────────────────────────────────────────────────────────────────────────
 
    def _adjacency_alignment_loss(
        self, A_atac_soft: Tensor, A_rna_soft: Tensor
    ) -> Tensor:
        """
        Component 1: Align soft ATAC adjacency to soft RNA adjacency.
        Provides coarse modal alignment but no explicit discriminability.
        """
        if self.alignment_loss == "mse":
            return F.mse_loss(A_atac_soft, A_rna_soft)
        else:
            return F.kl_div(
                (A_atac_soft + 1e-8).log(),
                A_rna_soft,
                reduction="batchmean",
            )
 
    def _contrastive_loss(
        self,
        z_atac: Tensor,
        A_rna_binary: Tensor,
    ) -> Tensor:
        """
        Component 2: Graph-contrastive attraction + repulsion.
 
        Uses the RNA KNN graph as an unsupervised oracle:
          • Attraction: for each cell i, pull its ATAC embedding toward each
            of its RNA-graph neighbors j ∈ N(i).
            Loss_attr = mean over (i,j)∈E of (1 - cos_sim(z_i, z_j))
 
          • Repulsion: for each cell i, sample n_negative_samples non-neighbors
            k ∉ N(i) and push their ATAC embeddings apart with a hinge loss.
            Loss_rep  = mean over (i,k)∉E of max(0, γ − d_cos(z_i, z_k))
 
        The key insight: cos distance is bounded in [0,2], so the hinge margin γ
        in (0,1) is a natural discriminability target. At γ=0.5, non-neighbors
        need to be at least 0.5 cosine distance apart (roughly 60° apart on the
        unit hypersphere).
 
        Args:
            z_atac:      [N, D] L2-normalized ATAC embeddings
            A_rna_binary:[N, N] bool — RNA KNN adjacency
        Returns:
            scalar loss
        """
        N      = z_atac.size(0)
        device = z_atac.device
 
        # Full pairwise cosine similarity [N, N]
        cos_sim  = z_atac @ z_atac.T                               # [N, N] ∈ [-1,1]
        cos_dist = 1.0 - cos_sim                                   # [N, N] ∈ [0, 2]
 
        # ── Attraction ────────────────────────────────────────────────────────
        # Pull RNA-graph neighbors together in ATAC space.
        # Only average over cells that actually have neighbors.
        neighbor_dist = cos_dist * A_rna_binary.float()            # [N, N]
        n_neighbors   = A_rna_binary.float().sum(dim=1).clamp(min=1)
        attr_loss     = (neighbor_dist.sum(dim=1) / n_neighbors).mean()
 
        # ── Repulsion ─────────────────────────────────────────────────────────
        # For each cell, sample non-neighbors and apply a hinge loss.
        # Non-neighbor mask: not a neighbor AND not self.
        diag         = torch.eye(N, dtype=torch.bool, device=device)
        non_neighbor = ~A_rna_binary & ~diag                       # [N, N]
 
        # Clamp n_negative_samples to the actual number of non-neighbors
        n_neg      = min(self.n_negative_samples, non_neighbor.sum(dim=1).min().item())
        n_neg      = max(n_neg, 1)
 
        # Sample n_neg non-neighbors per cell using masked sampling
        # Add small uniform noise to non-neighbor mask for random selection
        noise      = torch.rand(N, N, device=device)
        noise      = noise.masked_fill(~non_neighbor, -1.0)
        _, neg_idx = noise.topk(n_neg, dim=1)                      # [N, n_neg]
 
        neg_dist   = cos_dist.gather(1, neg_idx)                   # [N, n_neg]
        hinge      = F.relu(self.contrastive_margin - neg_dist)    # [N, n_neg]
        rep_loss   = hinge.mean()
 
        return attr_loss + rep_loss
 
    def _laplacian_loss(
        self, z_atac: Tensor, L_rna: Tensor
    ) -> Tensor:
        """
        Component 3: Graph Laplacian smoothing.
 
        Minimizes trace(Z_atac^T · L_rna · Z_atac) / N.
 
        Geometric meaning: for each edge (i,j) in the RNA graph, penalize
        || z_atac_i − z_atac_j ||² proportionally to the normalized edge weight.
        This encourages ATAC embeddings that are connected in the RNA graph
        (same cluster) to lie close together — directly enforcing intra-cluster
        compactness without needing labels.
 
        Note: unlike attraction loss, Laplacian smoothing is a global spectral
        constraint — it operates on the full graph structure simultaneously,
        not just pairwise.
 
        Args:
            z_atac: [N, D] ATAC embeddings (need not be normalized)
            L_rna:  [N, N] normalized graph Laplacian from RNA KNN graph
        Returns:
            scalar loss
        """
        # trace(Z^T L Z) = sum_ij L_ij * (z_i · z_j)
        # Equivalent to || D^{-1/2}(z_i - z_j) ||² summed over edges
        return torch.trace(z_atac.T @ L_rna @ z_atac) / z_atac.size(0)
 
    def _diffusion_alignment_loss(
        self, A_atac_soft: Tensor, A_rna_ppr: Tensor
    ) -> Tensor:
        """
        Component 4: Align ATAC adjacency to PPR-diffused RNA adjacency.
 
        PPR propagates neighborhood structure transitively — two cells that
        share many common neighbors in the RNA graph will have high PPR
        affinity even if they're not direct KNN neighbors. Using this as the
        target makes the alignment objective sensitive to cluster-level
        structure rather than just immediate neighbors.
 
        Args:
            A_atac_soft: [N, N] soft ATAC adjacency
            A_rna_ppr:   [N, N] PPR-diffused RNA adjacency
        Returns:
            scalar loss
        """
        return F.mse_loss(A_atac_soft, A_rna_ppr)
 
    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────
 
    def forward(
        self, z_rna: Tensor, z_atac: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Args:
            z_rna:  [N, D] RNA latent embeddings
            z_atac: [N, D] ATAC latent embeddings
        Returns:
            loss:    scalar total loss
            metrics: dict with per-component loss values for logging
        """
        z_rna_in = z_rna.detach() if self.detach_rna else z_rna
        z_rna_n  = F.normalize(z_rna_in,  dim=1)
        z_atac_n = F.normalize(z_atac,    dim=1)
 
        # Build RNA graph — this is the unsupervised oracle
        A_rna_binary, A_rna_soft = self._build_knn_graph(z_rna_n)
 
        # Build ATAC graph for alignment
        _, A_atac_soft = self._build_knn_graph(z_atac_n)
 
        total    = z_rna.new_zeros(())
        metrics  = {}
 
        # 1. Adjacency alignment
        if self.weight_alignment > 0.0:
            L_align = self._adjacency_alignment_loss(A_atac_soft, A_rna_soft)
            total   = total + self.weight_alignment * L_align
            metrics["align"] = L_align.item()
 
        # 2. Graph-contrastive attraction + repulsion
        if self.weight_contrastive > 0.0:
            L_con   = self._contrastive_loss(z_atac_n, A_rna_binary)
            total   = total + self.weight_contrastive * L_con
            metrics["contrastive"] = L_con.item()
 
        # 3. Laplacian smoothing
        if self.weight_laplacian > 0.0:
            L_rna   = self._normalized_laplacian(A_rna_binary)
            L_lap   = self._laplacian_loss(z_atac, L_rna)
            total   = total + self.weight_laplacian * L_lap
            metrics["laplacian"] = L_lap.item()
 
        # 4. PPR diffusion alignment (optional — set weight_diffusion > 0)
        if self.weight_diffusion > 0.0:
            A_rna_ppr = self._ppr_diffusion(A_rna_soft.detach())
            L_ppr     = self._diffusion_alignment_loss(A_atac_soft, A_rna_ppr)
            total     = total + self.weight_diffusion * L_ppr
            metrics["ppr_diffusion"] = L_ppr.item()
 
        return total # , metrics

# ──────────────────────────────────────────────────────────────────────────────
# 5. Variance-Invariance-Covariance REGULARIZATION
# ──────────────────────────────────────────────────────────────────────────────

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance regularization (Bardes et al. 2022).
    Applied to z_rna and z_atac embeddings independently.

    - Variance:   force std of each latent dim > gamma (prevent dead dims)
    - Covariance: penalize off-diagonal covariance (prevent redundant dims)

    The invariance term (MSE between z_rna and z_atac) is already handled
    by your contrastive loss, so we expose it as optional here.
    """
    def __init__(
        self,
        gamma: float = 1.0,       # target std per dimension
        weight_var: float = 25.0, # scale from original paper — start here
        weight_cov: float = 1.0,
    ):
        super().__init__()
        self.gamma      = gamma
        self.weight_var = weight_var
        self.weight_cov = weight_cov

    def _variance_loss(self, z: Tensor) -> Tensor:
        # z: [N, D]. Penalize dims whose std falls below gamma.
        std = z.std(dim=0)                              # [D]
        return F.relu(self.gamma - std).mean()

    def _covariance_loss(self, z: Tensor) -> Tensor:
        # z: [N, D], zero-centered. Penalize off-diagonal covariance entries.
        N, D  = z.shape
        z     = z - z.mean(dim=0)
        cov   = (z.T @ z) / (N - 1)                    # [D, D]
        diag  = torch.eye(D, device=z.device, dtype=torch.bool)
        off   = cov[~diag].pow(2).sum() / D
        return off

    def forward(self, z_rna: Tensor, z_atac: Tensor) -> tuple[Tensor, dict]:
        var_loss = self._variance_loss(z_rna) + self._variance_loss(z_atac)
        cov_loss = self._covariance_loss(z_rna) + self._covariance_loss(z_atac)
        loss = self.weight_var * var_loss + self.weight_cov * cov_loss
        return loss # , {"vicreg_var": var_loss.item(), "vicreg_cov": cov_loss.item()}
