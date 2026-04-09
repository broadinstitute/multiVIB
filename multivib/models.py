"""
multiVIB model backbone classes.

Models
------
multivib        Bi-modal backbone with MaskedLinear translator
                (vertical / horizontal integration).
multivibLoRA    Bi-modal backbone with LoRA translator.
multivibS       Multi-species backbone with per-species MaskedLinear translators.
multivibLoRAS   Multi-species backbone with shared low-rank translators.
multivibR       Single-modality backbone with cell-type classification head.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .layers import (
    MaskedLinear,
    LoRALinear,
    VariationalEncoder,
    CellTypeClassifier,
)
from .utils import init_weights


# ---------------------------------------------------------------------------
# Bi-modal model — MaskedLinear translator
# ---------------------------------------------------------------------------

class multivib(nn.Module):
    """
    Bi-modal multiVIB backbone.

    Architecture: ``modality-B → MaskedLinear translator → shared encoder
    → shared projector``.

    Supports both **vertical** (paired) and **horizontal** (unpaired)
    integration via the ``joint`` flag.

    Args:
        n_input_a:  Input dimensionality of modality A.
        n_input_b:  Input dimensionality of modality B.
        n_hidden:   Hidden-layer width of the encoder.
        n_latent:   Latent-space dimensionality.
        n_batch:    Number of batch covariates for the projector.
        mask:       Optional ``(n_input_a, n_input_b)`` prior mask for the
                    translator.
        joint:      If ``True`` the projector is bypassed (joint-cell mode).
        relation:   ``"positive"`` or ``"negative"`` — sign of translator
                    weight initialisation.
    """

    def __init__(
        self,
        n_input_a: int = 2000,
        n_input_b: int = 2000,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_batch: int = 1,
        mask: Optional[torch.Tensor] = None,
        joint: bool = True,
        relation: str = "positive",
    ) -> None:
        super().__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.joint = joint

        masked_linear = MaskedLinear(self.n_input_b, self.n_input_a)
        if mask is not None:
            masked_linear.set_mask(mask)
        self.translator = nn.Sequential(masked_linear, nn.BatchNorm1d(self.n_input_a))

        self.encoder = VariationalEncoder(
            n_input=self.n_input_a, n_hidden=self.n_hidden, n_latent=self.n_latent
        )
        self.projecter = nn.Linear(self.n_latent + self.n_batch, 64)

        self.apply(init_weights)

        # Biologically-informed weight initialisation
        initial_weights = (
            torch.ones if relation == "positive" else lambda *a: -torch.ones(*a)
        )(self.n_input_a, self.n_input_b)
        if mask is not None:
            initial_weights[mask != 1] = 1e-6
        self.translator[0].weight.data = initial_weights.to(
            self.translator[0].weight.device, self.translator[0].weight.dtype
        )

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        batcha: torch.Tensor,
        batchb: torch.Tensor,
    ) -> dict:
        x_BtoA = self.translator(x_b)
        qz_a, z_a = self.encoder(x_a)
        qz_b, z_b = self.encoder(x_BtoA)

        if self.joint:
            p_a, p_b = z_a, z_b
        else:
            p_a = self.projecter(torch.cat((z_a, batcha), dim=1))
            p_b = self.projecter(torch.cat((z_b, batchb), dim=1))

        return {
            "z_a": z_a, "z_b": z_b,
            "qz_a": qz_a, "qz_b": qz_b,
            "proj_a": p_a, "proj_b": p_b,
            "a_trans": x_BtoA,
        }


# ---------------------------------------------------------------------------
# Bi-modal model — LoRA translator
# ---------------------------------------------------------------------------

class multivibLoRA(nn.Module):
    """
    Bi-modal multiVIB backbone with a low-rank (LoRA) translator.

    Identical to :class:`multivib` but replaces the MaskedLinear translator
    with a :class:`~multivib.layers.LoRALinear` module, which parameterises
    the translation matrix as a low-rank product ``W ≈ B · A``.

    Args:
        n_input_a: Input dimensionality of modality A.
        n_input_b: Input dimensionality of modality B.
        n_hidden:  Hidden-layer width.
        n_latent:  Latent-space dimensionality.
        n_batch:   Number of batch covariates.
        rank:      Inner rank of the LoRA translator.
        joint:     If ``True`` the projector is bypassed.
    """

    def __init__(
        self,
        n_input_a: int = 2000,
        n_input_b: int = 2000,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_batch: int = 1,
        rank: int = 128,
        joint: bool = True,
    ) -> None:
        super().__init__()
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.rank = rank
        self.joint = joint

        self.encoder = VariationalEncoder(
            n_input=self.n_input_a, n_hidden=self.n_hidden, n_latent=self.n_latent
        )
        self.projecter = nn.Linear(self.n_latent + self.n_batch, 64)
        self.apply(init_weights)

        # LoRA translator (initialised *after* apply so Kaiming isn't overwritten)
        self.translator = LoRALinear(self.n_input_b, self.n_input_a, self.rank)

    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        batcha: torch.Tensor,
        batchb: torch.Tensor,
    ) -> dict:
        x_BtoA = self.translator(x_b)
        qz_b, z_b = self.encoder(x_BtoA)
        qz_a, z_a = self.encoder(x_a)

        if self.joint:
            p_a, p_b = z_a, z_b
        else:
            p_a = self.projecter(torch.cat((z_a, batcha), dim=1))
            p_b = self.projecter(torch.cat((z_b, batchb), dim=1))

        return {
            "z_a": z_a, "z_b": z_b,
            "qz_a": qz_a, "qz_b": qz_b,
            "proj_a": p_a, "proj_b": p_b,
        }


# ---------------------------------------------------------------------------
# Multi-species model — MaskedLinear translators
# ---------------------------------------------------------------------------

class multivibS(nn.Module):
    """
    Multi-species multiVIB backbone with per-species MaskedLinear translators.

    Each species (or dataset) has its own modality-to-shared-space translator.
    All species share a single variational encoder, projector, and classifier.

    Args:
        n_input:        List of input dimensionalities, one per species.
        n_shared_input: Shared feature-space dimensionality (translator output).
        masks:          List of optional prior masks, one per species.
        relations:      List of ``"positive"`` / ``"negative"`` strings.
        n_hidden:       Encoder hidden-layer width.
        n_latent:       Latent-space dimensionality.
        n_batch:        Number of batch covariates.
        n_class:        Number of cell-type classes for the classifier.
    """

    def __init__(
        self,
        n_input: List[int] = (2000, 2000, 2000),
        n_shared_input: int = 1000,
        masks: Optional[List[Optional[torch.Tensor]]] = None,
        relations: Optional[List[str]] = None,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_batch: int = 1,
        n_class: int = 1,
    ) -> None:
        super().__init__()
        n_species = len(n_input)
        if masks is None:
            masks = [None] * n_species
        if relations is None:
            relations = ["positive"] * n_species

        self.n_input = n_input
        self.n_shared_input = n_shared_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch

        # Register translators as nn.ModuleList so PyTorch tracks their params
        translators = []
        for i in range(n_species):
            ml = MaskedLinear(n_input[i], n_shared_input)
            init_w = (
                torch.ones if relations[i] == "positive" else lambda *a: -torch.ones(*a)
            )(n_shared_input, n_input[i])
            if masks[i] is not None:
                ml.set_mask(masks[i])
                init_w[masks[i] != 1] = 0.0
            ml.weight.data = init_w.to(ml.weight.device, ml.weight.dtype)
            translators.append(nn.Sequential(ml, nn.BatchNorm1d(n_shared_input)))
        self.translators = nn.ModuleList(translators)

        self.encoder = VariationalEncoder(
            n_input=n_shared_input, n_hidden=n_hidden, n_latent=n_latent
        )
        self.projecter = nn.Linear(n_latent + n_batch, 64)
        self.classifier = CellTypeClassifier(input_dim=n_latent, num_classes=n_class)

        self.apply(init_weights)

    def forward(self, xs: List[torch.Tensor], batches: List[torch.Tensor]) -> dict:
        z, qz, proj, y = [], [], [], []
        for i, (x_i, b_i) in enumerate(zip(xs, batches)):
            xt = self.translators[i](x_i)
            qz_i, z_i = self.encoder(xt)
            p_i = self.projecter(torch.cat((z_i, b_i), dim=1))
            y_i = self.classifier(z_i)
            z.append(z_i)
            qz.append(qz_i)
            proj.append(p_i)
            y.append(y_i)
        return {"z": z, "qz": qz, "proj": proj, "y": y}


# ---------------------------------------------------------------------------
# Multi-species model — LoRA translators (shared B matrix)
# ---------------------------------------------------------------------------

class multivibLoRAS(nn.Module):
    """
    Multi-species multiVIB backbone with shared low-rank translators.

    Each species has its own species-specific matrix A while they share a
    common matrix B, allowing cross-species feature alignment with fewer
    parameters: ``W_i = B · A_i``.

    Args:
        n_input:        List of input dimensionalities, one per species.
        n_shared_input: Shared feature-space dimensionality (B output).
        n_hidden:       Encoder hidden-layer width.
        n_latent:       Latent-space dimensionality.
        n_batch:        Number of batch covariates.
        n_class:        Number of cell-type classes.
        rank:           Inner rank of each A matrix.
    """

    def __init__(
        self,
        n_input: List[int] = (2000, 2000, 2000),
        n_shared_input: int = 1000,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_batch: int = 1,
        n_class: int = 1,
        rank: int = 128,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_shared_input = n_shared_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.rank = rank

        self.encoder = VariationalEncoder(
            n_input=n_shared_input, n_hidden=n_hidden, n_latent=n_latent
        )
        self.projecter = nn.Linear(n_latent + n_batch, 64)
        self.classifier = CellTypeClassifier(input_dim=n_latent, num_classes=n_class)

        # Register as nn.ModuleList so parameters are tracked
        self.matrixA = nn.ModuleList(
            [nn.Linear(d, rank, bias=False) for d in n_input]
        )
        self.matrixB = nn.Linear(rank, n_shared_input, bias=True)
        self.batchnorm = nn.BatchNorm1d(n_shared_input)

        self.apply(init_weights)

        # Re-initialise A and B explicitly (apply() may have overwritten them)
        for layer in self.matrixA:
            nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_uniform_(self.matrixB.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.batchnorm.weight, 1.0, 0.02)
        nn.init.zeros_(self.batchnorm.bias)

    def forward(self, xs: List[torch.Tensor], batches: List[torch.Tensor]) -> dict:
        z, qz, proj, y = [], [], [], []
        for i, (x_i, b_i) in enumerate(zip(xs, batches)):
            xt = self.batchnorm(self.matrixB(self.matrixA[i](x_i)))
            qz_i, z_i = self.encoder(xt)
            p_i = self.projecter(torch.cat((z_i, b_i), dim=1))
            y_i = self.classifier(z_i)
            z.append(z_i)
            qz.append(qz_i)
            proj.append(p_i)
            y.append(y_i)
        return {"z": z, "qz": qz, "proj": proj, "y": y}


# ---------------------------------------------------------------------------
# Single-modality model
# ---------------------------------------------------------------------------

class multivibR(nn.Module):
    """
    Single-modality multiVIB backbone with cell-type classification head.

    Suitable for reference-atlas building from a single omics layer (e.g.
    scRNA-seq only).

    Args:
        n_input_a: Input dimensionality.
        n_hidden:  Encoder hidden-layer width.
        n_latent:  Latent-space dimensionality.
        n_batch:   Number of batch covariates.
        n_class:   Number of cell-type classes.
    """

    def __init__(
        self,
        n_input_a: int = 2000,
        n_hidden: int = 256,
        n_latent: int = 10,
        n_batch: int = 1,
        n_class: int = 10,
    ) -> None:
        super().__init__()
        self.n_input_a = n_input_a
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.n_class = n_class

        self.encoder = VariationalEncoder(
            n_input=n_input_a, n_hidden=n_hidden, n_latent=n_latent
        )
        self.projecter = nn.Linear(n_latent + n_batch, 64)
        self.classifier = CellTypeClassifier(input_dim=n_latent, num_classes=n_class)

        self.apply(init_weights)

    def forward(self, x_a: torch.Tensor, batcha: torch.Tensor) -> dict:
        qz_a, z_a = self.encoder(x_a)
        p_a = self.projecter(torch.cat((z_a, batcha), dim=1))
        y_a = self.classifier(z_a)
        return {"z_a": z_a, "qz_a": qz_a, "proj_a": p_a, "y_a": y_a}
