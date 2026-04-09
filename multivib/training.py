"""
Training loops for all multiVIB integration scenarios.

Functions
---------
multivib_vertical_training    Paired + unpaired vertical integration.
multivib_horizontal_training  Unpaired horizontal integration.
multivib_species_training     Multi-species / mosaic integration.
multivibR_training            Single-modality with cell-type supervision.
"""

import numpy as np
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from sklearn.linear_model import LinearRegression
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .losses import DCL, OODAlignmentLoss, GraphNeighborhoodReg, VICRegLoss
from .utils import crossover_augmentation


# ---------------------------------------------------------------------------
# Vertical integration (paired + unpaired cells, two modalities)
# ---------------------------------------------------------------------------

def multivib_vertical_training(
    model,
    Xa, Xb,
    Xa_pair, Xb_pair,
    batcha, batchb,
    batcha_pair, batchb_pair,
    epoch: int = 100,
    batch_size: int = 128,
    temp: float = 0.15,
    alpha: float = 0.05,
    beta: float = 0.2,
    crossover_rate: float = 0.0,
    gaussian_rate_var: float = 1.0,
    random_seed: int = 0,
    if_lr: bool = False,
):
    """
    Train a :class:`~multivib.models.multivib` model for **vertical
    integration** (jointly-profiled anchor cells + unpaired OOD cells).

    The loss combines:

    * **Contrastive loss** (DCL) on paired cells and self-augmented OOD cells.
    * **KL regularisation** on the VAE posterior.
    * **OOD alignment** (Sinkhorn OT) on unpaired projections.

    Args:
        model:             A :class:`~multivib.models.multivib` instance.
        Xa / Xb:           Unpaired data matrices for modalities A and B,
                           shape ``(N, G_A)`` / ``(M, G_B)``.
        Xa_pair / Xb_pair: Paired (anchor) data matrices.
        batcha / batchb:   Batch covariate arrays for unpaired data.
        batcha_pair / batchb_pair:
                           Batch covariate arrays for paired data.
        epoch:             Number of training epochs.
        batch_size:        Mini-batch size.
        temp:              Contrastive-loss temperature.
        alpha:             KL loss weight.
        beta:              OOD alignment loss weight.
        crossover_rate:    CrossOver augmentation rate (0 = disabled).
        gaussian_rate_var: Gaussian noise standard deviation added to inputs.
        random_seed:       Base random seed for reproducible shuffling.
        if_lr:             Initialise translator weights via linear regression.

    Returns:
        List of per-epoch log-losses.
    """
    if if_lr:
        print("Initialising translator via linear regression …")
        lr = LinearRegression().fit(Xb_pair, Xa_pair)
        with torch.no_grad():
            model.translator[0].weight.copy_(torch.from_numpy(lr.coef_))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)
    ood_alignment = OODAlignmentLoss(
        n_prototypes=64, latent_dim=64, sinkhorn_eps=0.05,
        ot_weight=1.0, cluster_momentum=0.99, use_pseudo_labels_for_ot=False,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=5e-4)
    loss_history = []

    for e in range(epoch):
        model.to(device)

        # ------ shuffle and tensorise ----------------------------------------
        rA = np.random.RandomState(random_seed + e).permutation(Xa.shape[0])
        rB = np.random.RandomState(random_seed + e).permutation(Xb.shape[0])
        rP = np.random.RandomState(random_seed + e).permutation(Xa_pair.shape[0])

        X_tA = torch.tensor(Xa[rA]).float()
        y_tA = torch.tensor(batcha[rA]).float()
        X_tB = torch.tensor(Xb[rB]).float()
        y_tB = torch.tensor(batchb[rB]).float()
        X_tAp = torch.tensor(Xa_pair[rP]).float()
        y_tAp = torch.tensor(batcha_pair[rP]).float()
        X_tBp = torch.tensor(Xb_pair[rP]).float()
        y_tBp = torch.tensor(batchb_pair[rP]).float()

        n = min(Xa.shape[0], Xb.shape[0], Xa_pair.shape[0])
        total_loss = []

        with tqdm(total=n // batch_size, desc=f"Epoch {e+1}/{epoch}",
                  unit="batch", bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False) as pbar:

            for i in range(n // batch_size):
                pbar.update(1)
                opt.zero_grad()

                sl = slice(i * batch_size, (i + 1) * batch_size)

                def _aug(t):
                    t = t.to(device)
                    t = crossover_augmentation(t, crossover_rate)
                    c, m = t.shape
                    t = t + torch.normal(0, gaussian_rate_var, (c, m), device=device)
                    return t

                a1, a2 = _aug(X_tA[sl]), _aug(X_tA[sl])
                b1, b2 = _aug(X_tB[sl]), _aug(X_tB[sl])
                ba = y_tA[sl].to(device)
                bb = y_tB[sl].to(device)

                ap = X_tAp[sl].to(device)
                bp = X_tBp[sl].to(device)
                bap = y_tAp[sl].to(device)
                bbp = y_tBp[sl].to(device)

                model.joint = False
                out1 = model(a1, b1, ba, bb)
                out2 = model(a2, b2, ba, bb)

                model.joint = True
                out_pair = model(ap, bp, bap, bbp)

                cont_loss = (
                    contrastive_loss(out_pair["proj_a"], out_pair["proj_b"])
                    + contrastive_loss(out1["proj_a"], out2["proj_a"])
                    + contrastive_loss(out1["proj_b"], out2["proj_b"])
                )

                pz = Normal(
                    torch.zeros_like(out1["qz_a"].mean),
                    torch.ones_like(out1["qz_a"].mean),
                )
                kl_loss = (
                    kl(out1["qz_a"], pz).sum(dim=1).mean()
                    + kl(out1["qz_b"], pz).sum(dim=1).mean()
                )
                ood_loss, _ = ood_alignment(out1["proj_a"], out1["proj_b"])

                loss = cont_loss + kl_loss * alpha + ood_loss * beta
                loss.backward()
                opt.step()
                total_loss.append(loss)

        loss_history.append(sum(total_loss).log().cpu().detach().numpy())

    return loss_history


# ---------------------------------------------------------------------------
# Horizontal integration (unpaired, two modalities)
# ---------------------------------------------------------------------------

def multivib_horizontal_training(
    model,
    Xa, Xb,
    batcha, batchb,
    epoch: int = 100,
    batch_size: int = 128,
    temp: float = 0.15,
    alpha: float = 0.05,
    beta: float = 0.2,
    crossover_rate: float = 0.0,
    gaussian_rate_var: float = 1.0,
    random_seed: int = 0,
):
    """
    Train a :class:`~multivib.models.multivib` model for **horizontal
    integration** (no paired cells; datasets anchored through shared features).

    The loss combines:

    * **Contrastive loss** (DCL) on self-augmented views within each modality.
    * **KL regularisation** on the VAE posterior.
    * **OOD alignment** (Sinkhorn OT).
    * **Graph neighbourhood regularisation** (KNN graph from RNA space).
    * **VICReg** to prevent latent dimension collapse.

    Args:
        model:             A :class:`~multivib.models.multivib` instance.
        Xa / Xb:           Data matrices for modalities A and B.
        batcha / batchb:   Batch covariate arrays.
        epoch:             Number of training epochs.
        batch_size:        Mini-batch size.
        temp:              Contrastive-loss temperature.
        alpha:             KL loss weight.
        beta:              OOD alignment loss weight.
        crossover_rate:    CrossOver augmentation rate.
        gaussian_rate_var: Gaussian noise std added to inputs.
        random_seed:       Base random seed.

    Returns:
        List of per-epoch log-losses.
    """
    model.joint = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contrastive_loss = DCL(temperature=temp)
    ood_alignment = OODAlignmentLoss(
        n_prototypes=64, latent_dim=64, sinkhorn_eps=0.05,
        ot_weight=1.0, cluster_momentum=0.99, use_pseudo_labels_for_ot=False,
    ).to(device)
    graph_reg = GraphNeighborhoodReg(
        k=15, weight_alignment=0.5, weight_contrastive=1.0,
        weight_laplacian=0.5, weight_diffusion=0.0,
        contrastive_margin=0.5, n_negative_samples=64,
    ).to(device)
    vicreg = VICRegLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=5e-4)
    loss_history = []

    for e in range(epoch):
        model.to(device)

        rA = np.random.RandomState(random_seed + e).permutation(Xa.shape[0])
        rB = np.random.RandomState(random_seed + e).permutation(Xb.shape[0])
        X_tA = torch.tensor(Xa[rA]).float()
        y_tA = torch.tensor(batcha[rA]).float()
        X_tB = torch.tensor(Xb[rB]).float()
        y_tB = torch.tensor(batchb[rB]).float()

        n = min(Xa.shape[0], Xb.shape[0])
        total_loss = []

        with tqdm(total=n // batch_size, desc=f"Epoch {e+1}/{epoch}",
                  unit="batch", bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False) as pbar:

            for i in range(n // batch_size):
                pbar.update(1)
                opt.zero_grad()

                sl = slice(i * batch_size, (i + 1) * batch_size)

                def _aug(t):
                    t = t.to(device)
                    t = crossover_augmentation(t, crossover_rate)
                    c, m = t.shape
                    return t + torch.normal(0, gaussian_rate_var, (c, m), device=device)

                a1, a2 = _aug(X_tA[sl]), _aug(X_tA[sl])
                b1, b2 = _aug(X_tB[sl]), _aug(X_tB[sl])
                ba = y_tA[sl].to(device)
                bb = y_tB[sl].to(device)

                out1 = model(a1, b1, ba, bb)
                out2 = model(a2, b2, ba, bb)

                cont_loss = (
                    contrastive_loss(out1["proj_a"], out2["proj_a"])
                    + contrastive_loss(out1["proj_b"], out2["proj_b"]) * 2.0
                )

                pz = Normal(
                    torch.zeros_like(out1["qz_a"].mean),
                    torch.ones_like(out1["qz_a"].mean),
                )
                kl_loss = (
                    kl(out1["qz_a"], pz).sum(dim=1).mean()
                    + kl(out1["qz_b"], pz).sum(dim=1).mean()
                )
                ood_loss, _ = ood_alignment(out1["proj_a"], out1["proj_b"])
                graph_loss = graph_reg(out1["proj_a"], out1["proj_b"])
                vic_loss = 0.1 * vicreg(out1["proj_a"], out1["proj_b"])

                loss = cont_loss + kl_loss * alpha + ood_loss * beta + graph_loss + vic_loss
                loss.backward()
                opt.step()
                total_loss.append(loss)

        loss_history.append(sum(total_loss).log().cpu().detach().numpy())

    return loss_history


# ---------------------------------------------------------------------------
# Multi-species integration
# ---------------------------------------------------------------------------

def multivib_species_training(
    model,
    Xs,
    batches, cell_types,
    epoch: int = 100,
    batch_size: int = 128,
    temp: float = 0.15,
    alpha: float = 0.05,
    beta: float = 0.1,
    param_setup: str = "1st",
    crossover_rate: float = 0.25,
    gaussian_rate_var: float = 1.0,
    random_seed: int = 0,
):
    """
    Train a :class:`~multivib.models.multivibS` (or
    :class:`~multivib.models.multivibLoRAS`) model for **multi-species
    cross-species integration**.

    One species is treated as the reference (index 0); all others are
    aligned to it via OOD alignment and graph neighbourhood regularisation.
    Supervised cell-type labels (``"Unknown"`` for unannotated cells) are
    used where available.

    Args:
        model:             A :class:`~multivib.models.multivibS` or
                           :class:`~multivib.models.multivibLoRAS` instance.
        Xs:                List of data matrices, one per species.
        batches:           List of batch covariate arrays.
        cell_types:        List of cell-type label arrays (use ``"Unknown"``
                           for unlabelled cells).
        epoch:             Number of training epochs.
        batch_size:        Mini-batch size.
        temp:              Contrastive-loss temperature.
        alpha:             KL loss weight.
        beta:              OOD alignment loss weight.
        param_setup:       ``"1st"`` uses ``model.translators``; ``"2nd"``
                           uses ``model.matrixA`` (LoRA variant).
        crossover_rate:    CrossOver augmentation rate.
        gaussian_rate_var: Gaussian noise std.
        random_seed:       Base random seed.

    Returns:
        List of per-epoch log-losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contrastive_loss = DCL(temperature=temp)
    ood_alignment = OODAlignmentLoss(
        n_prototypes=64, latent_dim=64, sinkhorn_eps=0.05,
        ot_weight=1.0, cluster_momentum=0.99, use_pseudo_labels_for_ot=False,
    ).to(device)
    # graph_reg = GraphNeighborhoodReg(
    #     k=15, weight_alignment=0.5, weight_contrastive=1.0,
    #     weight_laplacian=0.5, weight_diffusion=0.0,
    #     contrastive_margin=0.5, n_negative_samples=64,
    # ).to(device)
    vicreg = VICRegLoss()

    # ------ cell-type encoding -----------------------------------------------
    ct_flat = np.concatenate([np.asarray(c) for c in cell_types])
    ct_enc = LabelEncoder()
    ct_enc.fit(ct_flat)
    encoded_ct = ct_enc.transform(ct_flat)
    unknown_class = ct_enc.transform(["Unknown"])[0]

    classes = np.unique(encoded_ct)
    cw = class_weight.compute_class_weight("balanced", classes=classes, y=encoded_ct)
    cw[classes == unknown_class] = 0.0
    cls_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(device)
    )

    # ------ optimiser --------------------------------------------------------
    params = list(model.parameters())
    if param_setup == "1st":
        for t in model.translators:
            params += list(t.parameters())
    elif param_setup == "2nd":
        for t in model.matrixA:
            params += list(t.parameters())
    opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=5e-4)

    # ------ move to device ---------------------------------------------------
    model.to(device)
    n_species = len(Xs)
    if param_setup == "1st":
        for t in model.translators:
            t.to(device)
    elif param_setup == "2nd":
        for t in model.matrixA:
            t.to(device)

    n = min(x.shape[0] for x in Xs)
    loss_history = []

    for e in range(epoch):

        X_tensor, y_tensor, ct_tensor = [], [], []
        offset = 0
        for i, Xi in enumerate(Xs):
            ni = Xi.shape[0]
            r = np.random.RandomState(random_seed + e).permutation(ni)
            X_tensor.append(torch.tensor(Xi[r]).float())
            y_tensor.append(torch.tensor(batches[i][r]).float())
            cts_i = np.asarray(cell_types[i])[r]
            ct_tensor.append(torch.tensor(ct_enc.transform(cts_i), dtype=torch.long))
            offset += ni

        total_loss = []

        with tqdm(total=n // batch_size, desc=f"Epoch {e+1}/{epoch}",
                  unit="batch", bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False) as pbar:

            for i in range(n // batch_size):
                pbar.update(1)
                opt.zero_grad()

                sl = slice(i * batch_size, (i + 1) * batch_size)
                inputs1, inputs2, batch, ct_batch = [], [], [], []

                for j in range(n_species):
                    x1 = X_tensor[j][sl].to(device)
                    x2 = X_tensor[j][sl].to(device)
                    b = y_tensor[j][sl].to(device)
                    ct = ct_tensor[j][sl].to(device)
                    c, m = x1.shape
                    x1 = crossover_augmentation(x1, crossover_rate) + torch.normal(
                        0, gaussian_rate_var, (c, m), device=device
                    )
                    x2 = crossover_augmentation(x2, crossover_rate) + torch.normal(
                        0, gaussian_rate_var, (c, m), device=device
                    )
                    inputs1.append(x1)
                    inputs2.append(x2)
                    batch.append(b)
                    ct_batch.append(ct)

                out1 = model(inputs1, batch)
                out2 = model(inputs2, batch)

                pz = Normal(
                    torch.zeros_like(out1["qz"][0].mean),
                    torch.ones_like(out1["qz"][0].mean),
                )

                # Reference species (index 0)
                kl_loss = kl(out1["qz"][0], pz).sum(dim=1).mean()
                cont_loss = contrastive_loss(out1["proj"][0], out2["proj"][0])
                known_0 = ct_batch[0] != unknown_class
                if known_0.any():
                    clf_loss = cls_criterion(
                        out1["y"][0][known_0], ct_batch[0][known_0]
                    )
                    loss = cont_loss + kl_loss * alpha + clf_loss
                else:
                    loss = cont_loss + kl_loss * alpha

                # Non-reference species
                for s in range(1, n_species):
                    c_s = contrastive_loss(out1["proj"][s], out2["proj"][s])
                    kl_s = kl(out1["qz"][s], pz).sum(dim=1).mean()
                    ood_s, _ = ood_alignment(out1["proj"][s], out1["proj"][s - 1])
                    # g_s = graph_reg(out1["proj"][s], out1["proj"][s - 1])
                    v_s = 0.1 * vicreg(out1["proj"][s], out1["proj"][s - 1])

                    known_s = ct_batch[s] != unknown_class
                    if known_s.any():
                        clf_s = cls_criterion(out1["y"][s][known_s], ct_batch[s][known_s])
                        loss += c_s + kl_s * alpha + ood_s * beta + clf_s + v_s # + g_s
                    else:
                        loss += c_s + kl_s * alpha + ood_s * beta + v_s # + g_s

                loss.backward()
                opt.step()
                total_loss.append(loss)

        loss_history.append(sum(total_loss).log().cpu().detach().numpy())

    return loss_history


# ---------------------------------------------------------------------------
# Single-modality training
# ---------------------------------------------------------------------------

def multivibR_training(
    model,
    Xa, batcha, cell_types,
    epoch: int = 100,
    batch_size: int = 128,
    temp: float = 0.15,
    alpha: float = 0.05,
    crossover_rate: float = 0.25,
    gaussian_rate_var: float = 1.0,
    random_seed: int = 0,
):
    """
    Train a :class:`~multivib.models.multivibR` model for **single-modality**
    integration with optional cell-type supervision.

    Args:
        model:             A :class:`~multivib.models.multivibR` instance.
        Xa:                Data matrix, shape ``(N, G)``.
        batcha:            Batch covariate array, shape ``(N, n_batch)``.
        cell_types:        Cell-type label array; use ``"Unknown"`` for
                           unlabelled cells.
        epoch:             Number of training epochs.
        batch_size:        Mini-batch size.
        temp:              Contrastive-loss temperature.
        alpha:             KL loss weight.
        crossover_rate:    CrossOver augmentation rate.
        gaussian_rate_var: Gaussian noise std.
        random_seed:       Base random seed.

    Returns:
        List of per-epoch log-losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=5e-4)

    ct_enc = LabelEncoder()
    encoded_ct = ct_enc.fit_transform(cell_types)
    unknown_class = ct_enc.transform(["Unknown"])[0]

    classes = np.unique(encoded_ct)
    cw = class_weight.compute_class_weight("balanced", classes=classes, y=encoded_ct)
    cw[classes == unknown_class] = 0.0
    cls_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(device)
    )

    loss_history = []
    for e in range(epoch):
        model.to(device)

        r = np.random.RandomState(random_seed + e).permutation(Xa.shape[0])
        X_tA = torch.tensor(Xa[r]).float()
        y_tA = torch.tensor(batcha[r]).float()
        ct_tA = torch.tensor(encoded_ct[r], dtype=torch.long)

        n = Xa.shape[0]
        total_loss = []

        with tqdm(total=n // batch_size, desc=f"Epoch {e+1}/{epoch}",
                  unit="batch", bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False) as pbar:

            for i in range(n // batch_size):
                pbar.update(1)
                opt.zero_grad()

                sl = slice(i * batch_size, (i + 1) * batch_size)
                x1 = X_tA[sl].to(device)
                x2 = X_tA[sl].to(device)
                ba = y_tA[sl].to(device)
                ct = ct_tA[sl].to(device)
                c, m = x1.shape

                x1 = crossover_augmentation(x1, crossover_rate) + torch.normal(
                    0, gaussian_rate_var, (c, m), device=device
                )
                x2 = crossover_augmentation(x2, crossover_rate) + torch.normal(
                    0, gaussian_rate_var, (c, m), device=device
                )

                out1 = model(x1, ba)
                out2 = model(x2, ba)

                cont_loss = contrastive_loss(out1["proj_a"], out2["proj_a"])
                pz = Normal(
                    torch.zeros_like(out1["qz_a"].mean),
                    torch.ones_like(out1["qz_a"].mean),
                )
                kl_loss = kl(out1["qz_a"], pz).sum(dim=1).mean()

                known = ct != unknown_class
                if known.any():
                    clf_loss = cls_criterion(out1["y_a"][known], ct[known])
                    loss = cont_loss + kl_loss * alpha + clf_loss
                else:
                    loss = cont_loss + kl_loss * alpha

                loss.backward()
                opt.step()
                total_loss.append(loss)

        loss_history.append(sum(total_loss).log().cpu().detach().numpy())

    return loss_history
