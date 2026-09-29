"""
Training loops for all multiVIB integration scenarios.

Functions
---------
multivib_vertical_training    Paired + unpaired vertical integration.
multivib_horizontal_training  Unpaired horizontal integration.
multivib_species_training     Multi-species / mosaic integration.
multivibR_training            Single-modality with cell-type supervision.
multivib_joint_training       Simultaneous cross-species + cross-modality integration.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from sklearn.linear_model import LinearRegression
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .layers import GMMPrior
from .losses import DCL, OODAlignmentLoss, GraphNeighborhoodReg, VICRegLoss
from .utils import crossover_augmentation, kl_annealing_weight, EMA, latent_mixup


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
    kl_anneal: bool = True,
    kl_anneal_schedule: str = "cyclical",
    kl_anneal_cycles: int = 4,
    kl_anneal_ratio: float = 0.5,
    use_gmm_prior: bool = False,
    n_gmm_components: int = 10,
    mixup_alpha: float = 0.0,
):
    """
    Train a :class:`~multivib.models.multivib` model for **vertical
    integration** (jointly-profiled anchor cells + unpaired OOD cells).

    The loss combines:

    * **Contrastive loss** (DCL) on paired cells and self-augmented OOD cells.
    * **KL regularisation** on the VAE posterior (standard N(0,1) or GMM).
    * **OOD alignment** (Sinkhorn OT) on unpaired projections.
    * **Latent Mixup** (optional) — additional contrastive views formed by
      interpolating pairs of latent codes in the embedding space.

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
        alpha:             KL loss weight (peak value once annealing ramps up).
        beta:              OOD alignment loss weight.
        crossover_rate:    CrossOver augmentation rate (0 = disabled).
        gaussian_rate_var: Gaussian noise standard deviation added to inputs.
        random_seed:       Base random seed for reproducible shuffling.
        if_lr:             Initialise translator weights via linear regression.
        kl_anneal:         Anneal the KL weight instead of holding it fixed
                           at ``alpha`` from epoch 0 (mitigates posterior
                           collapse / under-regularisation).
        kl_anneal_schedule: ``"constant"``, ``"monotonic"``, or ``"cyclical"``.
        kl_anneal_cycles:  Number of ramp cycles (``"cyclical"`` only).
        kl_anneal_ratio:   Fraction of each cycle spent ramping up.
        use_gmm_prior:     Replace the standard ``N(0, I)`` prior with a
                           learnable ``K``-component GMM prior.  The GMM
                           parameters are jointly optimised with the model.
        n_gmm_components:  Number of GMM components ``K`` (only used when
                           ``use_gmm_prior=True``).
        mixup_alpha:       Beta distribution concentration for latent Mixup.
                           ``0.0`` disables Mixup; ``0.4`` is a good default.

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

    # GMM prior setup — its parameters are added to the optimiser so that
    # the component means / variances are learned alongside the encoder.
    if use_gmm_prior:
        gmm_prior = GMMPrior(
            n_components=n_gmm_components, n_latent=model.n_latent
        ).to(device)
        params = list(model.parameters()) + list(gmm_prior.parameters())
    else:
        gmm_prior = None
        params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=5e-4)
    loss_history = []

    for e in range(epoch):
        model.to(device)

        kl_weight = alpha * (
            kl_annealing_weight(
                e, epoch, schedule=kl_anneal_schedule,
                n_cycles=kl_anneal_cycles, ratio=kl_anneal_ratio,
            ) if kl_anneal else 1.0
        )

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

                # Latent Mixup: interpolate unpaired latent codes and use the
                # mixed representation as an additional hard positive view.
                if mixup_alpha > 0.0:
                    z_mix_a, _ = latent_mixup(out1["z_a"], alpha=mixup_alpha)
                    z_mix_b, _ = latent_mixup(out1["z_b"], alpha=mixup_alpha)
                    proj_mix_a = model.projecter(z_mix_a, ba)
                    proj_mix_b = model.projecter(z_mix_b, bb)
                    cont_loss = (
                        cont_loss
                        + contrastive_loss(out2["proj_a"], proj_mix_a)
                        + contrastive_loss(out2["proj_b"], proj_mix_b)
                    )

                # KL regularisation — use GMM prior if available, else N(0, I)
                if gmm_prior is not None:
                    kl_loss = (
                        gmm_prior.kl_divergence(out1["qz_a"], out1["z_a"]).mean()
                        + gmm_prior.kl_divergence(out1["qz_b"], out1["z_b"]).mean()
                    )
                else:
                    pz = Normal(
                        torch.zeros_like(out1["qz_a"].mean),
                        torch.ones_like(out1["qz_a"].mean),
                    )
                    kl_loss = (
                        kl(out1["qz_a"], pz).sum(dim=1).mean()
                        + kl(out1["qz_b"], pz).sum(dim=1).mean()
                    )

                ood_loss, _ = ood_alignment(out1["proj_a"], out1["proj_b"])

                loss = cont_loss + kl_loss * kl_weight + ood_loss * beta
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
    kl_anneal: bool = True,
    kl_anneal_schedule: str = "cyclical",
    kl_anneal_cycles: int = 4,
    kl_anneal_ratio: float = 0.5,
    graph_warmup_epochs: int = 10,
    use_ema_teacher: bool = True,
    graph_ema_decay: float = 0.996,
    use_gmm_prior: bool = False,
    n_gmm_components: int = 10,
    mixup_alpha: float = 0.0,
):
    """
    Train a :class:`~multivib.models.multivib` model for **horizontal
    integration** (no paired cells; datasets anchored through shared features).

    The loss combines:

    * **Contrastive loss** (DCL) on self-augmented views within each modality.
    * **KL regularisation** on the VAE posterior (standard N(0,1) or GMM).
    * **OOD alignment** (Sinkhorn OT).
    * **Graph neighbourhood regularisation** (KNN graph from RNA space).
    * **VICReg** to prevent latent dimension collapse.
    * **Latent Mixup** (optional) — additional contrastive views formed by
      interpolating pairs of latent codes in the embedding space.

    Args:
        model:             A :class:`~multivib.models.multivib` instance.
        Xa / Xb:           Data matrices for modalities A and B.
        batcha / batchb:   Batch covariate arrays.
        epoch:             Number of training epochs.
        batch_size:        Mini-batch size.
        temp:              Contrastive-loss temperature.
        alpha:             KL loss weight (peak value once annealing ramps up).
        beta:              OOD alignment loss weight.
        crossover_rate:    CrossOver augmentation rate.
        gaussian_rate_var: Gaussian noise std added to inputs.
        random_seed:       Base random seed.
        kl_anneal:         Anneal the KL weight instead of holding it fixed
                           at ``alpha`` from epoch 0.
        kl_anneal_schedule: ``"constant"``, ``"monotonic"``, or ``"cyclical"``.
        kl_anneal_cycles:  Number of ramp cycles (``"cyclical"`` only).
        kl_anneal_ratio:   Fraction of each cycle spent ramping up.
        graph_warmup_epochs:
                           Linearly ramp the graph-regularisation weight from
                           0 to 1 over this many epochs, instead of applying
                           it at full strength against a still-random RNA
                           latent space from epoch 0. ``0`` disables warmup.
        use_ema_teacher:   Build the RNA KNN graph from an EMA "teacher" copy
                           of the model instead of the student's own
                           in-training forward pass, so the graph target is
                           stable rather than shifting every step
                           (mean-teacher style, Tarvainen & Valpola 2017).
        graph_ema_decay:   EMA decay rate for the teacher (only used if
                           ``use_ema_teacher`` is ``True``).
        use_gmm_prior:     Replace the standard ``N(0, I)`` prior with a
                           learnable ``K``-component GMM prior.
        n_gmm_components:  Number of GMM components ``K``.
        mixup_alpha:       Beta distribution concentration for latent Mixup.
                           ``0.0`` disables Mixup; ``0.4`` is a good default.

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

    if use_gmm_prior:
        gmm_prior = GMMPrior(
            n_components=n_gmm_components, n_latent=model.n_latent
        ).to(device)
        params = list(model.parameters()) + list(gmm_prior.parameters())
    else:
        gmm_prior = None
        params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=5e-4)
    teacher = EMA(model, decay=graph_ema_decay).to(device) if use_ema_teacher else None
    loss_history = []

    for e in range(epoch):
        model.to(device)

        kl_weight = alpha * (
            kl_annealing_weight(
                e, epoch, schedule=kl_anneal_schedule,
                n_cycles=kl_anneal_cycles, ratio=kl_anneal_ratio,
            ) if kl_anneal else 1.0
        )
        graph_weight = (
            min(1.0, (e + 1) / graph_warmup_epochs) if graph_warmup_epochs > 0 else 1.0
        )

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

                # Latent Mixup: create additional hard positive views by
                # interpolating latent codes from different cells.
                if mixup_alpha > 0.0:
                    z_mix_a, _ = latent_mixup(out1["z_a"], alpha=mixup_alpha)
                    z_mix_b, _ = latent_mixup(out1["z_b"], alpha=mixup_alpha)
                    proj_mix_a = model.projecter(z_mix_a, ba)
                    proj_mix_b = model.projecter(z_mix_b, bb)
                    cont_loss = (
                        cont_loss
                        + contrastive_loss(out2["proj_a"], proj_mix_a)
                        + contrastive_loss(out2["proj_b"], proj_mix_b)
                    )

                # KL regularisation
                if gmm_prior is not None:
                    kl_loss = (
                        gmm_prior.kl_divergence(out1["qz_a"], out1["z_a"]).mean()
                        + gmm_prior.kl_divergence(out1["qz_b"], out1["z_b"]).mean()
                    )
                else:
                    pz = Normal(
                        torch.zeros_like(out1["qz_a"].mean),
                        torch.ones_like(out1["qz_a"].mean),
                    )
                    kl_loss = (
                        kl(out1["qz_a"], pz).sum(dim=1).mean()
                        + kl(out1["qz_b"], pz).sum(dim=1).mean()
                    )

                ood_loss, _ = ood_alignment(out1["proj_a"], out1["proj_b"])

                if use_ema_teacher:
                    z_rna_target = teacher(a1, b1, ba, bb)["proj_a"]
                else:
                    z_rna_target = out1["proj_a"]
                graph_loss = graph_weight * graph_reg(z_rna_target, out1["proj_b"])
                vic_loss = 0.1 * vicreg(out1["proj_a"], out1["proj_b"])

                loss = cont_loss + kl_loss * kl_weight + ood_loss * beta + graph_loss + vic_loss
                loss.backward()
                opt.step()
                if use_ema_teacher:
                    teacher.update(model)
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
    kl_anneal: bool = True,
    kl_anneal_schedule: str = "cyclical",
    kl_anneal_cycles: int = 4,
    kl_anneal_ratio: float = 0.5,
    use_gmm_prior: bool = False,
    n_gmm_components: int = 10,
    mixup_alpha: float = 0.0,
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
        kl_anneal:         Anneal the KL weight instead of holding it fixed
                           at ``alpha`` from epoch 0.
        kl_anneal_schedule: ``"constant"``, ``"monotonic"``, or ``"cyclical"``.
        kl_anneal_cycles:  Number of ramp cycles (``"cyclical"`` only).
        kl_anneal_ratio:   Fraction of each cycle spent ramping up.
        use_gmm_prior:     Replace the standard ``N(0, I)`` prior with a
                           learnable ``K``-component GMM prior.
        n_gmm_components:  Number of GMM components ``K``.
        mixup_alpha:       Beta distribution concentration for latent Mixup.
                           ``0.0`` disables Mixup; ``0.4`` is a good default.

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

    if use_gmm_prior:
        gmm_prior = GMMPrior(
            n_components=n_gmm_components, n_latent=model.n_latent
        ).to(device)
        params += list(gmm_prior.parameters())
    else:
        gmm_prior = None

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

        kl_weight = alpha * (
            kl_annealing_weight(
                e, epoch, schedule=kl_anneal_schedule,
                n_cycles=kl_anneal_cycles, ratio=kl_anneal_ratio,
            ) if kl_anneal else 1.0
        )

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

                # Reference species (index 0)
                if gmm_prior is not None:
                    kl_loss = gmm_prior.kl_divergence(
                        out1["qz"][0], out1["z"][0]
                    ).mean()
                else:
                    pz = Normal(
                        torch.zeros_like(out1["qz"][0].mean),
                        torch.ones_like(out1["qz"][0].mean),
                    )
                    kl_loss = kl(out1["qz"][0], pz).sum(dim=1).mean()

                cont_loss = contrastive_loss(out1["proj"][0], out2["proj"][0])

                # Latent Mixup for the reference species
                if mixup_alpha > 0.0:
                    z_mix_0, _ = latent_mixup(out1["z"][0], alpha=mixup_alpha)
                    proj_mix_0 = model.projecter(z_mix_0, batch[0])
                    cont_loss = cont_loss + contrastive_loss(
                        out2["proj"][0], proj_mix_0
                    )

                known_0 = ct_batch[0] != unknown_class
                if known_0.any():
                    clf_loss = cls_criterion(
                        out1["y"][0][known_0], ct_batch[0][known_0]
                    )
                    loss = cont_loss + kl_loss * kl_weight + clf_loss
                else:
                    loss = cont_loss + kl_loss * kl_weight

                # Non-reference species
                for s in range(1, n_species):
                    if gmm_prior is not None:
                        kl_s = gmm_prior.kl_divergence(
                            out1["qz"][s], out1["z"][s]
                        ).mean()
                    else:
                        kl_s = kl(out1["qz"][s], pz).sum(dim=1).mean()

                    c_s = contrastive_loss(out1["proj"][s], out2["proj"][s])

                    # Latent Mixup for non-reference species
                    if mixup_alpha > 0.0:
                        z_mix_s, _ = latent_mixup(out1["z"][s], alpha=mixup_alpha)
                        proj_mix_s = model.projecter(z_mix_s, batch[s])
                        c_s = c_s + contrastive_loss(out2["proj"][s], proj_mix_s)

                    ood_s, _ = ood_alignment(out1["proj"][s], out1["proj"][s - 1])
                    # g_s = graph_reg(out1["proj"][s], out1["proj"][s - 1])
                    v_s = 0.1 * vicreg(out1["proj"][s], out1["proj"][s - 1])

                    known_s = ct_batch[s] != unknown_class
                    if known_s.any():
                        clf_s = cls_criterion(out1["y"][s][known_s], ct_batch[s][known_s])
                        loss += c_s + kl_s * kl_weight + ood_s * beta + clf_s + v_s # + g_s
                    else:
                        loss += c_s + kl_s * kl_weight + ood_s * beta + v_s # + g_s

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
    kl_anneal: bool = True,
    kl_anneal_schedule: str = "cyclical",
    kl_anneal_cycles: int = 4,
    kl_anneal_ratio: float = 0.5,
    use_gmm_prior: bool = False,
    n_gmm_components: int = 10,
    mixup_alpha: float = 0.0,
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
        kl_anneal:         Anneal the KL weight instead of holding it fixed
                           at ``alpha`` from epoch 0.
        kl_anneal_schedule: ``"constant"``, ``"monotonic"``, or ``"cyclical"``.
        kl_anneal_cycles:  Number of ramp cycles (``"cyclical"`` only).
        kl_anneal_ratio:   Fraction of each cycle spent ramping up.
        use_gmm_prior:     Replace the standard ``N(0, I)`` prior with a
                           learnable ``K``-component GMM prior.
        n_gmm_components:  Number of GMM components ``K``.
        mixup_alpha:       Beta distribution concentration for latent Mixup.
                           ``0.0`` disables Mixup; ``0.4`` is a good default.

    Returns:
        List of per-epoch log-losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contrastive_loss = DCL(temperature=temp)

    if use_gmm_prior:
        gmm_prior = GMMPrior(
            n_components=n_gmm_components, n_latent=model.n_latent
        ).to(device)
        params = list(model.parameters()) + list(gmm_prior.parameters())
    else:
        gmm_prior = None
        params = list(model.parameters())

    opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=5e-4)

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

        kl_weight = alpha * (
            kl_annealing_weight(
                e, epoch, schedule=kl_anneal_schedule,
                n_cycles=kl_anneal_cycles, ratio=kl_anneal_ratio,
            ) if kl_anneal else 1.0
        )

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

                # Latent Mixup: additional augmented view from interpolated codes
                if mixup_alpha > 0.0:
                    z_mix, _ = latent_mixup(out1["z_a"], alpha=mixup_alpha)
                    proj_mix = model.projecter(z_mix, ba)
                    cont_loss = cont_loss + contrastive_loss(out2["proj_a"], proj_mix)

                # KL regularisation
                if gmm_prior is not None:
                    kl_loss = gmm_prior.kl_divergence(
                        out1["qz_a"], out1["z_a"]
                    ).mean()
                else:
                    pz = Normal(
                        torch.zeros_like(out1["qz_a"].mean),
                        torch.ones_like(out1["qz_a"].mean),
                    )
                    kl_loss = kl(out1["qz_a"], pz).sum(dim=1).mean()

                known = ct != unknown_class
                if known.any():
                    clf_loss = cls_criterion(out1["y_a"][known], ct[known])
                    loss = cont_loss + kl_loss * kl_weight + clf_loss
                else:
                    loss = cont_loss + kl_loss * kl_weight

                loss.backward()
                opt.step()
                total_loss.append(loss)

        loss_history.append(sum(total_loss).log().cpu().detach().numpy())

    return loss_history


# ---------------------------------------------------------------------------
# Joint cross-species + cross-modality integration
# ---------------------------------------------------------------------------

def _translate_entity(
    model,
    entity_idx: int,
    x: torch.Tensor,
    param_setup: str,
    species_id: int = 0,
    modality_id: int = 0,
) -> torch.Tensor:
    """
    Translate entity data to the shared feature space using that entity's
    learned translator.

    Dispatch logic:

    * ``multivibJoint`` (duck-typed via ``hasattr(model, "translate")``):
      calls ``model.translate(x, species_id, modality_id)`` so the correct
      per-``(species, modality)`` translator is always used.
    * ``multivibS`` (``param_setup="1st"``): uses
      ``model.translators[entity_idx]``.
    * ``multivibLoRAS`` (``param_setup="2nd"``): uses the shared ``B`` matrix
      with the per-entity ``A`` matrix: ``BN(B(A_i(x)))``.

    Args:
        model:       The multiVIB model instance.
        entity_idx:  Integer index of the entity in the current entity list.
        x:           Input tensor for this entity.
        param_setup: ``"1st"`` or ``"2nd"`` (ignored for ``multivibJoint``).
        species_id:  Integer species label (used only for ``multivibJoint``).
        modality_id: Integer modality label (used only for ``multivibJoint``).
    """
    # multivibJoint exposes model.translate(); dispatch by (species, modality)
    if hasattr(model, "translate"):
        return model.translate(x, species_id, modality_id)
    if param_setup == "1st":
        return model.translators[entity_idx](x)
    # LoRAS: shared B matrix, per-entity A matrix
    return model.batchnorm(model.matrixB(model.matrixA[entity_idx](x)))


def multivib_joint_training(
    model,
    Xs,
    batches,
    cell_types,
    species_ids,
    modality_ids,
    paired_map=None,
    epoch: int = 100,
    batch_size: int = 128,
    temp: float = 0.15,
    alpha: float = 0.05,
    beta_modal: float = 0.2,
    beta_species: float = 0.1,
    crossover_rate: float = 0.0,
    gaussian_rate_var: float = 1.0,
    random_seed: int = 0,
    kl_anneal: bool = True,
    kl_anneal_schedule: str = "cyclical",
    kl_anneal_cycles: int = 4,
    kl_anneal_ratio: float = 0.5,
    param_setup: str = "1st",
    use_gmm_prior: bool = False,
    n_gmm_components: int = 10,
    mixup_alpha: float = 0.0,
):
    """
    Train a multivibS or multivibLoRAS model for simultaneous cross-species
    and cross-modality integration.

    Each dataset is identified by two indices: ``species_ids[i]`` (organism)
    and ``modality_ids[i]`` (measurement type, e.g. 0=RNA, 1=ATAC).  The
    function automatically discovers which entities share a species
    (cross-modal alignment) and which share a modality (cross-species
    alignment), then applies the appropriate loss to each group.

    Alignment structure
    -------------------
    Two independent OOD-alignment modules are maintained so their prototype
    banks do not interfere:

    * ``ood_modal``: aligns entities with the **same species** but
      **different modality**.  Weighted by ``beta_modal``.  If paired cells
      are provided via ``paired_map``, a direct DCL loss is added for those
      pairs in addition to the unpaired OOD loss.

    * ``ood_species``: aligns entities with the **same modality** but
      **different species**.  Chain alignment within each modality group.
      Weighted by ``beta_species``.

    Full per-step loss::

        sum_entity  [DCL_self + KL * kl_weight + VICReg * 0.1 + clf]
      + sum_modal_pair   [OOD_modal * beta_modal]
      + sum_species_pair [OOD_species * beta_species]
      + sum_paired_pair  [DCL_paired]        (if paired_map provided)
      + optional Mixup DCL per entity

    Args:
        model:          A multivibS or multivibLoRAS instance.  Entity ``i``
                        maps to ``model.translators[i]`` (or
                        ``model.matrixA[i]`` for LoRAS).
        Xs:             List of data matrices, one per entity.
        batches:        List of batch-covariate arrays, one per entity.
        cell_types:     List of cell-type label arrays ("Unknown" for
                        unlabelled cells).
        species_ids:    Integer species index for each entity.  Entities
                        sharing a species index receive cross-modal alignment.
        modality_ids:   Integer modality index for each entity.  Entities
                        sharing a modality index receive cross-species alignment.
        paired_map:     Optional dict {(entity_i, entity_j): data} for entity
                        pairs with jointly-profiled cells.  Keys must use
                        (min_idx, max_idx) ordering.  Each value is a dict
                        with keys "Xa", "Xb", "ba", "bb" -- numpy arrays of
                        paired-cell data and batch covariates for entities i
                        and j respectively.
        epoch:          Number of training epochs.
        batch_size:     Mini-batch size.
        temp:           Contrastive-loss temperature.
        alpha:          KL loss peak weight.
        beta_modal:     Cross-modal OOD alignment weight.
        beta_species:   Cross-species OOD alignment weight.
        crossover_rate: CrossOver augmentation rate (0 = disabled).
        gaussian_rate_var: Gaussian noise std added to inputs.
        random_seed:    Base random seed.  Each entity is shuffled
                        independently via seed + epoch + entity * 10000.
        kl_anneal:      Enable KL weight annealing.
        kl_anneal_schedule: "constant", "monotonic", or "cyclical".
        kl_anneal_cycles:  Cycles for cyclical annealing.
        kl_anneal_ratio:   Ramp fraction per cycle.
        param_setup:    "1st" for multivibS; "2nd" for multivibLoRAS.
        use_gmm_prior:  Replace N(0,I) KL with a learnable GMM prior.
        n_gmm_components: GMM component count (use_gmm_prior=True only).
        mixup_alpha:    Beta-distribution concentration for latent Mixup
                        (0 = disabled; 0.4 is a good default).

    Returns:
        List of per-epoch log-losses.

    Example
    -------
    Two species, each measured with RNA (modality 0) and ATAC (modality 1).
    Species 0 has jointly-profiled co-assay RNA+ATAC cells::

        model = multivibS(
            n_input=[n_rna_sp0, n_atac_sp0, n_rna_sp1, n_atac_sp1],
            n_shared_input=1000, n_latent=20, n_class=10,
        )
        multivib_joint_training(
            model,
            Xs           = [rna_sp0, atac_sp0, rna_sp1, atac_sp1],
            batches      = [b_r0,    b_a0,     b_r1,    b_a1],
            cell_types   = [ct_r0,   ct_a0,    ct_r1,   ct_a1],
            species_ids  = [0, 0, 1, 1],
            modality_ids = [0, 1, 0, 1],
            paired_map   = {
                (0, 1): {"Xa": rna_paired, "Xb": atac_paired,
                         "ba": b_rna_paired, "bb": b_atac_paired},
            },
        )
    """
    from collections import defaultdict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_entities = len(Xs)

    # ------ Precompute alignment pairs -----------------------------------
    # cross_modal_pairs: consecutive entity pairs within the same species.
    # These entities share a species but differ in modality and need
    # within-species cross-modal alignment.
    modal_groups: dict = defaultdict(list)
    for e, s in enumerate(species_ids):
        modal_groups[s].append(e)
    cross_modal_pairs = []
    for ents in modal_groups.values():
        for k in range(len(ents) - 1):
            cross_modal_pairs.append((ents[k], ents[k + 1]))

    # cross_species_pairs: consecutive entity pairs within the same modality.
    # These entities measure the same modality but from different species.
    species_groups: dict = defaultdict(list)
    for e, m in enumerate(modality_ids):
        species_groups[m].append(e)
    cross_species_pairs = []
    for ents in species_groups.values():
        for k in range(len(ents) - 1):
            cross_species_pairs.append((ents[k], ents[k + 1]))

    # ------ Cell-type encoding -------------------------------------------
    ct_flat = np.concatenate([np.asarray(c) for c in cell_types])
    ct_enc = LabelEncoder()
    ct_enc.fit(ct_flat)
    unknown_class = ct_enc.transform(["Unknown"])[0]
    encoded_flat = ct_enc.transform(ct_flat)
    classes = np.unique(encoded_flat)
    cw = class_weight.compute_class_weight("balanced", classes=classes, y=encoded_flat)
    cw[classes == unknown_class] = 0.0
    cls_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(device)
    )
    # Split encoded labels back into per-entity arrays
    encoded_ct, off = [], 0
    for Xi in Xs:
        encoded_ct.append(encoded_flat[off: off + Xi.shape[0]])
        off += Xi.shape[0]

    # ------ Loss modules -------------------------------------------------
    contrastive_loss = DCL(temperature=temp)
    # Two independent OOD modules so prototype banks do not cross-contaminate.
    ood_modal_align = OODAlignmentLoss(
        n_prototypes=64, latent_dim=64, sinkhorn_eps=0.05,
        ot_weight=1.0, cluster_momentum=0.99, use_pseudo_labels_for_ot=False,
    ).to(device)
    ood_species_align = OODAlignmentLoss(
        n_prototypes=64, latent_dim=64, sinkhorn_eps=0.05,
        ot_weight=1.0, cluster_momentum=0.99, use_pseudo_labels_for_ot=False,
    ).to(device)
    vicreg = VICRegLoss()

    # ------ Optimiser ----------------------------------------------------
    # multivibJoint stores all translators in nn.ModuleDict, so
    # model.parameters() already covers them — no extra loops needed.
    params = list(model.parameters())
    if not hasattr(model, "translate"):
        # multivibS / multivibLoRAS: translators live outside model.parameters()
        if param_setup == "1st":
            for t in model.translators:
                params += list(t.parameters())
        elif param_setup == "2nd":
            for t in model.matrixA:
                params += list(t.parameters())

    if use_gmm_prior:
        gmm_prior = GMMPrior(
            n_components=n_gmm_components, n_latent=model.n_latent
        ).to(device)
        params += list(gmm_prior.parameters())
    else:
        gmm_prior = None

    opt = torch.optim.AdamW(params, lr=6e-4, weight_decay=5e-4)

    # ------ Move model to device -----------------------------------------
    model.to(device)
    if not hasattr(model, "translate"):
        if param_setup == "1st":
            for t in model.translators:
                t.to(device)
        elif param_setup == "2nd":
            for t in model.matrixA:
                t.to(device)

    # ------ Pre-tensorise paired data ------------------------------------
    # Keys are always (min_idx, max_idx); "Xa" / "ba" belong to the lower-
    # indexed entity, "Xb" / "bb" to the higher-indexed entity.
    paired_tensors: dict = {}
    if paired_map is not None:
        for (ei, ej), pdata in paired_map.items():
            key = (min(ei, ej), max(ei, ej))
            paired_tensors[key] = {
                "Xa": torch.tensor(pdata["Xa"]).float(),
                "Xb": torch.tensor(pdata["Xb"]).float(),
                "ba": torch.tensor(pdata["ba"]).float(),
                "bb": torch.tensor(pdata["bb"]).float(),
            }

    n = min(x.shape[0] for x in Xs)
    loss_history = []

    for e in range(epoch):
        kl_weight = alpha * (
            kl_annealing_weight(
                e, epoch, schedule=kl_anneal_schedule,
                n_cycles=kl_anneal_cycles, ratio=kl_anneal_ratio,
            ) if kl_anneal else 1.0
        )

        # Each entity is shuffled with an independent seed so cell-to-cell
        # correspondence is NOT assumed across entities (fully unpaired).
        X_tensor, y_tensor, ct_tensor = [], [], []
        for j in range(n_entities):
            ni = Xs[j].shape[0]
            r = np.random.RandomState(random_seed + e + j * 10000).permutation(ni)
            X_tensor.append(torch.tensor(Xs[j][r]).float())
            y_tensor.append(torch.tensor(batches[j][r]).float())
            ct_tensor.append(torch.tensor(encoded_ct[j][r], dtype=torch.long))

        # Paired data shuffled with a separate seed (decoupled from unpaired)
        paired_shuffled: dict = {}
        for key, pdata in paired_tensors.items():
            np_pair = pdata["Xa"].shape[0]
            rp = np.random.RandomState(random_seed + e + 999999).permutation(np_pair)
            paired_shuffled[key] = {k: v[rp] for k, v in pdata.items()}

        total_loss = []

        with tqdm(total=n // batch_size, desc=f"Epoch {e+1}/{epoch}",
                  unit="batch", bar_format="{l_bar}{bar:20}{r_bar}",
                  leave=False) as pbar:

            for i in range(n // batch_size):
                pbar.update(1)
                opt.zero_grad()

                sl = slice(i * batch_size, (i + 1) * batch_size)

                # Two independently augmented views for every entity
                inputs1, inputs2, batch_list, ct_batch = [], [], [], []
                for j in range(n_entities):
                    x = X_tensor[j][sl].to(device)
                    b = y_tensor[j][sl].to(device)
                    c, feat = x.shape
                    x1 = crossover_augmentation(x, crossover_rate) + torch.normal(
                        0, gaussian_rate_var, (c, feat), device=device
                    )
                    x2 = crossover_augmentation(x, crossover_rate) + torch.normal(
                        0, gaussian_rate_var, (c, feat), device=device
                    )
                    inputs1.append(x1)
                    inputs2.append(x2)
                    batch_list.append(b)
                    ct_batch.append(ct_tensor[j][sl].to(device))

                # multivibJoint requires explicit (species, modality) routing
                if hasattr(model, "translate"):
                    out1 = model(inputs1, batch_list, species_ids, modality_ids)
                    out2 = model(inputs2, batch_list, species_ids, modality_ids)
                else:
                    out1 = model(inputs1, batch_list)
                    out2 = model(inputs2, batch_list)

                # Standard N(0,I) prior -- built once, reused across entities
                if gmm_prior is None:
                    pz = Normal(
                        torch.zeros_like(out1["qz"][0].mean),
                        torch.ones_like(out1["qz"][0].mean),
                    )

                # ------ Per-entity losses --------------------------------
                # Accumulated as a list then summed to keep the autograd graph
                # clean and handle the optional classification term uniformly.
                loss_terms = []

                for j in range(n_entities):
                    # Self-supervised contrastive (two augmented views)
                    c_j = contrastive_loss(out1["proj"][j], out2["proj"][j])

                    # Optional latent Mixup: interpolated 3rd view
                    if mixup_alpha > 0.0:
                        z_mix_j, _ = latent_mixup(out1["z"][j], alpha=mixup_alpha)
                        proj_mix_j = model.projecter(z_mix_j, batch_list[j])
                        c_j = c_j + contrastive_loss(out2["proj"][j], proj_mix_j)

                    # KL regularisation
                    if gmm_prior is not None:
                        kl_j = gmm_prior.kl_divergence(
                            out1["qz"][j], out1["z"][j]
                        ).mean()
                    else:
                        kl_j = kl(out1["qz"][j], pz).sum(dim=1).mean()

                    # VICReg: prevent dimensional collapse within entity j
                    vic_j = 0.1 * vicreg(out1["proj"][j], out2["proj"][j])

                    term_j = c_j + kl_j * kl_weight + vic_j

                    # Supervised classification (zero-weighted for "Unknown")
                    known_j = ct_batch[j] != unknown_class
                    if known_j.any():
                        term_j = term_j + cls_criterion(
                            out1["y"][j][known_j], ct_batch[j][known_j]
                        )

                    loss_terms.append(term_j)

                # ------ Cross-modal alignment (within same species) ------
                # OOD (Sinkhorn OT) pulls projections from different modalities
                # of the same organism together.
                for (ei, ej) in cross_modal_pairs:
                    ood_m, _ = ood_modal_align(out1["proj"][ei], out1["proj"][ej])
                    loss_terms.append(ood_m * beta_modal)

                # ------ Cross-species alignment (within same modality) ---
                # OOD pulls the same modality across different species together.
                # Uses a separate prototype bank from the cross-modal module.
                for (ei, ej) in cross_species_pairs:
                    ood_s, _ = ood_species_align(out1["proj"][ei], out1["proj"][ej])
                    loss_terms.append(ood_s * beta_species)

                # ------ Paired-cell DCL (strongest within-species signal) -
                # For entity pairs with jointly-profiled cells, direct DCL on
                # matched projections is far stronger than OOD alignment.
                # This fires in addition to the OOD cross-modal loss above.
                for (key_i, key_j), pdata in paired_shuffled.items():
                    p_sl = slice(i * batch_size, (i + 1) * batch_size)
                    if p_sl.stop > pdata["Xa"].shape[0]:
                        continue  # paired data exhausted for this epoch

                    xp_i = pdata["Xa"][p_sl].to(device)
                    xp_j = pdata["Xb"][p_sl].to(device)
                    bp_i = pdata["ba"][p_sl].to(device)
                    bp_j = pdata["bb"][p_sl].to(device)

                    # Each entity has its own translator; shared encoder + projector.
                    # For multivibJoint, pass the (species, modality) labels so
                    # _translate_entity dispatches to the correct ModuleDict entry.
                    si_i = species_ids[key_i] if hasattr(model, "translate") else 0
                    mi_i = modality_ids[key_i] if hasattr(model, "translate") else 0
                    si_j = species_ids[key_j] if hasattr(model, "translate") else 0
                    mi_j = modality_ids[key_j] if hasattr(model, "translate") else 0
                    xt_i = _translate_entity(
                        model, key_i, xp_i, param_setup, si_i, mi_i
                    )
                    xt_j = _translate_entity(
                        model, key_j, xp_j, param_setup, si_j, mi_j
                    )
                    _, zp_i = model.encoder(xt_i)
                    _, zp_j = model.encoder(xt_j)
                    pp_i = model.projecter(zp_i, bp_i)
                    pp_j = model.projecter(zp_j, bp_j)

                    loss_terms.append(contrastive_loss(pp_i, pp_j))

                loss = sum(loss_terms)
                loss.backward()
                opt.step()
                total_loss.append(loss)

        loss_history.append(sum(total_loss).log().cpu().detach().numpy())

    return loss_history
