"""
Reference-mapping helpers for multiVIB.

Given a trained multiVIB model and a reference AnnData, these helpers align a
query AnnData to the reference feature space, embed both datasets into the
shared multiVIB latent space, and identify the reference nearest neighbours of
every query cell (optionally transferring cell-type labels by majority vote).

Functions
---------
match_features          Align query features (var_names) to the reference.
get_embedding           Embed a data matrix into the multiVIB latent space.
map_query_to_reference  Full pipeline: match → embed → KNN (→ label transfer).
map_species_query_to_reference
                        Same pipeline for multi-species models, with one
                        reference/query AnnData per species in translator order.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from .utils import scale_by_batch


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------

def match_features(query, reference, verbose: bool = True):
    """
    Align a query AnnData to the reference feature space.

    Features are re-ordered to the reference ``var_names`` order; reference
    features missing from the query are zero-filled so the output matrix has
    exactly the shape the model was trained on.

    Args:
        query:     Query ``AnnData``.
        reference: Reference ``AnnData`` whose ``var_names`` define the target
                   feature space.
        verbose:   Print the feature-overlap summary.

    Returns:
        A new ``AnnData`` with ``var_names`` identical to the reference
        (same order), ``obs`` copied from the query, and a dense ``.X``.
    """
    import anndata

    ref_features = reference.var_names
    shared = query.var_names.intersection(ref_features)
    if len(shared) == 0:
        raise ValueError(
            "Query and reference share no features - check that var_names "
            "use the same identifier scheme (e.g. gene symbols vs Ensembl IDs)."
        )
    if verbose:
        print(
            f"Feature matching: {len(shared)}/{len(ref_features)} reference "
            f"features found in query "
            f"({len(ref_features) - len(shared)} zero-filled)."
        )

    X = np.zeros((query.n_obs, len(ref_features)), dtype=np.float32)
    ref_idx = {f: i for i, f in enumerate(ref_features)}
    cols = [ref_idx[f] for f in shared]
    Xq = query[:, shared].X
    if not isinstance(Xq, np.ndarray):
        Xq = Xq.toarray()
    X[:, cols] = Xq

    matched = anndata.AnnData(
        X=X, obs=query.obs.copy(), var=reference.var.copy()
    )
    return matched


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _translate(model: nn.Module, x: torch.Tensor, translate) -> torch.Tensor:
    """
    Route ``x`` through the appropriate translator of ``model``.

    ``translate`` selects the path into the shared encoder input space:
        * ``None``            — no translation (modality-A input of
                                :class:`multivib` / :class:`multivibLoRA`,
                                or :class:`multivibR`).
        * ``True``            — the bi-modal ``model.translator``
                                (modality-B input).
        * ``int`` *i*         — species *i*: ``model.translators[i]``
                                (:class:`multivibS`) or the shared low-rank
                                path (:class:`multivibLoRAS`).
        * ``(sid, mid)`` tuple — :class:`multivibJoint` routing via
                                ``model.translate``.
    """
    if translate is None:
        return x
    if translate is True:
        return model.translator(x)
    if isinstance(translate, tuple):
        return model.translate(x, *translate)
    if isinstance(translate, int):
        if hasattr(model, "translators"):
            return model.translators[translate](x)
        # multivibLoRAS: shared B / batchnorm, species-specific A
        return model.batchnorm(model.matrixB(model.matrixA[translate](x)))
    raise TypeError(f"Unsupported translate argument: {translate!r}")


def get_embedding(
    model: nn.Module,
    X: np.ndarray,
    translate=None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 1024,
    use_mean: bool = True,
) -> np.ndarray:
    """
    Embed a (preprocessed) data matrix into the multiVIB latent space.

    Args:
        model:      Trained multiVIB model.
        X:          Data matrix, shape ``(n_cells, n_features)``, already
                    scaled the same way as during training.
        translate:  Translator routing — see :func:`_translate`.
        device:     Device to run inference on.
        batch_size: Minibatch size for inference.
        use_mean:   If ``True`` (default) return the posterior mean ``qz.mean``
                    (deterministic); otherwise a reparameterised sample, as in
                    the tutorials.

    Returns:
        Latent embedding, shape ``(n_cells, n_latent)``.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    zs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(
                np.asarray(X[start : start + batch_size]), dtype=torch.float32
            ).to(device)
            xt = _translate(model, xb, translate)
            qz, z = model.encoder(xt)
            zs.append((qz.mean if use_mean else z).cpu().numpy())
    return np.concatenate(zs, axis=0)


# ---------------------------------------------------------------------------
# Shared preprocessing / label-transfer helpers
# ---------------------------------------------------------------------------

def _prep(adata, batch_key: Optional[str], scale: bool) -> np.ndarray:
    """Densify ``.X`` and optionally z-score per ``obs[batch_key]`` batch."""
    X = adata.X
    X = np.asarray(X.toarray() if not isinstance(X, np.ndarray) else X)
    if not scale:
        return X
    labels = (
        adata.obs[batch_key].values
        if batch_key is not None and batch_key in adata.obs
        else np.zeros(adata.n_obs)
    )
    return scale_by_batch(X, labels)


def _majority_vote(
    ref_labels: np.ndarray, knn_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Majority-vote label transfer; returns (predicted, vote fraction)."""
    neighbor_labels = ref_labels[knn_indices]  # (n_query, n_neighbors)
    predicted, confidence = [], []
    for row in neighbor_labels:
        values, counts = np.unique(row, return_counts=True)
        best = counts.argmax()
        predicted.append(values[best])
        confidence.append(counts[best] / len(row))
    return np.asarray(predicted), np.asarray(confidence)


# ---------------------------------------------------------------------------
# Full pipeline: match features → embed → KNN (→ label transfer)
# ---------------------------------------------------------------------------

def map_query_to_reference(
    model: nn.Module,
    reference,
    query,
    translate_ref=None,
    translate_query=None,
    batch_key: Optional[str] = None,
    scale: bool = True,
    n_neighbors: int = 15,
    metric: str = "correlation",
    label_key: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 1024,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Map a query AnnData onto a reference with a trained multiVIB model.

    Pipeline:
        1. Align query features to the reference feature space
           (:func:`match_features`).
        2. Z-score both datasets (per ``batch_key`` batch if given, matching
           the training convention of :func:`~multivib.utils.scale_by_batch`).
        3. Embed reference and query into the shared latent space
           (:func:`get_embedding`); embeddings are stored in
           ``.obsm["X_multivib"]`` of both AnnData objects.
        4. Find the ``n_neighbors`` nearest reference cells of every query
           cell in latent space.
        5. If ``label_key`` is given, transfer labels by KNN majority vote.

    Args:
        model:           Trained multiVIB model.
        reference:       Reference ``AnnData`` (features define the model
                         input space; assumed already normalised, e.g.
                         log1p-CPM, as used for training).
        query:           Query ``AnnData`` with the same normalisation.
        translate_ref:   Translator routing for the reference — ``None``,
                         ``True``, species ``int``, or ``(sid, mid)`` tuple
                         (see :func:`get_embedding`).
        translate_query: Translator routing for the query, same options.
        batch_key:       ``obs`` column with batch labels for per-batch
                         scaling; if ``None`` each dataset is scaled as a
                         single batch.
        scale:           Set ``False`` if ``.X`` is already z-scored.
        n_neighbors:     Number of reference neighbours per query cell.
        metric:          KNN distance metric (default ``"correlation"``, as
                         in the tutorials).
        label_key:       Optional reference ``obs`` column to transfer to the
                         query by majority vote.
        device:          Device for model inference.
        batch_size:      Minibatch size for inference.
        verbose:         Print progress messages.

    Returns:
        dict with:
            * ``"query_matched"``  — feature-matched query ``AnnData`` (with
              ``.obsm["X_multivib"]`` and, if ``label_key`` was given,
              ``.obs["predicted_<label_key>"]`` / ``"..._confidence"``).
            * ``"ref_embedding"`` / ``"query_embedding"`` — latent matrices.
            * ``"knn_indices"``    — ``(n_query, n_neighbors)`` integer
              positions into the reference.
            * ``"knn_distances"``  — matching distance matrix.
            * ``"predicted_labels"`` / ``"prediction_confidence"`` — only if
              ``label_key`` was given (confidence = vote fraction).
    """
    # 1. Feature matching -------------------------------------------------
    query_matched = match_features(query, reference, verbose=verbose)

    # 2. Scaling -----------------------------------------------------------
    X_ref = _prep(reference, batch_key, scale)
    X_query = _prep(query_matched, batch_key, scale)

    # 3. Embedding ---------------------------------------------------------
    if verbose:
        print(f"Embedding {reference.n_obs} reference and "
              f"{query_matched.n_obs} query cells...")
    z_ref = get_embedding(model, X_ref, translate_ref, device, batch_size)
    z_query = get_embedding(model, X_query, translate_query, device, batch_size)
    reference.obsm["X_multivib"] = z_ref
    query_matched.obsm["X_multivib"] = z_query

    # 4. KNN search --------------------------------------------------------
    nn_index = NearestNeighbors(
        n_neighbors=n_neighbors, metric=metric, algorithm="brute"
    ).fit(z_ref)
    knn_distances, knn_indices = nn_index.kneighbors(z_query)

    results: Dict[str, Any] = {
        "query_matched": query_matched,
        "ref_embedding": z_ref,
        "query_embedding": z_query,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
    }

    # 5. Optional label transfer -------------------------------------------
    if label_key is not None:
        predicted, confidence = _majority_vote(
            np.asarray(reference.obs[label_key]), knn_indices
        )
        query_matched.obs[f"predicted_{label_key}"] = predicted
        query_matched.obs[f"predicted_{label_key}_confidence"] = confidence
        results["predicted_labels"] = predicted
        results["prediction_confidence"] = confidence
        if verbose:
            print(f"Transferred '{label_key}' labels "
                  f"(mean vote confidence {np.mean(confidence):.2f}).")

    return results


# ---------------------------------------------------------------------------
# Multi-species pipeline (multivibS / multivibLoRAS)
# ---------------------------------------------------------------------------

def map_species_query_to_reference(
    model: nn.Module,
    references: List[Any],
    queries: List[Optional[Any]],
    batch_key: Optional[str] = None,
    scale: bool = True,
    n_neighbors: int = 15,
    metric: str = "correlation",
    label_key: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 1024,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Map multi-species query data onto a multi-species reference with a trained
    :class:`~multivib.models.multivibS` / :class:`~multivib.models.multivibLoRAS`
    model.

    ``references`` and ``queries`` are lists ordered like the model inputs
    ``[X_speciesA, X_speciesB, X_speciesC]`` — species ``i`` is routed through
    translator ``i`` (``model.translators[i]`` or the shared low-rank path).
    Each query is feature-matched against the reference of the *same species*,
    then all cells are embedded into the shared latent space, and every query
    cell's nearest neighbours are searched across the **combined** reference
    of all species.

    Args:
        model:       Trained multi-species multiVIB model.
        references:  One reference ``AnnData`` per species, in translator order
                     (e.g. ``[X_speciesA, None, None]``); ``None`` entries are
                     skipped — no embedding is computed for that species.
        queries:     One query ``AnnData`` per species (same order); use
                     ``None`` for species without query cells.  A query
                     requires the same-species reference to be present.
        batch_key:   ``obs`` column with batch labels for per-batch scaling;
                     if ``None`` each AnnData is scaled as a single batch.
        scale:       Set ``False`` if ``.X`` is already z-scored.
        n_neighbors: Number of reference neighbours per query cell.
        metric:      KNN distance metric.
        label_key:   Optional reference ``obs`` column to transfer to the
                     queries by majority vote (must exist in every reference).
        device:      Device for model inference.
        batch_size:  Minibatch size for inference.
        verbose:     Print progress messages.

    Returns:
        dict with:
            * ``"queries_matched"``  — list of feature-matched query
              ``AnnData`` (``None`` where the input was ``None``), each with
              ``.obsm["X_multivib"]`` and, if ``label_key`` was given,
              ``.obs["predicted_<label_key>"]`` / ``"..._confidence"``.
            * ``"ref_embedding"`` / ``"query_embedding"`` — concatenated
              latent matrices (species stacked in list order).
            * ``"ref_species"`` / ``"query_species"`` — species index of every
              row of the concatenated embeddings.
            * ``"knn_indices"``    — ``(n_query, n_neighbors)`` positions into
              the concatenated reference (use ``ref_species`` to recover which
              species a neighbour belongs to).
            * ``"knn_distances"``  — matching distance matrix.
            * ``"predicted_labels"`` / ``"prediction_confidence"`` — only if
              ``label_key`` was given, aligned with the concatenated query.

    Example::

        results = map_species_query_to_reference(
            model,
            references=[ref_human, ref_mouse, ref_marmoset],
            queries=[query_human, None, query_marmoset],
            label_key="cell_type",
        )
    """
    if len(queries) != len(references):
        raise ValueError(
            f"queries has {len(queries)} entries but references has "
            f"{len(references)} - both must list one entry per species, "
            "in the model's translator order (use None for species without "
            "query cells)."
        )

    # 1 + 2. Per-species feature matching and scaling ------------------------
    queries_matched: List[Optional[Any]] = []
    for i, (ref_i, query_i) in enumerate(zip(references, queries)):
        if query_i is None:
            queries_matched.append(None)
            continue
        if ref_i is None:
            raise ValueError(
                f"queries[{i}] is given but references[{i}] is None - a query "
                "can only be matched against a reference of the same species."
            )
        if verbose:
            print(f"Species {i}:", end=" ")
        queries_matched.append(match_features(query_i, ref_i, verbose=verbose))

    # 3. Embedding — species i goes through translator i; None entries
    #    (species without data) are skipped entirely.
    z_ref_list, ref_species = [], []
    z_query_list, query_species = [], []
    for i, ref_i in enumerate(references):
        if ref_i is None:
            continue
        z_i = get_embedding(
            model, _prep(ref_i, batch_key, scale), i, device, batch_size
        )
        ref_i.obsm["X_multivib"] = z_i
        z_ref_list.append(z_i)
        ref_species.append(np.full(ref_i.n_obs, i))
    for i, qm_i in enumerate(queries_matched):
        if qm_i is None:
            continue
        z_i = get_embedding(
            model, _prep(qm_i, batch_key, scale), i, device, batch_size
        )
        qm_i.obsm["X_multivib"] = z_i
        z_query_list.append(z_i)
        query_species.append(np.full(qm_i.n_obs, i))

    z_ref = np.concatenate(z_ref_list, axis=0)
    z_query = np.concatenate(z_query_list, axis=0)
    ref_species = np.concatenate(ref_species)
    query_species = np.concatenate(query_species)
    if verbose:
        print(f"Embedded {z_ref.shape[0]} reference and {z_query.shape[0]} "
              f"query cells across {len(references)} species.")

    # 4. KNN across the combined reference of all species --------------------
    nn_index = NearestNeighbors(
        n_neighbors=n_neighbors, metric=metric, algorithm="brute"
    ).fit(z_ref)
    knn_distances, knn_indices = nn_index.kneighbors(z_query)

    results: Dict[str, Any] = {
        "queries_matched": queries_matched,
        "ref_embedding": z_ref,
        "query_embedding": z_query,
        "ref_species": ref_species,
        "query_species": query_species,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
    }

    # 5. Optional label transfer ---------------------------------------------
    if label_key is not None:
        ref_labels = np.concatenate(
            [np.asarray(r.obs[label_key]) for r in references if r is not None]
        )
        predicted, confidence = _majority_vote(ref_labels, knn_indices)
        offset = 0
        for qm_i in queries_matched:
            if qm_i is None:
                continue
            sl = slice(offset, offset + qm_i.n_obs)
            qm_i.obs[f"predicted_{label_key}"] = predicted[sl]
            qm_i.obs[f"predicted_{label_key}_confidence"] = confidence[sl]
            offset += qm_i.n_obs
        results["predicted_labels"] = predicted
        results["prediction_confidence"] = confidence
        if verbose:
            print(f"Transferred '{label_key}' labels "
                  f"(mean vote confidence {np.mean(confidence):.2f}).")

    return results
