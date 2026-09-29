"""
Preprocessing helpers for multiVIB.

Total-count normalisation + log1p transformation of AnnData (via scanpy's
``sc.pp.normalize_total`` / ``sc.pp.log1p``), with a disk-backed streaming
mode for datasets too large to fit in memory.

Functions
---------
normalize_log1p        In-memory normalisation + log1p of an AnnData.
preprocess_h5ad        Streaming (backed) normalisation + log1p of an ``.h5ad``
                       file, writing the processed data to a new ``.h5ad``
                       chunk by chunk.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# In-memory version
# ---------------------------------------------------------------------------

def normalize_log1p(adata, target_sum: Optional[float] = 1e4):
    """
    Total-count normalise and log1p-transform an in-memory AnnData.

    Thin wrapper around ``sc.pp.normalize_total(adata, target_sum=...)``
    followed by ``sc.pp.log1p(adata)``; ``.X`` is modified in place.

    Args:
        adata:      AnnData with raw counts in ``.X``.
        target_sum: Counts each cell is scaled to.  If ``None``, the median of
                    per-cell totals is used (scanpy convention).

    Returns:
        The same AnnData, modified in place.
    """
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


# ---------------------------------------------------------------------------
# Disk-backed streaming version
# ---------------------------------------------------------------------------

def _normalize_log1p_chunk(X, target_sum: float) -> sp.csr_matrix:
    """
    Apply ``sc.pp.normalize_total`` + ``sc.pp.log1p`` to one row chunk.

    Both transforms are purely per-cell, so processing the matrix in row
    chunks is exact.  Returns float32 CSR regardless of input type.
    """
    import anndata
    import scanpy as sc

    chunk = anndata.AnnData(X=sp.csr_matrix(X, dtype=np.float32))
    sc.pp.normalize_total(chunk, target_sum=target_sum)
    sc.pp.log1p(chunk)
    return chunk.X.astype(np.float32)


def preprocess_h5ad(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_sum: Optional[float] = 1e4,
    chunk_size: int = 10000,
    verbose: bool = True,
) -> Path:
    """
    Normalise + log1p a (huge) ``.h5ad`` file in disk-backed streaming mode.

    The input is opened in backed read-only mode and processed
    ``chunk_size`` cells at a time with ``sc.pp.normalize_total`` /
    ``sc.pp.log1p``; the transformed matrix is appended directly to the
    output file as CSR, so peak memory stays at roughly one chunk regardless
    of dataset size.  ``obs``, ``var``, ``uns``, ``obsm`` and ``varm`` are
    copied over; ``layers`` and ``raw`` are not.

    Args:
        input_path:  Path to the raw-counts ``.h5ad``.
        output_path: Path for the processed ``.h5ad`` (overwritten if present).
        target_sum:  Counts each cell is scaled to.  If ``None``, a first
                     pass computes the median of per-cell totals (scanpy
                     convention) so all chunks share the same target.
        chunk_size:  Number of cells per chunk.
        verbose:     Print progress.

    Returns:
        The output path.

    Example::

        preprocess_h5ad("atlas_raw.h5ad", "atlas_processed.h5ad")
        reference = anndata.read_h5ad("atlas_processed.h5ad")  # or backed="r"
    """
    import anndata
    import h5py
    from anndata.experimental import write_elem

    input_path, output_path = Path(input_path), Path(output_path)
    adata = anndata.read_h5ad(input_path, backed="r")
    n_obs, n_vars = adata.shape
    if verbose:
        print(f"Backed mode: {n_obs} cells x {n_vars} features from {input_path}")

    def _chunks():
        for start in range(0, n_obs, chunk_size):
            yield start, adata.X[start : min(start + chunk_size, n_obs)]

    # Optional first pass: global median per-cell total, so every chunk is
    # scaled to the same target (sc.pp.normalize_total's median would
    # otherwise be computed per chunk).
    if target_sum is None:
        totals = np.concatenate(
            [np.asarray(X.sum(axis=1)).ravel() for _, X in _chunks()]
        )
        target_sum = float(np.median(totals[totals > 0]))
        if verbose:
            print(f"target_sum=None -> using median per-cell total: {target_sum:.1f}")

    with h5py.File(output_path, "w") as f:
        # Stream X chunk by chunk as an appendable CSR group
        g = f.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        g.attrs["shape"] = np.array([n_obs, n_vars], dtype=np.int64)
        data_ds = g.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype="float32"
        )
        indices_ds = g.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype="int64"
        )
        indptr_ds = g.create_dataset("indptr", shape=(n_obs + 1,), dtype="int64")
        indptr_ds[0] = 0

        nnz = 0
        for start, X_chunk in _chunks():
            csr = _normalize_log1p_chunk(X_chunk, target_sum)
            new_nnz = nnz + csr.nnz
            data_ds.resize((new_nnz,))
            indices_ds.resize((new_nnz,))
            data_ds[nnz:new_nnz] = csr.data
            indices_ds[nnz:new_nnz] = csr.indices
            indptr_ds[start + 1 : start + 1 + csr.shape[0]] = (
                nnz + csr.indptr[1:]
            )
            nnz = new_nnz
            if verbose:
                done = min(start + chunk_size, n_obs)
                print(f"  processed {done}/{n_obs} cells", end="\r")
        if verbose:
            print(f"\n  written {nnz} non-zeros to {output_path}")

        # Copy metadata (obs/var are in memory even in backed mode)
        uns = dict(adata.uns)
        uns["log1p"] = {"base": None}
        uns["normalize_total"] = {"target_sum": target_sum}
        write_elem(f, "obs", adata.obs)
        write_elem(f, "var", adata.var)
        write_elem(f, "uns", uns)
        write_elem(f, "obsm", dict(adata.obsm))
        write_elem(f, "varm", dict(adata.varm))

    adata.file.close()
    return output_path
