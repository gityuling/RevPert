"""Load perturbation response galleries (predicted or observed ΔY)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd


def clean_ko(name: str) -> str:
    return str(name).replace("+ctrl", "").strip()


def load_prediction_dir(pred_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load ``all_predictions.json`` + ``gene_names.json`` (absolute expression)."""
    pred_dir = Path(pred_dir)
    genes = json.loads((pred_dir / "gene_names.json").read_text())
    raw = json.loads((pred_dir / "all_predictions.json").read_text())
    pred: dict[str, np.ndarray] = {}
    for key, vec in raw.items():
        ko = clean_ko(key)
        if ko.lower() in {"ctrl", "control", "non-targeting", "nt"}:
            continue
        arr = np.asarray(vec, dtype=float)
        if arr.shape[0] != len(genes):
            raise ValueError(f"{key}: pred length {arr.shape[0]} != n_genes {len(genes)}")
        pred[ko] = arr
    return genes, pred


def _align_vector(values: np.ndarray, source_genes: list[str], genes: list[str]) -> np.ndarray:
    gene_to_idx = {g: i for i, g in enumerate(source_genes)}
    return np.array(
        [values[gene_to_idx[g]] if g in gene_to_idx else np.nan for g in genes],
        dtype=float,
    )


def load_ctrl_vector(genes: list[str], dataset_h5ad: Path | None = None) -> np.ndarray:
    """Load control mean expression aligned to ``genes``.

    Preference order:
    1. Sibling ``eval_pseudobulk_means.h5ad`` (condition == ctrl)
    2. ``perturb_processed.h5ad`` with ``gene_name`` or gene-like ``var_names``
    """
    if dataset_h5ad is None:
        raise ValueError("dataset_h5ad is required to locate ctrl")
    dataset_h5ad = Path(dataset_h5ad)
    candidates = [
        dataset_h5ad.parent / "eval_pseudobulk_means.h5ad",
        dataset_h5ad,
    ]
    errors: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            adata = ad.read_h5ad(path)
            if "gene_name" in adata.var.columns:
                source_genes = list(adata.var["gene_name"].astype(str).values)
            else:
                source_genes = list(map(str, adata.var_names))
            # Skip numeric placeholder var names on raw perturb_processed
            if sum(g.isdigit() for g in source_genes[:20]) > 10:
                errors.append(f"{path.name}: numeric var_names")
                continue
            if path.name.startswith("eval_pseudobulk") or "condition" in adata.obs.columns:
                cond = adata.obs["condition"].astype(str)
                rows = cond == "ctrl"
                if not rows.any() and "clean_condition" in adata.obs.columns:
                    rows = adata.obs["clean_condition"].astype(str) == "ctrl"
                if not rows.any():
                    errors.append(f"{path.name}: no ctrl row")
                    continue
                ctrl = np.asarray(adata[rows].X.mean(axis=0)).ravel()
            else:
                rows = adata.obs["condition"].astype(str) == "ctrl"
                if not rows.any():
                    errors.append(f"{path.name}: no ctrl cells")
                    continue
                ctrl = np.asarray(adata[rows].X.mean(axis=0)).ravel()
            out = _align_vector(ctrl, source_genes, genes)
            if np.isfinite(out).mean() < 0.5:
                errors.append(f"{path.name}: poor gene overlap")
                continue
            return out
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.name}: {exc}")
    raise ValueError("Could not load ctrl; tried: " + "; ".join(errors))


def load_ctrl_from_perturb_processed(
    dataset_h5ad: Path, genes: list[str]
) -> np.ndarray:
    """Backward-compatible alias for :func:`load_ctrl_vector`."""
    return load_ctrl_vector(genes, dataset_h5ad=dataset_h5ad)


def absolute_to_delta(
    pred_abs: dict[str, np.ndarray], ctrl: np.ndarray
) -> dict[str, np.ndarray]:
    """Convert absolute expression predictions to ΔY = pred - ctrl."""
    return {ko: vec - ctrl for ko, vec in pred_abs.items()}


def load_observed_deltas(
    pseudobulk_deltas_h5ad: Path, genes: list[str] | None = None
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load observed pseudobulk ΔY from ``all_pseudobulk_deltas.h5ad``."""
    adata = ad.read_h5ad(pseudobulk_deltas_h5ad)
    X = np.asarray(adata.layers["change"] if "change" in adata.layers else adata.X)
    var_genes = list(map(str, adata.var_names))
    if genes is None:
        genes = var_genes
        gene_to_idx = {g: i for i, g in enumerate(var_genes)}
        align = lambda row: row  # noqa: E731
    else:
        gene_to_idx = {g: i for i, g in enumerate(var_genes)}

        def align(row: np.ndarray) -> np.ndarray:
            return np.array(
                [row[gene_to_idx[g]] if g in gene_to_idx else np.nan for g in genes],
                dtype=float,
            )

    cond_col = "perturbed_gene" if "perturbed_gene" in adata.obs.columns else "condition"
    obs_by_ko: dict[str, np.ndarray] = {}
    labels = adata.obs[cond_col].astype(str).map(clean_ko)
    for ko in labels.unique():
        if ko.lower() in {"ctrl", "control"}:
            continue
        rows = labels == ko
        mean_delta = np.asarray(X[rows].mean(axis=0)).ravel()
        obs_by_ko[ko] = align(mean_delta)
    return genes, obs_by_ko


def gallery_to_matrix(
    gallery: dict[str, np.ndarray], genes: list[str], kos: Iterable[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Stack gallery into (n_genes × n_kos) matrix."""
    ko_list = list(kos) if kos is not None else sorted(gallery.keys())
    mat = np.column_stack([gallery[k] for k in ko_list])
    if mat.shape[0] != len(genes):
        raise ValueError("gallery vectors must match gene list length")
    return ko_list, mat


def load_coverage_from_pca_tsv(pca_tsv: Path) -> set[str]:
    """Column names of a reference/stack P TSV (excluding empty)."""
    header = Path(pca_tsv).read_text().splitlines()[0].split("\t")
    return {clean_ko(c) for c in header if c and clean_ko(c).lower() != "ctrl"}


def write_gallery_tsv(
    path: Path, genes: list[str], gallery: dict[str, np.ndarray], kos: list[str] | None = None
) -> None:
    """Optional export: genes × KOs TSV of ΔY."""
    ko_list, mat = gallery_to_matrix(gallery, genes, kos)
    df = pd.DataFrame(mat, index=genes, columns=ko_list)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t")
