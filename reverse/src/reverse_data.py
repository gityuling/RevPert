"""Build reverse-retrieval datasets: observed ΔY queries + P prototypes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io_gallery import clean_ko, load_observed_deltas


def load_split_kos(split_path: Path) -> dict[str, list[str]]:
    raw = json.loads(Path(split_path).read_text())
    return {k: [clean_ko(x) for x in raw[k]] for k in ("train", "val", "test")}


def load_p_matrix(pca_tsv: Path) -> tuple[list[str], np.ndarray]:
    """Load perturbation embedding TSV: rows=dims, cols=KO names → (kos, n_ko×d)."""
    # Files are written as dims × kos with KO names in the header and no row index.
    df = pd.read_csv(pca_tsv, sep="\t")
    if df.shape[0] <= df.shape[1]:
        kos = [clean_ko(c) for c in df.columns.astype(str)]
        mat = df.to_numpy(dtype=float).T  # n_ko x d
    else:
        # fallback: kos as index
        df2 = pd.read_csv(pca_tsv, sep="\t", index_col=0)
        kos = [clean_ko(i) for i in df2.index.astype(str)]
        mat = df2.to_numpy(dtype=float)
    keep = [i for i, k in enumerate(kos) if k.lower() not in {"ctrl", "control", ""}]
    kos = [kos[i] for i in keep]
    mat = mat[keep]
    # drop all-zero P columns (uncovered in stack)
    norms = np.linalg.norm(mat, axis=1)
    keep2 = norms > 1e-8
    kos = [k for k, m in zip(kos, keep2) if m]
    mat = mat[keep2]
    return kos, mat


def assemble_reverse_tables(
    genes: list[str],
    obs_gallery: dict[str, np.ndarray],
    p_kos: list[str],
    p_mat: np.ndarray,
    split: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """
    For each split, keep KOs that have both observed ΔY and P.

    Returns dict of DataFrames with columns: ko, and arrays stored separately.
    """
    p_index = {k: i for i, k in enumerate(p_kos)}
    out: dict[str, dict] = {}
    for part, kos in split.items():
        rows_ko = []
        ys = []
        ps = []
        for ko in kos:
            if ko not in obs_gallery or ko not in p_index:
                continue
            y = obs_gallery[ko]
            if not np.isfinite(y).all():
                y = np.nan_to_num(y, nan=0.0)
            rows_ko.append(ko)
            ys.append(y)
            ps.append(p_mat[p_index[ko]])
        out[part] = {
            "kos": rows_ko,
            "Y": np.stack(ys, axis=0) if ys else np.zeros((0, len(genes))),
            "P": np.stack(ps, axis=0) if ps else np.zeros((0, p_mat.shape[1])),
        }
    return out


def load_reverse_bundle(
    pseudobulk_deltas: Path,
    gene_names_json: Path,
    split_path: Path,
    p_tsv: Path,
) -> tuple[list[str], dict, dict[str, list[str]]]:
    genes = json.loads(Path(gene_names_json).read_text())
    _, obs = load_observed_deltas(pseudobulk_deltas, genes)
    p_kos, p_mat = load_p_matrix(p_tsv)
    split = load_split_kos(split_path)
    tables = assemble_reverse_tables(genes, obs, p_kos, p_mat, split)
    meta = {
        "n_genes": len(genes),
        "p_dim": int(p_mat.shape[1]),
        "n_p_kos": len(p_kos),
        "counts": {k: len(v["kos"]) for k, v in tables.items()},
    }
    return genes, tables, meta
