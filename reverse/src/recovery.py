"""Multi-KO reverse recovery evaluation (vectorized)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore_cols(mat: np.ndarray) -> np.ndarray:
    """Column-wise z-score; zero-variance columns → 0."""
    x = np.asarray(mat, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, np.nan, sd)
    z = (x - mu) / sd
    return np.nan_to_num(z, nan=0.0)


def pearson_rank_matrix(
    query_mat: np.ndarray, gallery_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    query_mat, gallery_mat :
        genes × n matrices (same gene axis). NaNs allowed.

    Returns
    -------
    scores : n_query × n_gallery Pearson correlations
    ranks : n_query × n_gallery ranks (1 = best / highest score) per query row
    """
    # Mask genes that are finite in both for each pair is expensive; use
    # intersection mask: gene finite in ALL columns of both (conservative) OR
    # per-column nan_to_num after gene-wise pairwise — use shared finite genes.
    q_ok = np.isfinite(query_mat).all(axis=1)
    g_ok = np.isfinite(gallery_mat).all(axis=1)
    ok = q_ok & g_ok
    if ok.sum() < 10:
        raise ValueError("Too few shared finite genes for recovery matrix")
    qz = _zscore_cols(query_mat[ok])
    gz = _zscore_cols(gallery_mat[ok])
    # With population z-score, Pearson r = (z_q · z_g) / n_genes
    scores = (qz.T @ gz) / ok.sum()
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)
    return scores, ranks


def run_multi_ko_recovery(
    query_kos: list[str],
    query_gallery: dict[str, np.ndarray],
    search_gallery: dict[str, np.ndarray],
    genes: list[str],
) -> pd.DataFrame:
    """
    For each query KO, take its vector from ``query_gallery`` as ΔY*,
    rank all KOs in ``search_gallery``.
    """
    search_kos = sorted(search_gallery.keys())
    q_kos = [k for k in query_kos if k in query_gallery and k in search_gallery]
    if not q_kos:
        raise ValueError("No overlapping query KOs")

    q_mat = np.column_stack([query_gallery[k] for k in q_kos])
    s_mat = np.column_stack([search_gallery[k] for k in search_kos])
    scores, ranks = pearson_rank_matrix(q_mat, s_mat)

    # index of true ko in search list
    search_index = {k: i for i, k in enumerate(search_kos)}
    rows = []
    for i, ko in enumerate(q_kos):
        j = search_index[ko]
        rows.append(
            {
                "true_ko": ko,
                "rank": int(ranks[i, j]),
                "score": float(scores[i, j]),
                "best_ko": search_kos[int(np.argmax(scores[i]))],
                "best_score": float(np.max(scores[i])),
                "n_gallery": len(search_kos),
            }
        )
    return pd.DataFrame(rows)


def summarize_recovery(df: pd.DataFrame) -> dict:
    r = df["rank"].astype(float)
    n = len(df)
    return {
        "n_query": n,
        "median_rank": float(r.median()),
        "mean_rank": float(r.mean()),
        "pct_top1": float((r <= 1).mean() * 100),
        "pct_top10": float((r <= 10).mean() * 100),
        "pct_top50": float((r <= 50).mean() * 100),
        "pct_top100": float((r <= 100).mean() * 100),
        "mean_score": float(df["score"].mean()),
        "median_score": float(df["score"].median()),
    }
