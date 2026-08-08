"""Lightweight benchmarks for reverse scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .score import score_gallery


def sign_flip_scores(
    gallery: dict[str, np.ndarray],
    delta_y_star: np.ndarray,
    genes: list[str],
    metric: str = "pearson",
    covered: set[str] | None = None,
) -> pd.DataFrame:
    """Scores against -ΔY*; top genes should reshuffle vs forward scores."""
    return score_gallery(gallery, -delta_y_star, genes, metric=metric, covered=covered)


def random_null_max_score(
    gallery: dict[str, np.ndarray],
    delta_y_star: np.ndarray,
    genes: list[str],
    n_rand: int = 50,
    seed: int = 0,
    metric: str = "pearson",
    covered: set[str] | None = None,
) -> pd.DataFrame:
    """Max gallery score under randomly permuted ΔY* (same finite mask)."""
    rng = np.random.default_rng(seed)
    mask = np.isfinite(delta_y_star)
    base = delta_y_star.copy()
    vals = []
    for i in range(n_rand):
        fake = base.copy()
        fake[mask] = rng.permutation(base[mask])
        df = score_gallery(gallery, fake, genes, metric=metric, covered=covered)
        vals.append({"rep": i, "max_score": float(np.nanmax(df["score"].values))})
    return pd.DataFrame(vals)


def oracle_recovery_rank(
    scored: pd.DataFrame, true_ko: str
) -> dict:
    """Rank / score of the KO whose observed (or synthetic) ΔY* was used."""
    hit = scored[scored["ko"] == true_ko]
    if hit.empty:
        return {"true_ko": true_ko, "found": False, "rank": np.nan, "score": np.nan}
    row = hit.iloc[0]
    return {
        "true_ko": true_ko,
        "found": True,
        "rank": float(row["rank"]),
        "score": float(row["score"]),
        "covered": bool(row["covered"]),
        "n_gallery": int(len(scored)),
    }
