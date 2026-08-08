"""Score gallery KO responses against ΔY* (reverse / connectivity-style)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _pair_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    x, y = a[mask], b[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if metric == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    if metric == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    if metric == "cosine":
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    raise ValueError(f"Unknown metric: {metric}")


def score_gallery(
    gallery: dict[str, np.ndarray],
    delta_y_star: np.ndarray,
    genes: list[str],
    metric: str = "pearson",
    covered: set[str] | None = None,
    primary_only_covered: bool = True,
) -> pd.DataFrame:
    """
    For each KO g, score s(g) = corr(ΔŶ(g), ΔY*).

    Higher score ⇒ predicted response more aligned with desired reverse shift.
    """
    if len(delta_y_star) != len(genes):
        raise ValueError("delta_y_star length must match genes")

    rows = []
    for ko, vec in gallery.items():
        if len(vec) != len(genes):
            raise ValueError(f"{ko}: bad vector length")
        is_cov = (ko in covered) if covered is not None else True
        s = _pair_metric(vec, delta_y_star, metric)
        n_overlap = int((np.isfinite(vec) & np.isfinite(delta_y_star)).sum())
        rows.append(
            {
                "ko": ko,
                "score": s,
                "metric": metric,
                "covered": bool(is_cov),
                "n_genes_scored": n_overlap,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Rank: higher score better; uncovered optionally pushed to bottom for primary rank
    df["score_for_rank"] = df["score"]
    if primary_only_covered and covered is not None:
        df.loc[~df["covered"], "score_for_rank"] = -np.inf
    df["rank"] = df["score_for_rank"].rank(ascending=False, method="min")
    df = df.drop(columns=["score_for_rank"])
    return df.sort_values(["rank", "ko"]).reset_index(drop=True)


def top_k(df: pd.DataFrame, k: int = 50, covered_only: bool = True) -> pd.DataFrame:
    sub = df[df["covered"]] if covered_only and "covered" in df.columns else df
    return sub.nsmallest(k, "rank")
