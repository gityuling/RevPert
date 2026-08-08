"""Build disease–control target shift ΔY*."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_vector_tsv(path: Path, gene_col: str = "gene", value_col: str = "value") -> pd.Series:
    """Load a gene→value table (TSV/CSV)."""
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    gcol = cols.get(gene_col.lower(), df.columns[0])
    vcol = cols.get(value_col.lower(), df.columns[1])
    s = pd.Series(df[vcol].astype(float).values, index=df[gcol].astype(str).values)
    return s.groupby(level=0).mean()


def delta_y_star_from_two_profiles(
    disease: pd.Series, control: pd.Series, mode: str = "control_minus_disease"
) -> pd.Series:
    """
    Build ΔY*.

    Parameters
    ----------
    mode :
        ``control_minus_disease`` — push disease toward control (default reverse goal).
        ``disease_minus_control`` — disease signature as usually reported in DEG.
    """
    genes = sorted(set(disease.index) & set(control.index))
    d = disease.reindex(genes).astype(float)
    c = control.reindex(genes).astype(float)
    if mode == "control_minus_disease":
        out = c - d
    elif mode == "disease_minus_control":
        out = d - c
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out.dropna()


def delta_y_star_from_gallery_ko(
    gallery: dict[str, np.ndarray], genes: list[str], ko: str
) -> pd.Series:
    """Use one KO's response vector as a synthetic ΔY* (oracle / smoke tests)."""
    if ko not in gallery:
        raise KeyError(f"KO {ko!r} not in gallery ({len(gallery)} keys)")
    return pd.Series(gallery[ko], index=genes, dtype=float)


def align_delta_y_star(
    delta_y_star: pd.Series, genes: list[str]
) -> np.ndarray:
    """Align ΔY* to gallery gene order; missing genes → NaN."""
    return delta_y_star.reindex(genes).astype(float).to_numpy()


def save_delta_y_star(path: Path, delta_y_star: pd.Series) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    delta_y_star.rename("value").rename_axis("gene").to_csv(path, sep="\t")
