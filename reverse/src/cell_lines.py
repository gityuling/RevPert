"""Canonical paths for reverse Task-1 across Essential cell lines and GEARS seeds.

Override roots with environment variables when the analysis tree is relocated:
  REVPERT_REPO_ROOT  — repository root (default: parents[2] of this file)
  REVPERT_BENCH_ROOT — path to the forward-benchmark tree containing
                       data/gears_pert_data and working_dir/results
                       (default: <repo>/linear_perturbation_prediction-Paper-main/benchmark)
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(os.environ.get("REVPERT_REPO_ROOT", Path(__file__).resolve().parents[2]))
BENCH = Path(
    os.environ.get(
        "REVPERT_BENCH_ROOT",
        _ROOT / "linear_perturbation_prediction-Paper-main" / "benchmark",
    )
)
STACK = BENCH / "working_dir/results/progressive_stack_fulltest"
PCA_DIR = STACK / "reference_pca"
DATA = BENCH / "data/gears_pert_data"
SPLITS = BENCH / "working_dir/results"
MATCHED = BENCH / "working_dir/results/matched_ml_baselines"
ESM = BENCH / "working_dir/external/esm2/esm2_pert_embedding_1280d.tsv"
REVERSE_RESULTS = _ROOT / "reverse" / "results"

CELL_LINES: dict[str, dict[str, str]] = {
    "hepg2": {
        "label": "HepG2",
        "dataset": "replogle_hepg2_essential",
        "pred_dir": "replogle_hepg2_essential__prog_L3_k562_rpe1_jurkat",
        "p_tsv": "replogle_hepg2_essential__prog_3ref_k562_rpe1_jurkat_30d.tsv",
        "gears_dir": "replogle_hepg2_essential__gears_seed1",
        "txpert_gat_dir": "replogle_hepg2_essential__txpert_gat_seed1",
        "txpert_xcell_dir": "replogle_hepg2_essential__txpert_gat_xcell_loo_seed1",
        "scgpt_ft_dir": "replogle_hepg2_essential__scgpt_ft_seed1",
    },
    "k562": {
        "label": "K562",
        "dataset": "replogle_k562_essential",
        "pred_dir": "replogle_k562_essential__prog_L3_rpe1_hepg2_jurkat",
        "p_tsv": "replogle_k562_essential__prog_3ref_rpe1_hepg2_jurkat_30d.tsv",
        "gears_dir": "replogle_k562_essential__gears_seed1",
        "txpert_gat_dir": "replogle_k562_essential__txpert_gat_seed1",
        "txpert_xcell_dir": "replogle_k562_essential__txpert_gat_xcell_loo_seed1",
        "scgpt_ft_dir": "replogle_k562_essential__scgpt_ft_seed1",
    },
    "rpe1": {
        "label": "RPE1",
        "dataset": "replogle_rpe1_essential",
        "pred_dir": "replogle_rpe1_essential__prog_L3_k562_hepg2_jurkat",
        "p_tsv": "replogle_rpe1_essential__prog_3ref_k562_hepg2_jurkat_30d.tsv",
        "gears_dir": "replogle_rpe1_essential__gears_seed1",
        "txpert_gat_dir": "replogle_rpe1_essential__txpert_gat_seed1",
        "txpert_xcell_dir": "replogle_rpe1_essential__txpert_gat_xcell_loo_seed1",
        "scgpt_ft_dir": "replogle_rpe1_essential__scgpt_ft_seed1",
    },
    "jurkat": {
        "label": "Jurkat",
        "dataset": "replogle_jurkat_essential",
        "pred_dir": "replogle_jurkat_essential__prog_L3_k562_rpe1_hepg2",
        "p_tsv": "replogle_jurkat_essential__prog_3ref_k562_rpe1_hepg2_30d.tsv",
        "gears_dir": "replogle_jurkat_essential__gears_seed1",
        "txpert_gat_dir": "replogle_jurkat_essential__txpert_gat_seed1",
        "txpert_xcell_dir": "replogle_jurkat_essential__txpert_gat_xcell_loo_seed1",
        "scgpt_ft_dir": "replogle_jurkat_essential__scgpt_ft_seed1",
    },
    # Bulk / genome-wide extension (not Essential). Reverse pilots via reverse.src.gwps_reverse;
    # pred_dir/p_tsv intentionally unset until a non-leaky GWPS forward gallery exists.
    "k562_gwps": {
        "label": "K562-GWPS",
        "dataset": "replogle_k562_gwps",
        "pred_dir": "",
        "p_tsv": "",
        "gears_dir": "",
        "txpert_gat_dir": "",
        "txpert_xcell_dir": "",
        "scgpt_ft_dir": "",
    },
}


def resolve_cell_paths(cell: str, seed: int = 1) -> dict[str, Path | str | int]:
    key = cell.strip().lower()
    if key not in CELL_LINES:
        raise KeyError(f"Unknown cell line {cell!r}; choose from {sorted(CELL_LINES)}")
    if seed not in range(1, 6):
        raise ValueError(f"seed must be 1..5, got {seed}")
    info = CELL_LINES[key]
    ds = info["dataset"]

    # GWPS bulk extension: observed ΔY + custom split only (see reverse.src.gwps_reverse)
    if key == "k562_gwps":
        return {
            "cell": key,
            "label": info["label"],
            "seed": seed,
            "pred_dir": Path(""),
            "dataset_h5ad": Path(""),
            "pseudobulk_deltas": DATA / ds / "all_pseudobulk_deltas.h5ad",
            "split": SPLITS / f"seed_{seed}_{ds}_split",
            "p_tsv": Path(""),
            "esm_tsv": ESM,
            "retrieval_out": REVERSE_RESULTS / f"retrieval_{key}_v2",
            "fair_out": REVERSE_RESULTS / f"fair_compare_{key}",
            "gears_dir": Path(""),
            "txpert_gat_dir": Path(""),
            "txpert_xcell_dir": Path(""),
            "scgpt_dir": Path(""),
            "gallery_compare_out": REVERSE_RESULTS / f"gallery_compare_{key}",
        }

    def _matched(name: str) -> Path:
        return MATCHED / name

    gears = _matched(info["gears_dir"])
    if not gears.is_dir():
        for alt in (
            "replogle_jurkat_essential__gears_seed1_cap30",
            "replogle_jurkat_essential__gears_seed1_cap20",
        ):
            if _matched(alt).is_dir():
                gears = _matched(alt)
                break

    scgpt_matched = _matched(info["scgpt_ft_dir"])
    scgpt_reverse = REVERSE_RESULTS / f"scgpt_{key}_seed1"
    if (scgpt_matched / "all_predictions.json").is_file():
        scgpt_dir = scgpt_matched
    elif (scgpt_reverse / "all_predictions.json").is_file():
        scgpt_dir = scgpt_reverse
    else:
        scgpt_dir = scgpt_matched

    # seed-1 keeps legacy dir names for backward compatibility
    if seed == 1:
        retrieval_out = REVERSE_RESULTS / f"retrieval_{key}_v2"
        fair_out = REVERSE_RESULTS / f"fair_compare_{key}"
        gallery_out = REVERSE_RESULTS / f"gallery_compare_{key}"
    else:
        retrieval_out = REVERSE_RESULTS / f"retrieval_{key}_seed{seed}_v2"
        fair_out = REVERSE_RESULTS / f"fair_compare_{key}_seed{seed}"
        gallery_out = REVERSE_RESULTS / f"gallery_compare_{key}_seed{seed}"

    return {
        "cell": key,
        "label": info["label"],
        "seed": seed,
        "pred_dir": STACK / info["pred_dir"],
        "dataset_h5ad": DATA / ds / "perturb_processed.h5ad",
        "pseudobulk_deltas": DATA / ds / "all_pseudobulk_deltas.h5ad",
        "split": SPLITS / f"seed_{seed}_{ds}_split",
        "p_tsv": PCA_DIR / info["p_tsv"],
        "esm_tsv": ESM,
        "retrieval_out": retrieval_out,
        "fair_out": fair_out,
        "gears_dir": gears,
        "txpert_gat_dir": _matched(info["txpert_gat_dir"]),
        "txpert_xcell_dir": _matched(info["txpert_xcell_dir"]),
        "scgpt_dir": scgpt_dir,
        "gallery_compare_out": gallery_out,
    }


def list_cells() -> list[str]:
    """Essential Task-1 lines only (excludes GWPS extension)."""
    return [c for c in CELL_LINES if c != "k562_gwps"]


def list_all_cells() -> list[str]:
    return list(CELL_LINES.keys())
