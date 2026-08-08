#!/usr/bin/env python3
"""Cross-line reference galleries for reverse matching (forward-style reference use).

Unlike \"fill missing KOs from another line\" (distractors), this keeps the *same*
target-line catalog and improves each KO's gallery vector by blending aligned
predicted ΔY from reference cell lines — analogous to multi-line forward reference.

Methods
-------
- ``target_only``: target-line predicted ΔY (baseline)
- ``ref_mean``: mean of {target + available refs} after per-vector z-score
- ``ref_blend``: (1-α)*target + α*mean(refs)  (refs-only mean; if no ref, target)

Claim boundary: still ranks within the target Essential catalog; does not add
unscreened genes (e.g. TP53 still absent from HepG2 Essential).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402
from reverse.src.compare_fair_baselines import pearson_gallery_ranks  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_reverse_bundle  # noqa: E402


def _zvec(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=float).copy()
    m = np.isfinite(x)
    if m.sum() < 10:
        return x
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    x[m] = (x[m] - mu) / sd
    return x


def load_line_delta_gallery(cell: str) -> tuple[list[str], dict[str, np.ndarray]]:
    paths = resolve_cell_paths(cell)
    genes, pred_abs = load_prediction_dir(Path(paths["pred_dir"]))
    ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
    return genes, absolute_to_delta(pred_abs, ctrl)


def align_gallery(
    genes_src: list[str],
    gal_src: dict[str, np.ndarray],
    genes_dst: list[str],
) -> dict[str, np.ndarray]:
    g2i = {g: i for i, g in enumerate(genes_src)}
    idx = np.array([g2i.get(g, -1) for g in genes_dst])
    out: dict[str, np.ndarray] = {}
    for ko, vec in gal_src.items():
        v = np.full(len(genes_dst), np.nan, dtype=float)
        ok = idx >= 0
        v[ok] = np.asarray(vec, dtype=float)[idx[ok]]
        if np.isfinite(v).sum() < 500:
            continue
        out[ko] = v
    return out


def build_reference_gallery(
    target: str,
    ref_cells: list[str],
    *,
    mode: str = "ref_blend",
    alpha: float = 0.5,
) -> tuple[list[str], dict[str, np.ndarray], dict]:
    """Build catalog on target gene axis / target KO set, optionally blended with refs."""
    genes_t, gal_t = load_line_delta_gallery(target)
    ref_aligned: dict[str, dict[str, np.ndarray]] = {}
    for c in ref_cells:
        if c == target:
            continue
        gs, gal = load_line_delta_gallery(c)
        ref_aligned[c] = align_gallery(gs, gal, genes_t)

    out_gal: dict[str, np.ndarray] = {}
    n_with_ref = 0
    n_ref_hits = []
    for ko, v_t in gal_t.items():
        refs = []
        for c, gal in ref_aligned.items():
            if ko in gal:
                refs.append(_zvec(gal[ko]))
        if mode == "target_only" or not refs:
            out_gal[ko] = np.asarray(v_t, dtype=float)
            n_ref_hits.append(0)
            continue
        n_with_ref += 1
        n_ref_hits.append(len(refs))
        z_t = _zvec(v_t)
        z_r = np.nanmean(np.vstack(refs), axis=0)
        if mode == "ref_mean":
            # mean of target + all refs (z-scored)
            stack = np.vstack([z_t, *refs])
            out_gal[ko] = np.nanmean(stack, axis=0)
        elif mode == "ref_blend":
            out_gal[ko] = (1.0 - alpha) * z_t + alpha * z_r
        else:
            raise ValueError(f"Unknown mode {mode}")

    meta = {
        "target": target,
        "ref_cells": list(ref_aligned.keys()),
        "mode": mode,
        "alpha": alpha if mode == "ref_blend" else None,
        "n_catalog": len(out_gal),
        "n_kos_with_ge1_ref": n_with_ref,
        "mean_n_refs_per_ko": float(np.mean(n_ref_hits)) if n_ref_hits else 0.0,
        "n_genes": len(genes_t),
    }
    return genes_t, out_gal, meta


def run_task1_compare(
    *,
    target: str = "hepg2",
    ref_cells: list[str] | None = None,
    alpha: float = 0.5,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """Pearson reverse on held-out target test: target_only vs ref_blend vs ref_mean."""
    if ref_cells is None:
        ref_cells = [c for c in list_cells() if c != target]

    paths = resolve_cell_paths(target)
    genes, tables, _meta = load_reverse_bundle(
        Path(paths["pseudobulk_deltas"]),
        Path(paths["pred_dir"]) / "gene_names.json",
        Path(paths["split"]),
        Path(paths["p_tsv"]),
    )
    cat_kos: list[str] = []
    for part in ("train", "val", "test"):
        cat_kos.extend(tables[part]["kos"])

    kte = list(tables["test"]["kos"])
    Yte = tables["test"]["Y"]

    modes = {
        "pearson_target_only": ("target_only", 0.0),
        "pearson_ref_blend": ("ref_blend", alpha),
        "pearson_ref_mean": ("ref_mean", alpha),
    }
    frames = []
    metas = {}
    for method, (mode, a) in modes.items():
        g_genes, gal, meta = build_reference_gallery(target, ref_cells, mode=mode, alpha=a)
        assert g_genes == genes
        g_kos = [k for k in cat_kos if k in gal]
        keep = [i for i, k in enumerate(kte) if k in gal]
        kte_m = [kte[i] for i in keep]
        Yte_m = Yte[keep]
        frames.append(pearson_gallery_ranks(Yte_m, kte_m, gal, g_kos, method))
        metas[method] = meta

    all_df = pd.concat(frames, ignore_index=True)
    out = Path(out_dir or (Path(paths["fair_out"]).parent / f"fair_compare_{target}_ref_method"))
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "task1_ranks.tsv", sep="\t", index=False)

    rows = []
    for m, sub in all_df.groupby("method"):
        s = summarize_recovery(sub)
        s["method"] = m
        s["mrr"] = float((1.0 / sub["rank"]).mean())
        s.update({f"meta_{k}": v for k, v in metas[m].items() if k != "ref_cells"})
        s["ref_cells"] = ",".join(metas[m]["ref_cells"])
        rows.append(s)
    sum_df = pd.DataFrame(rows).sort_values("median_rank")
    sum_df.to_csv(out / "task1_summary.tsv", sep="\t", index=False)

    # paired target_only vs ref_blend
    a = all_df[all_df.method == "pearson_target_only"].set_index("true_ko")
    b = all_df[all_df.method == "pearson_ref_blend"].set_index("true_ko")
    c = all_df[all_df.method == "pearson_ref_mean"].set_index("true_ko")
    common = sorted(set(a.index) & set(b.index) & set(c.index))
    paired = pd.DataFrame(
        {
            "true_ko": common,
            "rank_target_only": [int(a.loc[k, "rank"]) for k in common],
            "rank_ref_blend": [int(b.loc[k, "rank"]) for k in common],
            "rank_ref_mean": [int(c.loc[k, "rank"]) for k in common],
        }
    )
    paired["delta_blend"] = paired["rank_ref_blend"] - paired["rank_target_only"]
    paired["delta_mean"] = paired["rank_ref_mean"] - paired["rank_target_only"]
    paired.to_csv(out / "task1_paired.tsv", sep="\t", index=False)

    summary = {
        "target": target,
        "ref_cells": ref_cells,
        "alpha": alpha,
        "n_test": len(common),
        "by_method": sum_df[["method", "median_rank", "mean_rank", "pct_top10", "pct_top50", "mrr"]].to_dict(
            orient="records"
        ),
        "pct_improved_blend": float((paired["delta_blend"] < 0).mean() * 100),
        "pct_worsened_blend": float((paired["delta_blend"] > 0).mean() * 100),
        "pct_same_blend": float((paired["delta_blend"] == 0).mean() * 100),
        "pct_improved_mean": float((paired["delta_mean"] < 0).mean() * 100),
        "pct_worsened_mean": float((paired["delta_mean"] > 0).mean() * 100),
        "claim": (
            "Reference improves gallery vectors for the same target catalog; "
            "not cross-line KO fill; catalog membership unchanged."
        ),
    }
    (out / "compare.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "gallery_meta.json").write_text(json.dumps(metas, indent=2) + "\n")
    print(sum_df[["method", "median_rank", "mean_rank", "pct_top10", "pct_top50", "mrr"]].to_string(index=False))
    print(json.dumps({k: summary[k] for k in summary if k != "by_method"}, indent=2))
    print(json.dumps(summary["by_method"], indent=2))
    print(f"Wrote {out}")
    return sum_df


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-line reference gallery for reverse Task-1")
    p.add_argument("--target", default="hepg2")
    p.add_argument("--ref_cells", default="", help="Comma list; default=other Essential lines")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()
    refs = [c.strip() for c in args.ref_cells.split(",") if c.strip()] or None
    run_task1_compare(
        target=args.target,
        ref_cells=refs,
        alpha=args.alpha,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )


if __name__ == "__main__":
    main()
