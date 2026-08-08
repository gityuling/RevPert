#!/usr/bin/env python3
"""RevPert Block-A cross-cell identity transfer (Essential seed-1).

Protocol
--------
Source gene / PCA axis; target observed test ΔY + target L3 gallery reindexed
to source genes (missing → 0).

Scorers (Table 1 Block A + learn-only ablation):
  - pearson, cmap_lite, gem_lite  (expression-gallery matchers on aligned axis)
  - ridge_delta_to_P              (fit Y→P on *source* train; rank in *source* P atlas;
                                  queries restricted to KOs present in source P catalog)
  - gallery_dual, fuse_alpha_train, fuse_alpha_retune  (source RevPert ckpt)

Outputs: reverse/results/revpert/cross_cell_seed1/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.scripts.run_gallery_dual_g2 import (  # noqa: E402
    GalleryDual,
    eval_identity,
    pearson_matrix,
    score_learned_np,
)
from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_reverse_bundle  # noqa: E402
from reverse.src.reverse_model import PCAProjector  # noqa: E402

CKPT_ROOT = _ROOT / "reverse/results/revpert/essential"
OUT_DEFAULT = _ROOT / "reverse/results/revpert/cross_cell_seed1"


def _log(msg: str) -> None:
    print(msg, flush=True)


def ckpt_path(cell: str, seed: int = 1) -> Path:
    return CKPT_ROOT / cell.lower() / f"seed{seed}" / "gallery_dual_best.pt"


def _zrows(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def ranks_from_scores(
    scores: np.ndarray, query_kos: list[str], gallery_kos: list[str], method: str
) -> pd.DataFrame:
    g_index = {k: i for i, k in enumerate(gallery_kos)}
    rows = []
    for i, ko in enumerate(query_kos):
        order = np.argsort(-scores[i])
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        j = g_index[ko]
        rows.append(
            {
                "true_ko": ko,
                "rank": int(ranks[j]),
                "score": float(scores[i, j]),
                "best_ko": gallery_kos[int(order[0])],
                "method": method,
            }
        )
    return pd.DataFrame(rows)


def cmap_lite_ranks(
    query_Y: np.ndarray,
    query_kos: list[str],
    gallery: dict[str, np.ndarray],
    gallery_kos: list[str],
    top_n: int = 100,
) -> pd.DataFrame:
    g_mat = np.column_stack([gallery[k] for k in gallery_kos])
    n_genes = g_mat.shape[0]
    order = np.argsort(-np.nan_to_num(g_mat, nan=0.0), axis=0)
    ranks = np.empty_like(order)
    for j in range(g_mat.shape[1]):
        ranks[order[:, j], j] = np.arange(1, n_genes + 1)
    out_rows = []
    g_index = {k: i for i, k in enumerate(gallery_kos)}
    for i, ko in enumerate(query_kos):
        q = np.nan_to_num(query_Y[i], nan=0.0)
        up = np.argsort(-q)[:top_n]
        down = np.argsort(q)[:top_n]
        scores = -ranks[up, :].mean(axis=0) + ranks[down, :].mean(axis=0)
        order_s = np.argsort(-scores)
        rnk = np.empty_like(order_s)
        rnk[order_s] = np.arange(1, len(order_s) + 1)
        j = g_index[ko]
        out_rows.append(
            {
                "true_ko": ko,
                "rank": int(rnk[j]),
                "score": float(scores[j]),
                "best_ko": gallery_kos[int(order_s[0])],
                "method": "cmap_lite",
            }
        )
    return pd.DataFrame(out_rows)


def gem_lite_ranks(
    query_Y: np.ndarray,
    query_kos: list[str],
    gallery: dict[str, np.ndarray],
    gallery_kos: list[str],
    top_deg: int = 200,
) -> pd.DataFrame:
    g_mat = np.column_stack([gallery[k] for k in gallery_kos]).T
    scores = np.zeros((len(query_kos), len(gallery_kos)))
    for i in range(len(query_kos)):
        q = np.nan_to_num(query_Y[i], nan=0.0)
        idx = np.argsort(-np.abs(q))[:top_deg]
        qq = q[idx]
        qq = (qq - qq.mean()) / (qq.std() + 1e-12)
        sub = g_mat[:, idx]
        sub = (sub - sub.mean(axis=1, keepdims=True)) / (
            sub.std(axis=1, keepdims=True) + 1e-12
        )
        scores[i] = sub @ qq / len(idx)
    return ranks_from_scores(scores, query_kos, gallery_kos, "gem_lite")


def load_bundle_and_gallery(cell: str, seed: int = 1):
    paths = resolve_cell_paths(cell, seed=seed)
    genes, tables, meta = load_reverse_bundle(
        Path(paths["pseudobulk_deltas"]),
        Path(paths["pred_dir"]) / "gene_names.json",
        Path(paths["split"]),
        Path(paths["p_tsv"]),
    )
    genes2, pred_abs = load_prediction_dir(Path(paths["pred_dir"]))
    assert genes2 == genes
    ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
    pred_gal = absolute_to_delta(pred_abs, ctrl)
    all_kos: list[str] = []
    cat_P = []
    for part in ("train", "val", "test"):
        all_kos.extend(tables[part]["kos"])
        cat_P.append(tables[part]["P"])
    cat_P = np.vstack(cat_P)
    gal_kos = [k for k in dict.fromkeys(all_kos) if k in pred_gal]
    gal_mat = np.stack([np.nan_to_num(pred_gal[k], nan=0.0) for k in gal_kos], axis=0).astype(
        np.float32
    )
    pred_gal_aligned = {k: gal_mat[i] for i, k in enumerate(gal_kos)}
    return {
        "cell": cell,
        "label": paths["label"],
        "genes": genes,
        "tables": tables,
        "gal_kos": gal_kos,
        "gal_mat": gal_mat,
        "pred_gal": pred_gal_aligned,
        "cat_kos": list(all_kos),
        "cat_P": cat_P.astype(np.float32),
        "meta": meta,
    }


def align_rows(mat: np.ndarray, src_genes: list[str], dst_genes: list[str]) -> np.ndarray:
    idx = {g: i for i, g in enumerate(dst_genes)}
    out = np.zeros((mat.shape[0], len(src_genes)), dtype=np.float32)
    for j, g in enumerate(src_genes):
        i = idx.get(g)
        if i is not None:
            out[:, j] = mat[:, i]
    return out


def load_model(ckpt_file: Path, device: torch.device) -> tuple[GalleryDual, float, float, dict]:
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    pca = PCAProjector()
    pca.load_state_dict(ckpt["pca"])
    args = ckpt.get("args", {})
    model = GalleryDual(
        pca,
        emb_dim=int(args.get("emb_dim", 128)),
        hidden=int(args.get("hidden", 512)),
        dropout=float(args.get("dropout", 0.1)),
        shared=bool(args.get("shared", 1)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    a_train = float(ckpt.get("alpha_star", 0.5))
    a_retune = float(ckpt.get("alpha_retune", a_train))
    meta = {
        "alpha_star": a_train,
        "alpha_retune": a_retune,
        "n_genes_pca": int(pca.mean_.shape[0]),
        "pca_dim": int(pca.n_components),
    }
    return model, a_train, a_retune, meta


def _summary_row(
    rank_df: pd.DataFrame,
    method: str,
    src_pack: dict,
    tgt_pack: dict,
    source: str,
    target: str,
    overlap: float,
    n_gallery: int,
    extra: dict | None = None,
) -> dict:
    s = summarize_recovery(rank_df)
    s["mrr"] = float((1.0 / rank_df["rank"]).mean())
    out = {
        "source": src_pack["label"],
        "target": tgt_pack["label"],
        "source_key": source,
        "target_key": target,
        "same_cell": source == target,
        "method": method,
        "alpha": np.nan,
        "gene_overlap_frac": overlap,
        "n_src_genes": len(src_pack["genes"]),
        "n_tgt_genes": len(tgt_pack["genes"]),
        "n_gallery": n_gallery,
        "n": int(s["n_query"]),
        "median_rank": float(s["median_rank"]),
        "mean_rank": float(s["mean_rank"]),
        "mrr": float(s["mrr"]),
        "recall@1": float(s["pct_top1"]) / 100.0,
        "recall@10": float(s["pct_top10"]) / 100.0,
        "recall@100": float(s["pct_top100"]) / 100.0,
    }
    if extra:
        out.update(extra)
    return out


def ridge_transfer_ranks(
    src_pack: dict,
    Yte_a: np.ndarray,
    Kte: list[str],
) -> pd.DataFrame:
    """Fit Y→P on source train; score target queries in source P atlas."""
    Ytr = np.nan_to_num(src_pack["tables"]["train"]["Y"], nan=0.0)
    Ptr = np.nan_to_num(src_pack["tables"]["train"]["P"], nan=0.0)
    cat_kos = src_pack["cat_kos"]
    cat_P = np.nan_to_num(src_pack["cat_P"], nan=0.0)
    src_set = set(cat_kos)
    keep = [i for i, k in enumerate(Kte) if k in src_set]
    if not keep:
        return pd.DataFrame(columns=["true_ko", "rank", "score", "best_ko", "method"])
    kte = [Kte[i] for i in keep]
    Yte = Yte_a[keep]
    reg = Ridge(alpha=1.0, random_state=0)
    reg.fit(Ytr, Ptr)
    Phat = _zrows(reg.predict(np.nan_to_num(Yte, nan=0.0)))
    Pcat = _zrows(cat_P)
    scores = Phat @ Pcat.T / Phat.shape[1]
    return ranks_from_scores(scores, kte, cat_kos, "ridge_delta_to_P")


def run_pair(
    source: str,
    target: str,
    seed: int,
    device: torch.device,
    cache: dict,
) -> list[dict]:
    src_pack = cache[source]
    tgt_pack = cache[target]
    model, a_train, a_retune, mmeta = load_model(ckpt_path(source, seed), device)

    src_genes = src_pack["genes"]
    if mmeta["n_genes_pca"] != len(src_genes):
        raise RuntimeError(
            f"{source}: PCA n_genes={mmeta['n_genes_pca']} != gene_names={len(src_genes)}"
        )

    Yte = np.nan_to_num(tgt_pack["tables"]["test"]["Y"], nan=0.0).astype(np.float32)
    Kte = tgt_pack["tables"]["test"]["kos"]
    Yte_a = align_rows(Yte, src_genes, tgt_pack["genes"])
    gal_a = align_rows(tgt_pack["gal_mat"], src_genes, tgt_pack["genes"])
    gal_kos = tgt_pack["gal_kos"]
    # gallery dict on source axis for cmap/gem
    gal_dict = {k: gal_a[i] for i, k in enumerate(gal_kos)}

    overlap = float(np.mean([g in set(tgt_pack["genes"]) for g in src_genes]))
    rows: list[dict] = []

    # --- expression-gallery scorers (same query/catalog as RevPert) ---
    pear = pearson_matrix(Yte_a, gal_a)
    learn = score_learned_np(model, Yte_a, gal_a, device)
    for name, a in [
        ("pearson", 0.0),
        ("gallery_dual", 1e9),
        ("fuse_alpha_train", a_train),
        ("fuse_alpha_retune", a_retune),
    ]:
        _ranks, sm = eval_identity(Yte_a, Kte, gal_kos, gal_a, pear, learn, a)
        rows.append(
            {
                "source": src_pack["label"],
                "target": tgt_pack["label"],
                "source_key": source,
                "target_key": target,
                "same_cell": source == target,
                "method": name,
                "alpha": a,
                "gene_overlap_frac": overlap,
                "n_src_genes": len(src_genes),
                "n_tgt_genes": len(tgt_pack["genes"]),
                "n_gallery": len(gal_kos),
                **sm,
            }
        )

    cmap_df = cmap_lite_ranks(Yte_a, Kte, gal_dict, gal_kos, top_n=100)
    rows.append(
        _summary_row(
            cmap_df, "cmap_lite", src_pack, tgt_pack, source, target, overlap, len(gal_kos)
        )
    )
    gem_df = gem_lite_ranks(Yte_a, Kte, gal_dict, gal_kos, top_deg=200)
    rows.append(
        _summary_row(
            gem_df, "gem_lite", src_pack, tgt_pack, source, target, overlap, len(gal_kos)
        )
    )

    # --- ridge: source Y→P, source P atlas ---
    ridge_df = ridge_transfer_ranks(src_pack, Yte_a, Kte)
    if len(ridge_df):
        rows.append(
            _summary_row(
                ridge_df,
                "ridge_delta_to_P",
                src_pack,
                tgt_pack,
                source,
                target,
                overlap,
                len(src_pack["cat_kos"]),
                extra={"n_gallery": len(src_pack["cat_kos"])},
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--cells", nargs="+", default=None)
    args = ap.parse_args()

    cells = args.cells or list_cells()
    device = torch.device(
        args.device
        if (not str(args.device).startswith("cuda") or torch.cuda.is_available())
        else "cpu"
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    _log(f"Device={device} seed={args.seed} cells={cells}")
    cache = {}
    for c in cells:
        if not ckpt_path(c, args.seed).is_file():
            raise FileNotFoundError(ckpt_path(c, args.seed))
        _log(f"Loading data {c} ...")
        cache[c] = load_bundle_and_gallery(c, args.seed)

    all_rows: list[dict] = []
    for src in cells:
        for tgt in cells:
            _log(f"=== {src} → {tgt} ===")
            all_rows.extend(run_pair(src, tgt, args.seed, device, cache))

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "all_pairs.tsv", sep="\t", index=False)

    # matrices per method
    for method in [
        "pearson",
        "cmap_lite",
        "gem_lite",
        "ridge_delta_to_P",
        "gallery_dual",
        "fuse_alpha_train",
    ]:
        sub = df[df.method == method]
        if sub.empty:
            continue
        sub.pivot(index="source", columns="target", values="median_rank").to_csv(
            out / f"{method}_median_rank_matrix.tsv", sep="\t"
        )
        sub.pivot(index="source", columns="target", values="recall@10").to_csv(
            out / f"{method}_recall10_matrix.tsv", sep="\t"
        )

    fuse = df[df.method == "fuse_alpha_train"]
    summary = {
        "seed": args.seed,
        "protocol": (
            "source gene/PCA axis; target L3 gallery reindexed; "
            "ridge uses source Y→P and source P atlas"
        ),
        "methods": sorted(df.method.unique().tolist()),
        "cells": cells,
        "same_cell_fuse_median_rank_mean": float(fuse[fuse.same_cell]["median_rank"].mean()),
        "cross_cell_fuse_median_rank_mean": float(fuse[~fuse.same_cell]["median_rank"].mean()),
        "same_cell_by_method_median_rank_mean": {
            m: float(g[g.same_cell]["median_rank"].mean())
            for m, g in df.groupby("method")
            if g.same_cell.any()
        },
        "cross_cell_by_method_median_rank_mean": {
            m: float(g[~g.same_cell]["median_rank"].mean())
            for m, g in df.groupby("method")
            if (~g.same_cell).any()
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(json.dumps(summary["cross_cell_by_method_median_rank_mean"], indent=2))
    _log(f"Wrote {out}")


if __name__ == "__main__":
    main()
