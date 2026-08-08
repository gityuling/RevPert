#!/usr/bin/env python3
"""
Gallery-protocol baselines on PDGrapher genetic 10 lines (same Task-1 metrics).

Methods (no Essential checkpoints — those gene sets do not transfer):
  - pearson_train_gallery (recomputed for consistency)
  - gem_lite_topDEG_corr
  - cmap_lite_pred_gallery
  - ridge_delta_to_gallery  (Ridge ΔY → multi-hot over train-gallery KOs)

Merges into reverse/results/pdgrapher_genetic_task1_unified/ with dual + PDGrapher.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.run_pdgrapher_genetic_compare import (  # noqa: E402
    build_train_gallery,
    load_or_cache_cell,
    pearson_score_matrix,
)

CELLS = [
    "A549",
    "A375",
    "AGS",
    "BICR6",
    "ES2",
    "HT29",
    "MCF7",
    "PC3",
    "U251MG",
    "YAPC",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _idcg(num_correct: int, num_nodes: int) -> float:
    idcg = 0.0
    for rank in range(1, num_correct + 1):
        idcg += (1.0 - rank / num_nodes) / np.log2(rank + 1)
    return idcg


def gem_lite_scores(queries: np.ndarray, gallery_mat: np.ndarray, top_deg: int = 200) -> np.ndarray:
    """queries n_q×G, gallery genes×n_gal → n_q×n_gal."""
    g_mat = gallery_mat.T  # n_gal × genes
    n_q = queries.shape[0]
    n_gal = g_mat.shape[0]
    scores = np.zeros((n_q, n_gal), dtype=np.float64)
    for i in range(n_q):
        q = np.nan_to_num(queries[i], nan=0.0)
        idx = np.argsort(-np.abs(q))[:top_deg]
        qq = q[idx]
        qq = (qq - qq.mean()) / (qq.std() + 1e-12)
        sub = g_mat[:, idx]
        sub = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-12)
        scores[i] = sub @ qq / max(len(idx), 1)
    return scores


def cmap_lite_scores(queries: np.ndarray, gallery_mat: np.ndarray, top_n: int = 100) -> np.ndarray:
    """queries n_q×G, gallery genes×n_gal → n_q×n_gal."""
    g_mat = np.nan_to_num(gallery_mat, nan=0.0)  # genes × n_gal
    n_genes, n_gal = g_mat.shape
    order = np.argsort(-g_mat, axis=0)
    ranks = np.empty_like(order)
    for j in range(n_gal):
        ranks[order[:, j], j] = np.arange(1, n_genes + 1)
    n_q = queries.shape[0]
    scores = np.zeros((n_q, n_gal), dtype=np.float64)
    for i in range(n_q):
        q = np.nan_to_num(queries[i], nan=0.0)
        up = np.argsort(-q)[:top_n]
        down = np.argsort(q)[:top_n]
        scores[i] = (-ranks[up, :].mean(axis=0)) + ranks[down, :].mean(axis=0)
    return scores


def ridge_gallery_scores(
    deltas: np.ndarray,
    interv: np.ndarray,
    gene_symbols: list[str],
    train_idx: list[int],
    test_idx: list[int],
    gallery_kos: list[str],
    pca_dim: int = 256,
) -> np.ndarray:
    from sklearn.decomposition import PCA

    name_to_gal = {k: j for j, k in enumerate(gallery_kos)}
    n_gal = len(gallery_kos)
    Ytr = np.nan_to_num(deltas[np.asarray(train_idx)], nan=0.0)
    T = np.zeros((len(train_idx), n_gal), dtype=np.float64)
    for i, sample_i in enumerate(train_idx):
        for j in interv[sample_i]:
            name = gene_symbols[int(j)]
            if name in name_to_gal:
                T[i, name_to_gal[name]] = 1.0
    if T.sum() == 0:
        return np.zeros((len(test_idx), n_gal), dtype=np.float64)
    k = min(pca_dim, Ytr.shape[0] - 1, Ytr.shape[1])
    pca = PCA(n_components=k, random_state=0)
    Ytr_p = pca.fit_transform(Ytr)
    reg = Ridge(alpha=1.0, random_state=0)
    reg.fit(Ytr_p, T)
    Yte = np.nan_to_num(deltas[np.asarray(test_idx)], nan=0.0)
    return reg.predict(pca.transform(Yte))


def summarize_from_scores(
    method: str,
    scores: np.ndarray,
    deltas_test_idx: list[int],
    interv: np.ndarray,
    gene_symbols: list[str],
    gallery_kos: list[str],
) -> dict:
    n_nodes = len(gene_symbols)
    n_gal = len(gallery_kos)
    name_to_gal = {k: j for j, k in enumerate(gallery_kos)}
    order = np.argsort(-scores, axis=1, kind="mergesort")
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(n_gal)

    r1, r10, r100, r1000 = [], [], [], []
    ranking_scores, ndcgs, med_ranks = [], [], []
    n_partial = 0
    for t, sample_i in enumerate(deltas_test_idx):
        correct_names = [gene_symbols[int(j)] for j in interv[sample_i]]
        ranks0 = [
            int(inv[t, name_to_gal[cn]]) if cn in name_to_gal else n_gal for cn in correct_names
        ]
        n_c = max(len(ranks0), 1)
        med_ranks.append(min(ranks0) + 1)

        def hit(k: int) -> float:
            return sum(1 for r in ranks0 if r < k) / n_c

        r1.append(hit(1))
        r10.append(hit(10))
        r100.append(hit(100))
        r1000.append(hit(1000))
        dcg = 0.0
        for r0 in ranks0:
            ranking_scores.append(1.0 - (r0 / n_nodes))
            rank1 = r0 + 1
            dcg += (1.0 - rank1 / n_nodes) / np.log2(rank1 + 1)
        idcg = _idcg(len(ranks0), n_nodes)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        if any(r < len(ranks0) for r in ranks0):
            n_partial += 1

    return {
        "method": method,
        "n_test": len(deltas_test_idx),
        "n_gallery_kos": n_gal,
        "recall@1": float(np.mean(r1)),
        "recall@10": float(np.mean(r10)),
        "recall@100": float(np.mean(r100)),
        "recall@1000": float(np.mean(r1000)),
        "pct_partially_accurate": 100.0 * n_partial / max(len(deltas_test_idx), 1),
        "ranking_score": float(np.mean(ranking_scores)) if ranking_scores else float("nan"),
        "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "median_rank_true": float(np.median(med_ranks)),
    }


def run_cell(cell: str, data_dir: Path, splits_dir: Path) -> list[dict]:
    blob = load_or_cache_cell(data_dir, cell)
    gene_symbols = blob["gene_symbols"]
    deltas = blob["deltas"]
    interv = blob["interv"]
    splits = torch.load(
        splits_dir / "genetic" / cell / "random" / "5fold" / "splits.pt",
        map_location="cpu",
        weights_only=False,
    )
    rows = []
    for fold_idx in sorted(splits.keys()):
        sp = splits[fold_idx]
        train_idx = list(sp["train_index_backward"])
        test_idx = list(sp["test_index_backward"])
        kos, mat = build_train_gallery(deltas, interv, gene_symbols, train_idx)
        queries = deltas[np.asarray(test_idx, dtype=np.int64)]
        score_map = {
            "pearson_train_gallery": pearson_score_matrix(queries, mat),
            "gem_lite_topDEG_corr": gem_lite_scores(queries, mat),
            "cmap_lite_pred_gallery": cmap_lite_scores(queries, mat),
            "ridge_delta_to_gallery": ridge_gallery_scores(
                deltas, interv, gene_symbols, train_idx, test_idx, kos
            ),
        }
        for method, scores in score_map.items():
            m = summarize_from_scores(method, scores, test_idx, interv, gene_symbols, kos)
            m.update({"cell": cell, "fold": int(fold_idx)})
            rows.append(m)
            _log(
                f"[{cell}] fold {fold_idx} {method}: "
                f"r@10={m['recall@10']:.4f} median_rank={m['median_rank_true']:.1f}"
            )
    return rows


def rebuild_unified(gallery_df: pd.DataFrame, out_dir: Path, dual_path: Path, official_dir: Path) -> None:
    dual = pd.read_csv(dual_path, sep="\t")
    dual = dual[dual["method"].isin(["dual_encoder"])].copy()
    off_frames = []
    for cell in CELLS:
        fp = official_dir / cell / "n_gnn_2" / "fold_metrics.tsv"
        d = pd.read_csv(fp, sep="\t")
        d["method"] = "pdgrapher_official"
        off_frames.append(d)
    off = pd.concat(off_frames, ignore_index=True)

    keep = [
        "method",
        "cell",
        "fold",
        "n_test",
        "recall@1",
        "recall@10",
        "recall@100",
        "recall@1000",
        "pct_partially_accurate",
        "ranking_score",
        "ndcg",
        "median_rank_true",
    ]
    # Prefer gallery_df pearson over dual's pearson (same protocol); drop dual pearson
    all_folds = pd.concat(
        [gallery_df[keep], dual[keep], off[keep]],
        ignore_index=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(out_dir / "all_fold_metrics.tsv", sep="\t", index=False)

    method_order = [
        "pearson_train_gallery",
        "cmap_lite_pred_gallery",
        "gem_lite_topDEG_corr",
        "ridge_delta_to_gallery",
        "pdgrapher_official",
        "dual_encoder",
    ]
    g = all_folds.groupby(["cell", "method"], as_index=False).agg(
        median_rank_mean=("median_rank_true", "mean"),
        median_rank_std=("median_rank_true", "std"),
        recall_at_10_mean=("recall@10", "mean"),
        recall_at_10_std=("recall@10", "std"),
        partial_pct_mean=("pct_partially_accurate", "mean"),
        ndcg_mean=("ndcg", "mean"),
        n_folds=("fold", "count"),
    )
    g["method"] = pd.Categorical(g["method"], method_order, ordered=True)
    g["cell"] = pd.Categorical(g["cell"], CELLS, ordered=True)
    g = g.sort_values(["cell", "method"])
    g.to_csv(out_dir / "main_table_median_rank_r10.tsv", sep="\t", index=False)

    # wide: median rank + r@10 for key methods
    wide_rank = g.pivot(index="cell", columns="method", values="median_rank_mean")
    wide_r10 = g.pivot(index="cell", columns="method", values="recall_at_10_mean")
    cols = {}
    for m, short in [
        ("pearson_train_gallery", "pearson"),
        ("cmap_lite_pred_gallery", "cmap_lite"),
        ("gem_lite_topDEG_corr", "gem_lite"),
        ("ridge_delta_to_gallery", "ridge"),
        ("pdgrapher_official", "pdgrapher"),
        ("dual_encoder", "dual"),
    ]:
        if m in wide_rank.columns:
            cols[f"{short}_median_rank"] = wide_rank[m]
            cols[f"{short}_r@10"] = 100.0 * wide_r10[m]
    wide = pd.DataFrame(cols)
    wide.to_csv(out_dir / "main_table_wide.tsv", sep="\t")

    si = all_folds.groupby(["cell", "method"], as_index=False).agg(
        partial_pct_mean=("pct_partially_accurate", "mean"),
        partial_pct_std=("pct_partially_accurate", "std"),
        recall_at_1_mean=("recall@1", "mean"),
        ndcg_mean=("ndcg", "mean"),
        ndcg_std=("ndcg", "std"),
    )
    si["method"] = pd.Categorical(si["method"], method_order, ordered=True)
    si["cell"] = pd.Categorical(si["cell"], CELLS, ordered=True)
    si = si.sort_values(["cell", "method"])
    si.to_csv(out_dir / "si_table_partial_ndcg.tsv", sep="\t", index=False)

    (out_dir / "README.md").write_text(
        "# PDGrapher genetic — unified Task-1 metrics\n\n"
        "Same reverse-retrieval protocol; PDGrapher genetic as extra data resource.\n\n"
        "**Main:** median rank + Recall@10 (`main_table_wide.tsv`)\n\n"
        "Methods: pearson / cmap_lite / gem_lite / ridge(gallery) / PDGrapher official / dual.\n\n"
        "Note: Essential GEARS/scGPT/TxPert checkpoints do **not** transfer "
        "(gene-set mismatch). Gallery baselines here need no those ckpts.\n\n"
        "**SI:** partial% / nDCG / R@1 (`si_table_partial_ndcg.tsv`)\n",
        encoding="utf-8",
    )
    _log(f"Wrote {out_dir}")
    _log("\n" + wide.to_string(float_format=lambda x: f"{x:.1f}"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=Path,
        default=_ROOT
        / "reverse/external/PDGrapher/data/processed/torch_data/real_lognorm",
    )
    p.add_argument(
        "--splits_dir",
        type=Path,
        default=_ROOT / "reverse/external/PDGrapher/data/processed/splits",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_genetic_gallery_baselines",
    )
    p.add_argument(
        "--unified_dir",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_genetic_task1_unified",
    )
    p.add_argument(
        "--dual_metrics",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_genetic_dual/all_fold_metrics.tsv",
    )
    p.add_argument(
        "--official_dir",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_official_genetic",
    )
    p.add_argument("--cells", nargs="+", default=CELLS)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for cell in args.cells:
        all_rows.extend(run_cell(cell, args.data_dir, args.splits_dir))
    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_dir / "all_fold_metrics.tsv", sep="\t", index=False)
    rebuild_unified(df, args.unified_dir, args.dual_metrics, args.official_dir)


if __name__ == "__main__":
    main()
