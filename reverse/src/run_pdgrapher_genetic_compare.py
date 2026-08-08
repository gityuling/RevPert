#!/usr/bin/env python3
"""
Run our reverse-retrieval protocol on PDGrapher genetic data + official splits.

Fast path:
  - torch.load once → cache deltas/interventions as .npz
  - vectorized Pearson (queries × gallery)
  - flushed logging
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _idcg(num_correct: int, num_nodes: int) -> float:
    idcg = 0.0
    for rank in range(1, num_correct + 1):
        gain = 1.0 - (rank / num_nodes)
        discount = 1.0 / np.log2(rank + 1)
        idcg += gain * discount
    return idcg


def _cache_path(data_dir: Path, cell: str) -> Path:
    return data_dir / f"_cache_backward_{cell}.npz"


def load_or_cache_cell(data_dir: Path, cell: str) -> dict:
    """Load PyG list once; cache float32 deltas + intervention indices."""
    cache = _cache_path(data_dir, cell)
    if cache.exists():
        _log(f"[{cell}] loading cache {cache}")
        z = np.load(cache, allow_pickle=True)
        return {
            "gene_symbols": z["gene_symbols"].tolist(),
            "deltas": z["deltas"],  # n_samples × n_genes
            "interv": z["interv"],  # object array of int arrays
        }

    path = data_dir / f"data_backward_{cell}.pt"
    _log(f"[{cell}] torch.load {path} (one-time; will cache) ...")
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    gene_symbols = list(dataset[0].gene_symbols)
    n = len(dataset)
    g = len(gene_symbols)
    deltas = np.empty((n, g), dtype=np.float32)
    interv = np.empty(n, dtype=object)
    for i, d in enumerate(dataset):
        deltas[i] = (d.treated - d.diseased).detach().cpu().numpy().astype(np.float32)
        interv[i] = torch.where(d.intervention)[0].cpu().numpy().astype(np.int32)
        if (i + 1) % 2000 == 0:
            _log(f"[{cell}] extracted {i+1}/{n}")
    del dataset
    np.savez_compressed(
        cache,
        gene_symbols=np.array(gene_symbols, dtype=object),
        deltas=deltas,
        interv=interv,
    )
    _log(f"[{cell}] wrote cache {cache}")
    return {"gene_symbols": gene_symbols, "deltas": deltas, "interv": interv}


def build_train_gallery(
    deltas: np.ndarray,
    interv: np.ndarray,
    gene_symbols: list[str],
    train_idx: list[int],
) -> tuple[list[str], np.ndarray]:
    """Mean ΔY per perturbed gene on train indices. Returns (kos, genes×n_gal)."""
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in train_idx:
        delta = deltas[i]
        for j in interv[i]:
            buckets[gene_symbols[int(j)]].append(delta)
    kos = sorted(buckets.keys())
    mat = np.column_stack(
        [np.mean(np.stack(buckets[k], axis=0), axis=0) for k in kos]
    ).astype(np.float32)
    return kos, mat


def pearson_score_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """
    queries: n_q × genes, gallery: genes × n_gal
    returns n_q × n_gal Pearson correlations (shared finite genes).
    """
    q = queries.T  # genes × n_q
    g = gallery
    ok = np.isfinite(q).all(axis=1) & np.isfinite(g).all(axis=1)
    qv = q[ok].astype(np.float64)
    gv = g[ok].astype(np.float64)
    qv = qv - qv.mean(axis=0, keepdims=True)
    gv = gv - gv.mean(axis=0, keepdims=True)
    qn = np.linalg.norm(qv, axis=0) + 1e-12
    gn = np.linalg.norm(gv, axis=0) + 1e-12
    # (n_q × genes_ok) @ (genes_ok × n_gal)
    return (qv.T @ gv) / np.outer(qn, gn)


def eval_split(
    deltas: np.ndarray,
    interv: np.ndarray,
    gene_symbols: list[str],
    test_idx: list[int],
    gallery_kos: list[str],
    gallery_mat: np.ndarray,
) -> dict:
    n_nodes = len(gene_symbols)
    name_to_gal = {k: j for j, k in enumerate(gallery_kos)}
    n_gal = len(gallery_kos)

    queries = deltas[np.asarray(test_idx, dtype=np.int64)]  # n_test × genes
    scores = pearson_score_matrix(queries, gallery_mat)  # n_test × n_gal
    # rank within gallery: 0 = best
    order = np.argsort(-scores, axis=1, kind="mergesort")
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(n_gal)

    recall_at_1, recall_at_10, recall_at_100, recall_at_1000 = [], [], [], []
    ranking_scores, ndcgs = [], []
    n_partial = 0
    per_sample = []

    for t, sample_i in enumerate(test_idx):
        correct_idx = [int(j) for j in interv[sample_i]]
        correct_names = [gene_symbols[j] for j in correct_idx]
        cset = set(correct_names)
        n_c = max(len(cset), 1)

        # global 0-based ranks: gallery genes keep gallery rank; others → n_gal..
        ranks_0 = []
        for cn in correct_names:
            if cn in name_to_gal:
                ranks_0.append(int(inv[t, name_to_gal[cn]]))
            else:
                ranks_0.append(n_gal)  # first unseen slot (pessimistic tie)
        best = min(ranks_0)

        def hit(k: int) -> float:
            return sum(1 for r in ranks_0 if r < k) / n_c

        recall_at_1.append(hit(1))
        recall_at_10.append(hit(10))
        recall_at_100.append(hit(100))
        recall_at_1000.append(hit(1000))

        rs = []
        dcg = 0.0
        for r0 in ranks_0:
            rs.append(1.0 - (r0 / n_nodes))
            rank1 = r0 + 1
            gain = 1.0 - (rank1 / n_nodes)
            discount = 1.0 / np.log2(rank1 + 1)
            dcg += gain * discount
        ranking_scores.extend(rs)
        idcg = _idcg(len(correct_names), n_nodes)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # partial: at least one correct in top-|intervention|
        if any(r < len(correct_names) for r in ranks_0):
            n_partial += 1

        per_sample.append(
            {
                "sample_idx": int(sample_i),
                "true_genes": ",".join(correct_names),
                "rank_true_min": int(best + 1),
                "recall_at_1": recall_at_1[-1],
                "recall_at_10": recall_at_10[-1],
                "in_train_gallery": all(cn in name_to_gal for cn in correct_names),
            }
        )

    return {
        "n_test": len(test_idx),
        "n_gallery_kos": n_gal,
        "recall@1": float(np.mean(recall_at_1)),
        "recall@10": float(np.mean(recall_at_10)),
        "recall@100": float(np.mean(recall_at_100)),
        "recall@1000": float(np.mean(recall_at_1000)),
        "pct_partially_accurate": 100.0 * n_partial / max(len(test_idx), 1),
        "ranking_score": float(np.mean(ranking_scores)) if ranking_scores else float("nan"),
        "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "median_rank_true": float(np.median([r["rank_true_min"] for r in per_sample])),
        "frac_true_in_train_gallery": float(
            np.mean([r["in_train_gallery"] for r in per_sample])
        ),
        "per_sample": per_sample,
    }


def run_cell(
    cell: str,
    data_dir: Path,
    splits_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    split_path = splits_dir / "genetic" / cell / "random" / "5fold" / "splits.pt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    blob = load_or_cache_cell(data_dir, cell)
    gene_symbols = blob["gene_symbols"]
    deltas = blob["deltas"]
    interv = blob["interv"]
    splits = torch.load(split_path, map_location="cpu", weights_only=False)
    _log(f"[{cell}] n_samples={len(deltas)} n_genes={len(gene_symbols)} n_folds={len(splits)}")

    fold_rows = []
    all_per_sample = []
    for fold_idx in sorted(splits.keys()):
        sp = splits[fold_idx]
        train_idx = list(sp["train_index_backward"])
        test_idx = list(sp["test_index_backward"])
        _log(f"[{cell}] fold {fold_idx}: build gallery train={len(train_idx)} test={len(test_idx)}")
        kos, mat = build_train_gallery(deltas, interv, gene_symbols, train_idx)
        metrics = eval_split(deltas, interv, gene_symbols, test_idx, kos, mat)
        _log(
            f"[{cell}] fold {fold_idx}: recall@1={metrics['recall@1']:.4f} "
            f"recall@10={metrics['recall@10']:.4f} "
            f"partial%={metrics['pct_partially_accurate']:.2f} "
            f"median_rank={metrics['median_rank_true']:.1f} "
            f"gallery={metrics['n_gallery_kos']}"
        )
        row = {k: v for k, v in metrics.items() if k != "per_sample"}
        row["cell"] = cell
        row["fold"] = fold_idx
        fold_rows.append(row)
        for r in metrics["per_sample"]:
            r = dict(r)
            r["cell"] = cell
            r["fold"] = fold_idx
            all_per_sample.append(r)

    df = pd.DataFrame(fold_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{cell}_fold_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(all_per_sample).to_csv(
        out_dir / f"{cell}_per_sample.tsv", sep="\t", index=False
    )
    summary = {
        "cell": cell,
        "n_folds": len(df),
        **{
            m: f"{df[m].mean():.6f}±{df[m].std():.6f}"
            for m in [
                "recall@1",
                "recall@10",
                "recall@100",
                "recall@1000",
                "pct_partially_accurate",
                "ranking_score",
                "ndcg",
                "median_rank_true",
            ]
        },
    }
    (out_dir / f"{cell}_summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"[{cell}] summary: {json.dumps(summary, indent=2)}")
    return df


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
        default=_ROOT / "reverse/results/pdgrapher_genetic_compare",
    )
    p.add_argument("--cells", nargs="+", default=["A549", "MCF7"])
    args = p.parse_args()

    if not (args.data_dir / f"data_backward_{args.cells[0]}.pt").exists():
        for c in [
            _ROOT / "reverse/external/PDGrapher/data/processed/torch_data/real_lognorm",
            _ROOT / "reverse/external/PDGrapher/data/processed/real_lognorm",
        ]:
            if (c / f"data_backward_{args.cells[0]}.pt").exists():
                args.data_dir = c
                break

    frames = []
    for cell in args.cells:
        frames.append(run_cell(cell, args.data_dir, args.splits_dir, args.out_dir))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.out_dir / "all_fold_metrics.tsv", sep="\t", index=False)
    rows = []
    for cell, g in all_df.groupby("cell"):
        rows.append(
            {
                "cell": cell,
                "recall@1_mean": g["recall@1"].mean(),
                "recall@1_std": g["recall@1"].std(),
                "recall@10_mean": g["recall@10"].mean(),
                "recall@10_std": g["recall@10"].std(),
                "partial_pct_mean": g["pct_partially_accurate"].mean(),
                "partial_pct_std": g["pct_partially_accurate"].std(),
                "ndcg_mean": g["ndcg"].mean(),
                "median_rank_mean": g["median_rank_true"].mean(),
            }
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "summary_by_cell.tsv", sep="\t", index=False)
    _log(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
