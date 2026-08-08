#!/usr/bin/env python3
"""Gallery-Fuse on PDGrapher genetic 10 lines (same splits / Task-1 metrics).

Train: InfoNCE on z(Pearson)+α·z(GalleryDual) vs train-KO gallery
Eval: median rank of true intervention gene(s) within train gallery
      (same protocol as pearson_train_gallery in the unified table)

Compares to frozen numbers: pdgrapher_official, dual_encoder, pearson_train_gallery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.reverse_model import PCAProjector  # noqa: E402
from reverse.src.run_pdgrapher_genetic_compare import (  # noqa: E402
    _idcg,
    _log,
    build_train_gallery,
    load_or_cache_cell,
    pearson_score_matrix,
)
from reverse.scripts.run_gallery_dual_g2 import (  # noqa: E402
    GalleryDual,
    fuse_scores,
    pearson_matrix,
    score_learned_np,
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

DATA_DIR = _ROOT / "reverse/external/PDGrapher/data/processed/torch_data/real_lognorm"
SPLITS_DIR = _ROOT / "reverse/external/PDGrapher/data/processed/splits"
UNIFIED = _ROOT / "reverse/results/pdgrapher_genetic_task1_unified/main_table_median_rank_r10.tsv"
OUT = _ROOT / "reverse/results/pdgrapher_gallery_fuse"


def _pairs(deltas, interv, gene_symbols, indices, name_to_gal):
    ys, labs = [], []
    for i in indices:
        for j in interv[i]:
            name = gene_symbols[int(j)]
            if name in name_to_gal:
                ys.append(deltas[i])
                labs.append(name_to_gal[name])
    if not ys:
        return np.zeros((0, deltas.shape[1]), np.float32), np.zeros((0,), np.int64)
    return np.stack(ys, 0).astype(np.float32), np.asarray(labs, np.int64)


def _zrows_t(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(dim=-1, keepdim=True)
    sd = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd


def eval_scores_gallery(
    scores: np.ndarray,
    test_idx: list[int],
    interv,
    gene_symbols,
    gallery_kos,
    n_nodes: int,
) -> dict:
    name_to_gal = {k: j for j, k in enumerate(gallery_kos)}
    n_gal = len(gallery_kos)
    order = np.argsort(-scores, axis=1, kind="mergesort")
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(n_gal)

    r1, r10, r100, r1000 = [], [], [], []
    ranking_scores, ndcgs, med_ranks = [], [], []
    n_partial = 0
    for t, sample_i in enumerate(test_idx):
        correct_names = [gene_symbols[int(j)] for j in interv[sample_i]]
        ranks_0 = [
            int(inv[t, name_to_gal[cn]]) if cn in name_to_gal else n_gal for cn in correct_names
        ]
        n_c = max(len(ranks_0), 1)
        med_ranks.append(min(ranks_0) + 1)

        def hit(k):
            return sum(1 for r in ranks_0 if r < k) / n_c

        r1.append(hit(1))
        r10.append(hit(10))
        r100.append(hit(100))
        r1000.append(hit(1000))
        dcg = 0.0
        for r0 in ranks_0:
            ranking_scores.append(1.0 - (r0 / n_nodes))
            rank1 = r0 + 1
            dcg += (1.0 - rank1 / n_nodes) / np.log2(rank1 + 1)
        idcg = _idcg(len(ranks_0), n_nodes)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        if any(r < len(correct_names) for r in ranks_0):
            n_partial += 1
    return {
        "n_test": len(test_idx),
        "n_gallery_kos": n_gal,
        "recall@1": float(np.mean(r1)),
        "recall@10": float(np.mean(r10)),
        "recall@100": float(np.mean(r100)),
        "recall@1000": float(np.mean(r1000)),
        "pct_partially_accurate": 100.0 * n_partial / max(len(test_idx), 1),
        "ranking_score": float(np.mean(ranking_scores)),
        "ndcg": float(np.mean(ndcgs)),
        "median_rank_true": float(np.median(med_ranks)),
    }


def train_fold_fuse(
    deltas,
    interv,
    gene_symbols,
    train_idx,
    val_idx,
    gallery_kos,
    gal_mat_gn,  # genes × n_gal
    device,
    epochs=25,
    batch_size=256,
    lr=1e-3,
    preserve_w=0.25,
):
    gal_mat = gal_mat_gn.T.astype(np.float32)  # n_gal × genes
    name_to_gal = {k: j for j, k in enumerate(gallery_kos)}
    Ytr, Ltr = _pairs(deltas, interv, gene_symbols, train_idx, name_to_gal)
    Yva, Lva = _pairs(deltas, interv, gene_symbols, val_idx, name_to_gal)
    if len(Ytr) < batch_size + 2:
        return None, 0.0, {}

    pca = PCAProjector(n_components=min(256, len(Ytr) - 1)).fit(Ytr)
    model = GalleryDual(pca, emb_dim=128, hidden=512, shared=True).to(device)
    fuse_beta = nn.Parameter(torch.tensor(-1.0, device=device))
    opt = torch.optim.AdamW(list(model.parameters()) + [fuse_beta], lr=lr, weight_decay=1e-4)

    pear_tr = pearson_matrix(Ytr, gal_mat)
    pear_va = pearson_matrix(Yva, gal_mat) if len(Yva) else None
    gallery_t = torch.tensor(gal_mat, dtype=torch.float32, device=device)

    ds = TensorDataset(
        torch.tensor(Ytr, dtype=torch.float32),
        torch.tensor(Ltr, dtype=torch.long),
        torch.tensor(pear_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(Ytr) // 2 * 2 or batch_size), shuffle=True, drop_last=True)

    best = {"score": -1e9, "state": None, "alpha": 0.3, "epoch": 0}
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for yb, lb, pb in loader:
            yb, lb, pb = yb.to(device), lb.to(device), pb.to(device)
            scale = model.logit_scale.exp().clamp(max=100.0)
            learn = model.scores(yb, gallery_t)
            alpha_t = F.softplus(fuse_beta).clamp(max=5.0)
            fused = _zrows_t(pb) + alpha_t * _zrows_t(learn)
            loss = F.cross_entropy(scale * fused, lb)
            if preserve_w > 0:
                log_p = F.log_softmax(scale * learn, dim=-1)
                with torch.no_grad():
                    mix = 0.5 * F.one_hot(lb, learn.shape[1]).float() + 0.5 * F.softmax(pb * 20.0, dim=-1)
                loss = loss + preserve_w * F.kl_div(log_p, mix, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + [fuse_beta], 1.0)
            opt.step()
            losses.append(float(loss.item()))

        a = float(F.softplus(fuse_beta).item())
        # val MRR on fused
        if len(Yva) == 0:
            mrr = 0.0
            med = 9999.0
        else:
            learn_va = score_learned_np(model, Yva, gal_mat, device)
            sc = fuse_scores(pear_va, learn_va, a)
            ranks = []
            for i, lab in enumerate(Lva):
                order = np.argsort(-sc[i], kind="mergesort")
                ranks.append(int(np.where(order == lab)[0][0]) + 1)
            ranks = np.asarray(ranks, float)
            mrr = float(np.mean(1.0 / ranks))
            med = float(np.median(ranks))
        score = mrr
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            _log(f"      ep {ep:02d} loss={np.mean(losses):.3f} α={a:.3f} val_MRR={mrr:.4f} med={med:.0f}")
        if score > best["score"]:
            best = {
                "score": score,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "alpha": a,
                "epoch": ep,
                "val_mrr": mrr,
                "val_med": med,
            }
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best["alpha"], best


def run_cell(cell, data_dir, splits_dir, out_dir, device, epochs, batch_size):
    split_path = splits_dir / "genetic" / cell / "random" / "5fold" / "splits.pt"
    blob = load_or_cache_cell(data_dir, cell)
    gene_symbols = blob["gene_symbols"]
    deltas = blob["deltas"]
    interv = blob["interv"]
    splits = torch.load(split_path, map_location="cpu", weights_only=False)
    n_nodes = len(gene_symbols)
    _log(f"[{cell}] samples={len(deltas)} genes={n_nodes} folds={len(splits)}")

    rows = []
    for fold_idx in sorted(splits.keys()):
        sp = splits[fold_idx]
        train_idx = list(sp["train_index_backward"])
        val_idx = list(sp["val_index_backward"])
        test_idx = list(sp["test_index_backward"])
        kos, mat_gn = build_train_gallery(deltas, interv, gene_symbols, train_idx)
        gal_mat = mat_gn.T.astype(np.float32)
        _log(f"[{cell}] fold {fold_idx}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} gal={len(kos)}")

        model, alpha, info = train_fold_fuse(
            deltas,
            interv,
            gene_symbols,
            train_idx,
            val_idx,
            kos,
            mat_gn,
            device,
            epochs=epochs,
            batch_size=batch_size,
        )
        queries = deltas[np.asarray(test_idx)]
        pear = pearson_score_matrix(queries, mat_gn)
        if model is None:
            learn = pear
            alpha = 0.0
        else:
            learn = score_learned_np(model, queries.astype(np.float32), gal_mat, device)
        fused = fuse_scores(pear, learn, alpha)

        m_p = eval_scores_gallery(pear, test_idx, interv, gene_symbols, kos, n_nodes)
        m_g = eval_scores_gallery(learn, test_idx, interv, gene_symbols, kos, n_nodes)
        m_f = eval_scores_gallery(fused, test_idx, interv, gene_symbols, kos, n_nodes)
        for method, m in [
            ("pearson_train_gallery", m_p),
            ("gallery_dual", m_g),
            ("gallery_fuse", m_f),
        ]:
            row = dict(m)
            row.update({"cell": cell, "fold": fold_idx, "method": method, "alpha": alpha})
            rows.append(row)
        _log(
            f"[{cell}] fold {fold_idx}: P med={m_p['median_rank_true']:.0f} "
            f"G med={m_g['median_rank_true']:.0f} F med={m_f['median_rank_true']:.0f} "
            f"α={alpha:.3f} R@10 F={m_f['recall@10']:.4f}"
        )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{cell}_fold_metrics.tsv", sep="\t", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=OUT)
    ap.add_argument("--data_dir", type=Path, default=DATA_DIR)
    ap.add_argument("--splits_dir", type=Path, default=SPLITS_DIR)
    args = ap.parse_args()

    device = torch.device(
        args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cell in args.cells:
        df = run_cell(
            cell,
            args.data_dir,
            args.splits_dir,
            args.out_dir,
            device,
            args.epochs,
            args.batch_size,
        )
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(args.out_dir / "all_fold_metrics.tsv", sep="\t", index=False)

    summary = (
        all_df.groupby(["cell", "method"])
        .agg(
            median_rank_mean=("median_rank_true", "mean"),
            median_rank_std=("median_rank_true", "std"),
            recall_at_10_mean=("recall@10", "mean"),
            recall_at_10_std=("recall@10", "std"),
            ndcg_mean=("ndcg", "mean"),
            n_folds=("fold", "count"),
        )
        .reset_index()
    )
    summary.to_csv(args.out_dir / "summary_by_cell.tsv", sep="\t", index=False)

    # compare to unified table
    if UNIFIED.exists():
        base = pd.read_csv(UNIFIED, sep="\t")
        cmp_rows = []
        for cell in summary.cell.unique():
            for method in ["gallery_fuse", "gallery_dual", "pearson_train_gallery"]:
                s = summary[(summary.cell == cell) & (summary.method == method)]
                if s.empty:
                    continue
                s = s.iloc[0]
                row = {
                    "cell": cell,
                    "method": method,
                    "median_rank_mean": s.median_rank_mean,
                    "recall_at_10_mean": s.recall_at_10_mean,
                }
                for ref in ["pdgrapher_official", "dual_encoder", "pearson_train_gallery"]:
                    b = base[(base.cell == cell) & (base.method == ref)]
                    if not b.empty:
                        row[f"{ref}_median"] = float(b.iloc[0].median_rank_mean)
                        row[f"vs_{ref}_ratio"] = float(b.iloc[0].median_rank_mean) / max(
                            float(s.median_rank_mean), 1e-9
                        )
                cmp_rows.append(row)
        cmp = pd.DataFrame(cmp_rows)
        cmp.to_csv(args.out_dir / "vs_pdgrapher_dual.tsv", sep="\t", index=False)
        _log("\n=== vs PDGrapher / Dual (median rank; ratio>1 means we are better) ===")
        show = cmp[cmp.method == "gallery_fuse"][
            [
                "cell",
                "median_rank_mean",
                "pdgrapher_official_median",
                "vs_pdgrapher_official_ratio",
                "dual_encoder_median",
                "vs_dual_encoder_ratio",
                "pearson_train_gallery_median",
            ]
        ]
        _log(show.to_string(index=False))
        n_beat_pdg = int((show["vs_pdgrapher_official_ratio"] > 1).sum())
        n_beat_dual = int((show["vs_dual_encoder_ratio"] > 1).sum())
        _log(f"\nGallery-Fuse beats PDGrapher: {n_beat_pdg}/{len(show)}")
        _log(f"Gallery-Fuse beats Dual-encoder: {n_beat_dual}/{len(show)}")

    (args.out_dir / "README.md").write_text(
        "# PDGrapher genetic × Gallery-Fuse\n\n"
        "Same official 5-fold backward splits; rank within train-KO gallery.\n",
        encoding="utf-8",
    )
    _log(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
