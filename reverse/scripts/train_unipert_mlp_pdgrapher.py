#!/usr/bin/env python3
"""UniPert-emb → ΔY MLP on PDGrapher genetic 10 lines (exploratory; not manuscript).

Per official 5-fold split (default: all folds):
  - Build train mean-ΔY gallery per gene (same as pearson_train_gallery)
  - Fit MLP: UniPert(emb) → train mean ΔY
  - Predicted gallery for genes with UniPert coverage
  - Score test queries with PDGrapher reverse metrics
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.io_gallery import clean_ko  # noqa: E402
from reverse.src.run_pdgrapher_genetic_compare import (  # noqa: E402
    build_train_gallery,
    eval_split,
    load_or_cache_cell,
)


class Emb2DeltaMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_unipert_emb(path: Path) -> dict[str, np.ndarray]:
    raw = pickle.loads(path.read_bytes())
    out: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        vec = np.asarray(v, dtype=np.float32).ravel()
        if vec.size == 0:
            continue
        ks = str(k)
        for key in (ks, clean_ko(ks), ks.upper(), clean_ko(ks).upper()):
            out[key] = vec
    return out


def get_emb(emb: dict[str, np.ndarray], ko: str) -> np.ndarray | None:
    if ko in emb:
        return emb[ko]
    if ko.upper() in emb:
        return emb[ko.upper()]
    return None


def train_mlp(
    Xtr: np.ndarray,
    Ytr: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    patience: int,
) -> Emb2DeltaMLP:
    model = Emb2DeltaMLP(Xtr.shape[1], Ytr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    dl = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
        batch_size=batch_size,
        shuffle=True,
    )
    # hold out 10% of train genes for early stopping
    n = len(Xtr)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_va = max(8, int(0.1 * n))
    va_idx, tr_idx = perm[:n_va], perm[n_va:]
    Xva_t = torch.tensor(Xtr[va_idx], device=device)
    Yva_t = torch.tensor(Ytr[va_idx], device=device)
    dl = DataLoader(
        TensorDataset(torch.tensor(Xtr[tr_idx]), torch.tensor(Ytr[tr_idx])),
        batch_size=batch_size,
        shuffle=True,
    )

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xva_t), Yva_t).item())
        if val < best_val - 1e-6:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def mlp_gallery(model: Emb2DeltaMLP, kos: list[str], emb: dict[str, np.ndarray], device: torch.device):
    model.eval()
    kept, mats = [], []
    for k in kos:
        e = get_emb(emb, k)
        if e is None:
            continue
        pred = model(torch.tensor(e[None, :], device=device)).cpu().numpy()[0]
        kept.append(k)
        mats.append(pred.astype(np.float32))
    if not kept:
        return [], np.zeros((0, 0), dtype=np.float32)
    return kept, np.column_stack(mats)


def run_cell(
    cell: str,
    data_dir: Path,
    splits_dir: Path,
    emb: dict[str, np.ndarray],
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> pd.DataFrame:
    split_path = splits_dir / "genetic" / cell / "random" / "5fold" / "splits.pt"
    blob = load_or_cache_cell(data_dir, cell)
    gene_symbols = blob["gene_symbols"]
    deltas = blob["deltas"]
    interv = blob["interv"]
    splits = torch.load(split_path, map_location="cpu", weights_only=False)
    fold_ids = sorted(splits.keys()) if args.folds < 0 else list(range(args.folds))

    rows = []
    for fold_idx in fold_ids:
        if fold_idx not in splits:
            continue
        sp = splits[fold_idx]
        train_idx = list(sp["train_index_backward"])
        test_idx = list(sp["test_index_backward"])
        train_kos, train_mat = build_train_gallery(deltas, interv, gene_symbols, train_idx)
        # train_mat: genes × n_gal
        xs, ys, kept = [], [], []
        for j, k in enumerate(train_kos):
            e = get_emb(emb, k)
            if e is None:
                continue
            xs.append(e)
            ys.append(train_mat[:, j])
            kept.append(k)
        if len(kept) < 20:
            print(f"[{cell}] fold{fold_idx}: too few emb-covered train genes ({len(kept)})", flush=True)
            continue
        Xtr = np.stack(xs).astype(np.float32)
        Ytr = np.stack(ys).astype(np.float32)

        # ridge
        ridge = Ridge(alpha=args.ridge_alpha)
        ridge.fit(Xtr, Ytr)
        r_kos, r_cols = [], []
        for k in gene_symbols:
            e = get_emb(emb, k)
            if e is None:
                continue
            r_kos.append(k)
            r_cols.append(ridge.predict(e.reshape(1, -1))[0].astype(np.float32))
        ridge_mat = np.column_stack(r_cols)

        # mlp
        model = train_mlp(
            Xtr, Ytr, device, args.epochs, args.batch_size, args.lr, args.hidden, args.patience
        )
        m_kos, m_mat = mlp_gallery(model, list(gene_symbols), emb, device)

        methods = {
            "pearson_train_gallery": (train_kos, train_mat),
            "unipert_ridge_gallery": (r_kos, ridge_mat),
            "unipert_mlp_gallery": (m_kos, m_mat),
        }
        for method, (kos, mat) in methods.items():
            metrics = eval_split(deltas, interv, gene_symbols, test_idx, kos, mat)
            rows.append(
                {
                    "cell": cell,
                    "fold": fold_idx,
                    "method": method,
                    "n_train_genes_fit": len(kept),
                    "n_gallery": len(kos),
                    "n_test": metrics["n_test"],
                    "median_rank": metrics["median_rank_true"],
                    "recall@10": metrics["recall@10"],
                    "recall@1": metrics["recall@1"],
                    "ndcg": metrics["ndcg"],
                    "frac_true_in_gallery": metrics["frac_true_in_train_gallery"],
                }
            )
            print(
                f"[{cell}] fold{fold_idx} {method}: med_rank={metrics['median_rank_true']:.1f} "
                f"R@10={metrics['recall@10']:.4f} gal={len(kos)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cells",
        nargs="+",
        default=["A375", "A549", "AGS", "BICR6", "ES2", "HT29", "MCF7", "PC3", "U251MG", "YAPC"],
    )
    ap.add_argument(
        "--data_dir",
        default=str(_ROOT / "reverse/external/PDGrapher/data/processed/torch_data/real_lognorm"),
    )
    ap.add_argument(
        "--splits_dir",
        default=str(_ROOT / "reverse/external/PDGrapher/data/processed/splits"),
    )
    ap.add_argument(
        "--emb_path",
        default=str(_ROOT / "reverse/external/unipert_probe/assets/current_model/unipert_reps.pkl"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(_ROOT / "reverse/results/revpert/unipert_probe/pdgrapher10"),
    )
    ap.add_argument("--folds", type=int, default=5, help="use first N folds; -1 = all")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    emb = load_unipert_emb(Path(args.emb_path))
    print(f"device={device} unipert_keys≈{len(emb)}", flush=True)

    frames = []
    for cell in args.cells:
        print(f"\n===== {cell} =====", flush=True)
        df = run_cell(
            cell,
            Path(args.data_dir),
            Path(args.splits_dir),
            emb,
            out,
            device,
            args,
        )
        df.to_csv(out / f"{cell}_fold_metrics.tsv", sep="\t", index=False)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "all_fold_metrics.tsv", sep="\t", index=False)
    # macro mean± over folds
    agg = (
        all_df.groupby(["cell", "method"], as_index=False)
        .agg(
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
            recall_at_10_mean=("recall@10", "mean"),
            recall_at_10_std=("recall@10", "std"),
            ndcg_mean=("ndcg", "mean"),
            n_folds=("fold", "nunique"),
        )
    )
    agg.to_csv(out / "summary_by_cell.tsv", sep="\t", index=False)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "protocol": "PDGrapher genetic official folds; UniPert frozen emb → MLP/Ridge → predicted gallery reverse",
                "note": "Exploratory only; not manuscript.",
                "cells": args.cells,
                "folds": args.folds,
            },
            indent=2,
        )
    )
    print(agg.to_string(index=False), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
