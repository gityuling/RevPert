#!/usr/bin/env python3
"""Train UniPert-emb → ΔY MLP on HepG2 Essential (one-line probe; not for manuscript).

Frozen UniPert gene embeddings + trainable forward head on seed-1 train KOs,
then predicted-gallery Pearson reverse vs L3 / GEARS / UniPert-ridge / RevPert ref.
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

from reverse.src.cell_lines import resolve_cell_paths  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    clean_ko,
    load_ctrl_from_perturb_processed,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_split_kos  # noqa: E402


def build_delta_gallery(pred_dir: Path, dataset_h5ad: Path):
    genes, pred_abs = load_prediction_dir(pred_dir)
    ctrl = load_ctrl_from_perturb_processed(dataset_h5ad, genes)
    return genes, absolute_to_delta(pred_abs, ctrl)


def pearson_ranks(
    query_Y: np.ndarray,
    query_kos: list[str],
    gallery: dict[str, np.ndarray],
    gallery_kos: list[str],
    method: str,
) -> pd.DataFrame:
    g_rows, present = [], []
    for k in gallery_kos:
        if k in gallery:
            g_rows.append(gallery[k])
            present.append(True)
        else:
            g_rows.append(np.zeros(query_Y.shape[1], dtype=float))
            present.append(False)
    g_mat = np.stack(g_rows, axis=0)
    present_a = np.asarray(present)
    ok = np.isfinite(query_Y).all(axis=0) & np.isfinite(g_mat[present_a]).all(axis=0)
    q = query_Y[:, ok]
    g = g_mat[:, ok]
    q = q - q.mean(axis=1, keepdims=True)
    g = g - g.mean(axis=1, keepdims=True)
    qsd = q.std(axis=1, keepdims=True)
    gsd = g.std(axis=1, keepdims=True)
    qsd[qsd < 1e-12] = 1.0
    gsd[gsd < 1e-12] = 1.0
    scores = (q / qsd) @ (g / gsd).T / max(q.shape[1], 1)
    scores[:, ~present_a] = -np.inf
    g_index = {k: i for i, k in enumerate(gallery_kos)}
    rows = []
    for i, ko in enumerate(query_kos):
        order = np.argsort(-scores[i], kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        j = g_index[ko]
        rows.append(
            {
                "true_ko": ko,
                "rank": int(ranks[j]),
                "score": float(scores[i, j]) if np.isfinite(scores[i, j]) else float("nan"),
                "best_ko": gallery_kos[int(order[0])],
                "method": method,
            }
        )
    return pd.DataFrame(rows)


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
        out[ks] = vec
        out[clean_ko(ks)] = vec
        out[ks.upper()] = vec
        out[clean_ko(ks).upper()] = vec
    return out


def get_emb(emb: dict[str, np.ndarray], ko: str) -> np.ndarray | None:
    if ko in emb:
        return emb[ko]
    if ko.upper() in emb:
        return emb[ko.upper()]
    return None


def stack_xy(kos: list[str], emb: dict[str, np.ndarray], obs: dict[str, np.ndarray]):
    xs, ys, kept = [], [], []
    for k in kos:
        e = get_emb(emb, k)
        if e is None or k not in obs:
            continue
        xs.append(e)
        ys.append(np.nan_to_num(obs[k], nan=0.0).astype(np.float32))
        kept.append(k)
    return np.stack(xs), np.stack(ys), kept


@torch.no_grad()
def predict_dict(model: nn.Module, kos: list[str], emb: dict[str, np.ndarray], device: torch.device):
    model.eval()
    out = {}
    for k in kos:
        e = get_emb(emb, k)
        if e is None:
            continue
        x = torch.tensor(e[None, :], device=device)
        out[k] = model(x).cpu().numpy()[0].astype(np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_line", default="hepg2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument(
        "--emb_path",
        default=str(_ROOT / "reverse/external/unipert_probe/assets/current_model/unipert_reps.pkl"),
    )
    ap.add_argument(
        "--out_dir",
        default=str(_ROOT / "reverse/results/revpert/unipert_probe/hepg2_train"),
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    paths = resolve_cell_paths(args.cell_line, seed=args.seed)
    split = load_split_kos(Path(paths["split"]))
    genes, l3_gal = build_delta_gallery(Path(paths["pred_dir"]), Path(paths["dataset_h5ad"]))
    _, obs = load_observed_deltas(Path(paths["pseudobulk_deltas"]), genes)
    emb = load_unipert_emb(Path(args.emb_path))

    gears_gal = None
    gears_dir = Path(paths["gears_dir"])
    if (gears_dir / "all_predictions.json").is_file():
        g_genes, g_raw = build_delta_gallery(gears_dir, Path(paths["dataset_h5ad"]))
        idx = {g: i for i, g in enumerate(g_genes)}
        gears_gal = {
            k: np.array([vec[idx[g]] if g in idx else np.nan for g in genes], dtype=float)
            for k, vec in g_raw.items()
        }

    catalog = sorted(set(obs) & set(l3_gal))
    if gears_gal is not None:
        catalog = sorted(set(catalog) & set(gears_gal))
    catalog = [k for k in catalog if get_emb(emb, k) is not None]

    train_kos = [clean_ko(k) for k in split["train"] if clean_ko(k) in catalog]
    val_kos = [clean_ko(k) for k in split["val"] if clean_ko(k) in catalog]
    test_kos = [clean_ko(k) for k in split["test"] if clean_ko(k) in catalog]
    print(f"catalog={len(catalog)} train={len(train_kos)} val={len(val_kos)} test={len(test_kos)}", flush=True)

    Xtr, Ytr, train_kept = stack_xy(train_kos, emb, obs)
    Xva, Yva, val_kept = stack_xy(val_kos, emb, obs)
    print(f"emb_dim={Xtr.shape[1]} n_genes={Ytr.shape[1]}", flush=True)

    # --- Ridge baseline (quick) ---
    ridge = Ridge(alpha=args.ridge_alpha)
    ridge.fit(Xtr, Ytr)
    ridge_gal = {k: ridge.predict(get_emb(emb, k).reshape(1, -1))[0] for k in catalog}

    # --- MLP train ---
    model = Emb2DeltaMLP(Xtr.shape[1], Ytr.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    dl = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    Xva_t = torch.tensor(Xva, device=device)
    Yva_t = torch.tensor(Yva, device=device)

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva_t), Yva_t).item())
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_mse": float(np.mean(tr_losses)), "val_mse": val_loss})
        print(f"epoch {epoch:03d} train_mse={np.mean(tr_losses):.5f} val_mse={val_loss:.5f}", flush=True)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "args": vars(args), "best_val_mse": best_val}, out / "mlp_best.pt")
    pd.DataFrame(history).to_csv(out / "train_history.tsv", sep="\t", index=False)

    mlp_gal = predict_dict(model, catalog, emb, device)
    # save galleries (compact: only keys)
    with open(out / "mlp_gallery_keys.json", "w") as f:
        json.dump({"n": len(mlp_gal), "n_genes": len(genes)}, f)

    queries = [k for k in test_kos if k in mlp_gal and k in catalog]
    Q = np.stack([obs[k] for k in queries], axis=0)
    frames = [
        pearson_ranks(Q, queries, mlp_gal, catalog, "unipert_mlp_gallery_pearson"),
        pearson_ranks(Q, queries, ridge_gal, catalog, "unipert_ridge_gallery_pearson"),
        pearson_ranks(Q, queries, {k: l3_gal[k] for k in catalog}, catalog, "linear_L3_gallery_pearson"),
    ]
    if gears_gal is not None:
        frames.append(
            pearson_ranks(Q, queries, {k: gears_gal[k] for k in catalog}, catalog, "gears_gallery_pearson")
        )

    ranks = pd.concat(frames, ignore_index=True)
    ranks.to_csv(out / "ranks.tsv", sep="\t", index=False)
    rows = []
    for method, sub in ranks.groupby("method"):
        row = summarize_recovery(sub)
        row["method"] = method
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("median_rank")
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)

    # RevPert seed1 reference if present
    rev = _ROOT / f"reverse/results/revpert/essential/seed1_per_query_ranks/{args.cell_line}_seed1_ranks.tsv"
    meta = {
        "cell_line": paths["label"],
        "seed": args.seed,
        "n_catalog": len(catalog),
        "n_test": len(queries),
        "best_val_mse": best_val,
        "protocol": "Frozen UniPert emb → train MLP(emb→ΔY) on Essential train → Pearson reverse",
        "revpert_seed1_median_rank_ref": 266.0 if args.cell_line == "hepg2" else None,
        "note": "Exploratory only; not manuscript. Not full UniPert-GEARS graph training.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
