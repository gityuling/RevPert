#!/usr/bin/env python3
"""Zero-shot KO ΔY prediction via gene-embedding → ΔY MLP.

Train on Essential (screened) KOs that have embeddings + observed ΔY, then
predict for arbitrary genes that have the same embedding table — including
genes never knocked out in the HepG2 Essential screen.

Claim boundary: this is coverage expansion / usage extrapolation. Quality for
unscreened genes is expected to be weaker than in-panel L3 predictions.
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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import resolve_cell_paths  # noqa: E402
from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    clean_ko,
    load_ctrl_from_perturb_processed,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.score import score_gallery, top_k  # noqa: E402

BENCH = _ROOT / "linear_perturbation_prediction-Paper-main" / "benchmark"
DEFAULT_ESM = BENCH / "working_dir/external/esm2/esm2_pert_embedding_1280d.tsv"
DEFAULT_WAVEGC = BENCH / "working_dir/external/wavegc/wavegc_pert_embedding_128d.tsv"


class EmbMLP(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_embedding_table(path: Path) -> pd.DataFrame:
    """Return DataFrame indexed by gene symbol, columns = embedding dims."""
    emb = pd.read_csv(path, sep="\t")
    # deposited as dims × genes
    if emb.shape[0] < emb.shape[1]:
        feat = emb.T
    else:
        feat = emb
    feat.index = feat.index.astype(str)
    return feat.astype(np.float32)


def _read_gene_list(path: Path) -> list[str]:
    genes: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        genes.append(s.split("\t")[0].split(",")[0].strip())
    return genes


def train_and_predict(
    *,
    cell: str,
    embedding_tsv: Path,
    target_kos: list[str],
    epochs: int = 300,
    hidden: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 1,
    device: str | None = None,
) -> dict:
    """Train embedding→ΔY on screened KOs; predict absolute expression for targets."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    paths = resolve_cell_paths(cell)
    genes, l3_pred = load_prediction_dir(Path(paths["pred_dir"]))
    ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
    _, obs = load_observed_deltas(Path(paths["pseudobulk_deltas"]), genes)

    feat = _load_embedding_table(embedding_tsv)
    train_kos = sorted(set(obs) & set(feat.index) & set(l3_pred))
    if len(train_kos) < 10:
        raise RuntimeError(f"Too few train KOs with embedding+obs ({len(train_kos)})")

    x = np.vstack([feat.loc[k].to_numpy(dtype=np.float32) for k in train_kos])
    y = np.vstack([obs[k].astype(np.float32) for k in train_kos])  # ΔY
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_n = (x - x_mean) / x_std

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = EmbMLP(x_n.shape[1], hidden, y.shape[1], dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    xt = torch.from_numpy(x_n).to(dev)
    yt = torch.from_numpy(y).to(dev)
    model.train()
    last_loss = float("nan")
    for ep in range(epochs):
        opt.zero_grad(set_to_none=True)
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
        if ep == 0 or (ep + 1) % 50 == 0:
            print(f"epoch={ep + 1} loss={last_loss:.6f}", flush=True)

    # predict targets
    wanted = []
    missing_emb = []
    for g in target_kos:
        g = clean_ko(g)
        if g in feat.index:
            wanted.append(g)
        else:
            missing_emb.append(g)

    model.eval()
    abs_pred: dict[str, np.ndarray] = {}
    delta_pred: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for g in wanted:
            xn = (feat.loc[g].to_numpy(dtype=np.float32) - x_mean.ravel()) / x_std.ravel()
            d = model(torch.from_numpy(xn[None, :]).to(dev)).cpu().numpy().ravel()
            delta_pred[g] = d.astype(float)
            abs_pred[g] = (d + ctrl).astype(float)

    return {
        "genes": genes,
        "ctrl": ctrl,
        "abs_pred": abs_pred,
        "delta_pred": delta_pred,
        "train_kos": train_kos,
        "missing_embedding": missing_emb,
        "in_panel": sorted(set(wanted) & set(l3_pred)),
        "out_of_panel": sorted(set(wanted) - set(l3_pred)),
        "last_loss": last_loss,
        "embedding_tsv": str(embedding_tsv),
        "n_train": len(train_kos),
        "paths": paths,
        "l3_pred": l3_pred,
    }


def merge_galleries(
    base_abs: dict[str, np.ndarray],
    extra_abs: dict[str, np.ndarray],
    *,
    prefer_extra_for: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Merge L3 (or other) absolute gallery with zero-shot predictions."""
    out = dict(base_abs)
    prefer = prefer_extra_for or set()
    for k, v in extra_abs.items():
        if k not in out or k in prefer:
            out[k] = v
    return out


def cmd_predict(args: argparse.Namespace) -> None:
    if args.genes_file:
        targets = _read_gene_list(Path(args.genes_file))
    elif args.genes:
        targets = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        raise SystemExit("Provide --genes_file or --genes")

    emb = Path(args.embedding_tsv)
    bundle = train_and_predict(
        cell=args.cell_line,
        embedding_tsv=emb,
        target_kos=targets,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # zero-shot-only gallery (absolute expr, L3 gene axis)
    zs_dir = out / "zeroshot_pred"
    zs_dir.mkdir(exist_ok=True)
    (zs_dir / "gene_names.json").write_text(json.dumps(bundle["genes"]))
    (zs_dir / "all_predictions.json").write_text(
        json.dumps({k: v.tolist() for k, v in bundle["abs_pred"].items()})
    )

    # merged: L3 + out-of-panel zero-shot (keep L3 for in-panel)
    merged = merge_galleries(bundle["l3_pred"], bundle["abs_pred"], prefer_extra_for=set())
    # force-add out-of-panel
    for g in bundle["out_of_panel"]:
        if g in bundle["abs_pred"]:
            merged[g] = bundle["abs_pred"][g]
    m_dir = out / "merged_l3_zeroshot_pred"
    m_dir.mkdir(exist_ok=True)
    (m_dir / "gene_names.json").write_text(json.dumps(bundle["genes"]))
    (m_dir / "all_predictions.json").write_text(
        json.dumps({k: v.tolist() for k, v in merged.items()})
    )

    meta = {
        "cell_line": args.cell_line,
        "embedding_tsv": bundle["embedding_tsv"],
        "n_train_screened_kos": bundle["n_train"],
        "last_train_mse": bundle["last_loss"],
        "requested": targets,
        "predicted": sorted(bundle["abs_pred"]),
        "missing_embedding": bundle["missing_embedding"],
        "in_panel_requested": bundle["in_panel"],
        "out_of_panel_predicted": bundle["out_of_panel"],
        "n_merged_gallery": len(merged),
        "claim_boundary": (
            "Zero-shot ΔY for unscreened genes via embedding→MLP; "
            "not wet-lab KO; weaker than in-panel L3; usage expansion only."
        ),
    }
    (out / "unseen_ko_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"Wrote {out}")


def cmd_score_disease(args: argparse.Namespace) -> None:
    """Score ΔY* against merged L3+zeroshot gallery built by ``predict``."""
    pred_dir = Path(args.pred_dir)
    genes, pred_abs = load_prediction_dir(pred_dir)
    paths = resolve_cell_paths(args.cell_line)
    ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
    gal = absolute_to_delta(pred_abs, ctrl)
    star = align_delta_y_star(load_vector_tsv(Path(args.delta_y_star)), genes)
    scored = score_gallery(gal, star, genes, metric=args.metric)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=args.top_k).to_csv(out / f"top{args.top_k}.tsv", sep="\t", index=False)

    focus = []
    if args.genes_file:
        focus = _read_gene_list(Path(args.genes_file))
    elif args.genes:
        focus = [g.strip() for g in args.genes.split(",") if g.strip()]
    if focus:
        sub = scored.set_index("ko")
        rows = []
        for g in focus:
            if g in sub.index:
                rows.append(
                    {
                        "ko": g,
                        "rank": int(sub.loc[g, "rank"]),
                        "score": float(sub.loc[g, "score"]),
                        "in_gallery": True,
                    }
                )
            else:
                rows.append({"ko": g, "rank": None, "score": None, "in_gallery": False})
        pd.DataFrame(rows).to_csv(out / "focus_gene_ranks.tsv", sep="\t", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {out}  gallery_size={len(gal)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot KO prediction for reverse gallery expansion")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("predict", help="Train embedding MLP; dump zeroshot + merged galleries")
    p1.add_argument("--cell_line", default="hepg2")
    p1.add_argument("--embedding_tsv", default=str(DEFAULT_WAVEGC), help="WaveGC (default) or ESM2 TSV")
    p1.add_argument("--genes_file", default=None)
    p1.add_argument("--genes", default=None, help="Comma-separated gene symbols")
    p1.add_argument("--out_dir", default=str(_ROOT / "reverse/results/unseen_ko_hepg2"))
    p1.add_argument("--epochs", type=int, default=300)
    p1.add_argument("--hidden", type=int, default=256)
    p1.add_argument("--seed", type=int, default=1)
    p1.set_defaults(func=cmd_predict)

    p2 = sub.add_parser("score_disease", help="Score disease ΔY* on a (merged) pred dir")
    p2.add_argument("--cell_line", default="hepg2")
    p2.add_argument("--pred_dir", required=True)
    p2.add_argument("--delta_y_star", required=True)
    p2.add_argument("--out_dir", required=True)
    p2.add_argument("--genes_file", default=None)
    p2.add_argument("--genes", default=None)
    p2.add_argument("--metric", default="pearson")
    p2.add_argument("--top_k", type=int, default=50)
    p2.set_defaults(func=cmd_score_disease)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
