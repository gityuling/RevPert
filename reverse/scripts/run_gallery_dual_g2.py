#!/usr/bin/env python3
"""Gallery-Dual: evolve reverse retrieval ON expression gallery for BOTH tasks.

Catalog object = predicted ΔŶ(g)  (not gene prototypes)
score_learn(g) = <f_θ(PCA(q)),  g_φ(PCA(ΔŶ(g)))>
score_fuse(g)  = z(pearson(q,ΔŶ)) + α · z(score_learn)

Train: InfoNCE(obs query → pred gallery) + soft Pearson-rank preservation
       so identity improves without destroying connectivity geometry.

Eval:
  - HepG2 identity (test): Pearson / G-Dual / fused@α*
  - Resistance dual-arm on ±Δ: same three scorers
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
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import resolve_cell_paths  # noqa: E402
from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv  # noqa: E402
from reverse.src.gwps_reverse import load_gwps_deltas  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_prediction_dir,
)
from reverse.src.reverse_data import load_reverse_bundle  # noqa: E402
from reverse.src.reverse_model import PCAProjector  # noqa: E402
from reverse.src.score import score_gallery  # noqa: E402

SIG = _ROOT / "reverse/data/signatures"
OUT = _ROOT / "reverse/results/gallery_dual_g2"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _summarize(ranks: np.ndarray) -> dict:
    r = np.asarray(ranks, float)
    return {
        "n": int(len(r)),
        "median_rank": float(np.median(r)),
        "mean_rank": float(np.mean(r)),
        "mrr": float(np.mean(1.0 / r)),
        "recall@1": float(np.mean(r <= 1)),
        "recall@10": float(np.mean(r <= 10)),
        "recall@100": float(np.mean(r <= 100)),
    }


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


class ProfileEncoder(nn.Module):
    def __init__(self, in_dim: int, emb_dim: int = 128, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class GalleryDual(nn.Module):
    """Encode PCA profiles for query and gallery (optionally shared tower)."""

    def __init__(
        self,
        pca: PCAProjector,
        emb_dim: int = 128,
        hidden: int = 512,
        dropout: float = 0.1,
        shared: bool = True,
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(pca.mean_))
        self.register_buffer("components", torch.tensor(pca.components_))
        k = int(pca.n_components)
        self.f = ProfileEncoder(k, emb_dim=emb_dim, hidden=hidden, dropout=dropout)
        self.g = self.f if shared else ProfileEncoder(k, emb_dim=emb_dim, hidden=hidden, dropout=dropout)
        self.shared = shared
        self.logit_scale = nn.Parameter(torch.tensor(2.3))

    def to_pca(self, y: torch.Tensor) -> torch.Tensor:
        y0 = torch.nan_to_num(y, nan=0.0)
        return (y0 - self.mean) @ self.components.T

    def encode_query(self, y: torch.Tensor) -> torch.Tensor:
        return self.f(self.to_pca(y))

    def encode_gallery(self, y: torch.Tensor) -> torch.Tensor:
        return self.g(self.to_pca(y))

    def scores(self, q: torch.Tensor, gal: torch.Tensor) -> torch.Tensor:
        """q (B,G), gal (N,G) → (B,N) cosine in embedding space."""
        zq = self.encode_query(q)
        zg = self.encode_gallery(gal)
        return zq @ zg.T


def pearson_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Row-wise pearson via z-cosine. queries (B,G), gallery (N,G)."""
    def z(X):
        X = np.nan_to_num(X, nan=0.0)
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        return (X - mu) / sd

    q = z(queries)
    g = z(gallery)
    return (q @ g.T) / q.shape[1]


@torch.no_grad()
def score_learned_np(
    model: GalleryDual,
    queries: np.ndarray,
    gal_mat: np.ndarray,
    device: torch.device,
    batch: int = 256,
) -> np.ndarray:
    model.eval()
    g = torch.tensor(gal_mat, dtype=torch.float32, device=device)
    zg = model.encode_gallery(g)
    outs = []
    for i in range(0, len(queries), batch):
        qb = torch.tensor(queries[i : i + batch], dtype=torch.float32, device=device)
        zq = model.encode_query(qb)
        outs.append((zq @ zg.T).cpu().numpy())
    return np.concatenate(outs, axis=0)


def fuse_scores(pearson_s: np.ndarray, learn_s: np.ndarray, alpha: float) -> np.ndarray:
    """Per-query zscore fuse. shapes (B,N). alpha=0 → pearson; +inf → learn."""
    if alpha <= 0:
        return pearson_s
    if not np.isfinite(alpha) or alpha > 1e6:
        return learn_s
    out = np.empty_like(pearson_s)
    for i in range(pearson_s.shape[0]):
        out[i] = _zscore_1d(pearson_s[i]) + alpha * _zscore_1d(learn_s[i])
    return out


def eval_identity(
    queries: np.ndarray,
    kos: list[str],
    gal_kos: list[str],
    gal_mat: np.ndarray,
    pearson_s: np.ndarray | None,
    learn_s: np.ndarray | None,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    idx = {k: i for i, k in enumerate(gal_kos)}
    keep = [i for i, k in enumerate(kos) if k in idx]
    kos_k = [kos[i] for i in keep]
    if pearson_s is None:
        pearson_s = pearson_matrix(queries[keep], gal_mat)
    else:
        pearson_s = pearson_s[keep]
    if learn_s is None:
        raise ValueError("learn_s required")
    learn_s = learn_s[keep]
    scores = fuse_scores(pearson_s, learn_s, alpha)
    ranks = np.array([int(_ranks_desc(scores[i])[idx[k]]) for i, k in enumerate(kos_k)])
    return ranks, _summarize(ranks)


def pearson_preserve_loss(
    learn_logits: torch.Tensor,
    pearson_target: torch.Tensor,
    labels: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Encourage learned similarities to follow pearson geometry on negatives+pos.

    Soft CE toward temperature-sharpened pearson distribution over catalog batch rows.
    """
    if weight <= 0:
        return learn_logits.new_zeros(())
    # Restrict to in-batch columns matching labels' gallery rows already in full catalog logits
    # Use full-catalog pearson rows for these queries
    with torch.no_grad():
        t = pearson_target * 20.0
        target = F.softmax(t, dim=-1)
    log_p = F.log_softmax(learn_logits, dim=-1)
    # mix hard label with soft pearson teacher
    n = learn_logits.shape[0]
    hard = torch.zeros_like(target)
    hard[torch.arange(n, device=learn_logits.device), labels] = 1.0
    mix = 0.5 * hard + 0.5 * target
    return weight * F.kl_div(log_p, mix, reduction="batchmean")


def train_model(
    model: GalleryDual,
    Ytr: np.ndarray,
    Ptr: list[str],
    Yva: np.ndarray,
    Pva: list[str],
    gal_kos: list[str],
    gal_mat: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    preserve_w: float,
    alphas: list[float],
) -> tuple[GalleryDual, dict]:
    g_index = {k: i for i, k in enumerate(gal_kos)}
    tr_keep = [i for i, k in enumerate(Ptr) if k in g_index]
    va_keep = [i for i, k in enumerate(Pva) if k in g_index]
    Ytr = np.nan_to_num(Ytr[tr_keep], nan=0.0).astype(np.float32)
    Yva = np.nan_to_num(Yva[va_keep], nan=0.0).astype(np.float32)
    tr_lab = np.array([g_index[Ptr[i]] for i in tr_keep], dtype=np.int64)
    va_kos = [Pva[i] for i in va_keep]

    # precompute pearson for train/val vs full gallery
    _log("  precomputing Pearson matrices ...")
    pear_tr = pearson_matrix(Ytr, gal_mat)
    pear_va = pearson_matrix(Yva, gal_mat)

    gallery_t = torch.tensor(gal_mat, dtype=torch.float32, device=device)
    # Train on fused logits: z(pearson) + softplus(β)·z(learn), β starts small
    # so identity gains ride on Pearson connectivity geometry.
    fuse_beta = nn.Parameter(torch.tensor(-1.0, device=device))  # softplus≈0.31
    opt = torch.optim.AdamW(
        list(model.parameters()) + [fuse_beta], lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    ds = TensorDataset(
        torch.tensor(Ytr, dtype=torch.float32),
        torch.tensor(tr_lab, dtype=torch.long),
        torch.tensor(pear_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    history = []
    best = {
        "score": -1e9,
        "state": None,
        "epoch": 0,
        "alpha": 0.0,
        "val_mrr": 0.0,
        "val_spearman_vs_pearson": 0.0,
        "fuse_beta": 0.0,
    }

    def _zrows_t(X: torch.Tensor) -> torch.Tensor:
        mu = X.mean(dim=-1, keepdim=True)
        sd = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (X - mu) / sd

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for yb, lb, pb in loader:
            yb, lb, pb = yb.to(device), lb.to(device), pb.to(device)
            scale = model.logit_scale.exp().clamp(max=100.0)
            learn = model.scores(yb, gallery_t)
            alpha_t = F.softplus(fuse_beta).clamp(max=5.0)
            fused = _zrows_t(pb) + alpha_t * _zrows_t(learn)
            logits = scale * fused
            loss_id = F.cross_entropy(logits, lb)
            # mild pull of learn tower toward pearson teacher
            loss_pr = pearson_preserve_loss(scale * learn, pb, lb, preserve_w)
            loss = loss_id + loss_pr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()) + [fuse_beta], 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        # val: scan alpha
        learn_va = score_learned_np(model, Yva, gal_mat, device)
        # pearson-vs-learn rank agreement on true labels' score vectors
        # use mean spearman of score vectors (connectivity geometry proxy)
        spears = []
        for i in range(len(Yva)):
            sp = spearmanr(pear_va[i], learn_va[i]).statistic
            if np.isfinite(sp):
                spears.append(float(sp))
        spear_mean = float(np.mean(spears)) if spears else 0.0

        # Checkpoint by fused identity at train-α; retune α under geometry constraint.
        r0, s0 = eval_identity(Yva, va_kos, gal_kos, gal_mat, pear_va, learn_va, 0.0)
        r1, s1 = eval_identity(Yva, va_kos, gal_kos, gal_mat, pear_va, learn_va, 1e9)

        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "val_spearman_vs_pearson": spear_mean,
            "logit_scale": float(model.logit_scale.exp().item()),
            "val_mrr_pearson": s0["mrr"],
            "val_med_pearson": s0["median_rank"],
            "val_mrr_gdual": s1["mrr"],
            "val_med_gdual": s1["median_rank"],
        }

        # Tune α: maximize fused MRR among alphas whose fused scores still
        # correlate with Pearson (mean spearman >= geom_min). Falls back to
        # best constrained-soft score if none pass.
        geom_min = 0.70
        best_ep_alpha, best_ep_mrr, best_ep_med = 0.0, s0["mrr"], s0["median_rank"]
        best_soft = -1e9
        soft_alpha, soft_mrr, soft_med = 0.0, s0["mrr"], s0["median_rank"]
        passed = False
        for a in alphas:
            r, sm = eval_identity(Yva, va_kos, gal_kos, gal_mat, pear_va, learn_va, a)
            fused = fuse_scores(pear_va, learn_va, a)
            spears_f = []
            for i in range(len(Yva)):
                sp = spearmanr(pear_va[i], fused[i]).statistic
                if np.isfinite(sp):
                    spears_f.append(float(sp))
            geom_f = float(np.mean(spears_f)) if spears_f else 0.0
            row[f"val_mrr_a{a}"] = sm["mrr"]
            row[f"val_med_a{a}"] = sm["median_rank"]
            row[f"val_geom_a{a}"] = geom_f
            soft = sm["mrr"] * (0.5 + 0.5 * max(geom_f, 0.0))
            if soft > best_soft:
                best_soft, soft_alpha, soft_mrr, soft_med = soft, a, sm["mrr"], sm["median_rank"]
            if geom_f >= geom_min and (
                (not passed)
                or sm["mrr"] > best_ep_mrr
                or (sm["mrr"] == best_ep_mrr and sm["median_rank"] < best_ep_med)
            ):
                passed = True
                best_ep_alpha, best_ep_mrr, best_ep_med = a, sm["mrr"], sm["median_rank"]
        if not passed:
            best_ep_alpha, best_ep_mrr, best_ep_med = soft_alpha, soft_mrr, soft_med

        row["pick_alpha"] = best_ep_alpha
        row["pick_mrr"] = best_ep_mrr
        row["pick_med"] = best_ep_med
        row["pick_joint"] = best_soft
        row["train_alpha"] = float(F.softplus(fuse_beta).item())
        history.append(row)
        _log(
            f"  ep {ep:02d} loss={row['train_loss']:.3f} α_train={row['train_alpha']:.3f} "
            f"P med={s0['median_rank']:.0f} MRR={s0['mrr']:.4f} | "
            f"G med={s1['median_rank']:.0f} MRR={s1['mrr']:.4f} | "
            f"pick α={best_ep_alpha} med={best_ep_med:.0f} MRR={best_ep_mrr:.4f} "
            f"spear={spear_mean:.3f}"
        )

        # Prefer fused val MRR at train α (connectivity-aware), with gdual as tie-break signal
        a_train = float(F.softplus(fuse_beta).detach().cpu().item())
        r_tr, s_tr = eval_identity(Yva, va_kos, gal_kos, gal_mat, pear_va, learn_va, a_train)
        ckpt_score = s_tr["mrr"] + 0.05 * s1["mrr"] + 0.0001 * (1.0 / max(s_tr["median_rank"], 1.0))

        if ckpt_score > best["score"]:
            best = {
                "score": ckpt_score,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": ep,
                "alpha": a_train,
                "alpha_retune": best_ep_alpha,
                "val_mrr": s_tr["mrr"],
                "val_med": s_tr["median_rank"],
                "val_mrr_gdual": s1["mrr"],
                "val_med_gdual": s1["median_rank"],
                "val_spearman_vs_pearson": spear_mean,
                "fuse_beta": float(fuse_beta.detach().cpu().item()),
            }

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best": {k: best[k] for k in best if k != "state"}, "history": history}


def dualarm_table(
    query: np.ndarray,
    genes: list[str],
    gallery: dict[str, np.ndarray],
    gal_kos: list[str],
    gal_mat: np.ndarray,
    model: GalleryDual | None,
    alpha: float,
    device: torch.device,
) -> pd.DataFrame:
    """Arm A: +query; Arm B: -query. Fuse pearson+learn per arm."""
    # pearson arms via score_gallery for finite-overlap consistency on external genes
    sa = score_gallery(gallery, query, genes, metric="pearson").set_index("ko")["score"]
    sb = score_gallery(gallery, -query, genes, metric="pearson").set_index("ko")["score"]
    pear_a = np.array([sa.get(k, np.nan) for k in gal_kos], float)
    pear_b = np.array([sb.get(k, np.nan) for k in gal_kos], float)

    if model is None or alpha <= 0:
        learn_a = pear_a
        learn_b = pear_b
        sc_a, sc_b = pear_a, pear_b
    else:
        q = query[None].astype(np.float32)
        learn_a = score_learned_np(model, q, gal_mat, device)[0]
        learn_b = score_learned_np(model, (-query)[None].astype(np.float32), gal_mat, device)[0]
        sc_a = fuse_scores(pear_a[None], learn_a[None], alpha)[0]
        sc_b = fuse_scores(pear_b[None], learn_b[None], alpha)[0]

    # nan-safe rank
    def safe_rank(s):
        s2 = np.where(np.isfinite(s), s, -np.inf)
        return _ranks_desc(s2)

    ra, rb = safe_rank(sc_a), safe_rank(sc_b)
    return pd.DataFrame(
        {
            "ko": gal_kos,
            "score_armA": sc_a,
            "score_armB": sc_b,
            "rank_armA": ra.astype(int),
            "rank_armB": rb.astype(int),
            "pref_arm": np.where(ra <= rb, "A", "B"),
            "rank_pref": np.minimum(ra, rb).astype(int),
        }
    )


def focus_from_df(df: pd.DataFrame, genes: list[str], tag: str, method: str) -> list[dict]:
    rows = []
    for g in genes:
        hit = df[df.ko == g]
        row = {"tag": tag, "method": method, "gene": g, "in_gallery": not hit.empty}
        if not hit.empty:
            r = hit.iloc[0]
            row.update(
                {
                    "rank_armA": int(r.rank_armA),
                    "rank_armB": int(r.rank_armB),
                    "pref_arm": r.pref_arm,
                    "rank_pref": int(r.rank_pref),
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_line", default="hepg2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--preserve_w", type=float, default=0.5)
    ap.add_argument("--shared", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=OUT)
    ap.add_argument("--skip_gwps", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(
        args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]

    paths = resolve_cell_paths(args.cell_line, seed=args.seed)
    _log(f"Loading {args.cell_line} seed={args.seed} ...")
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

    all_kos = []
    for part in ("train", "val", "test"):
        all_kos.extend(tables[part]["kos"])
    gal_kos = [k for k in dict.fromkeys(all_kos) if k in pred_gal]
    gal_mat = np.stack([np.nan_to_num(pred_gal[k], nan=0.0) for k in gal_kos], axis=0).astype(np.float32)
    _log(f"  genes={len(genes)} gallery={len(gal_kos)} train={len(tables['train']['kos'])}")

    # PCA on train observed + train predicted gallery (stabilize gallery geometry)
    Y_pca_fit = np.concatenate(
        [
            np.nan_to_num(tables["train"]["Y"], nan=0.0),
            np.stack(
                [
                    np.nan_to_num(pred_gal[k], nan=0.0)
                    for k in tables["train"]["kos"]
                    if k in pred_gal
                ],
                axis=0,
            ),
        ],
        axis=0,
    )
    pca = PCAProjector(n_components=args.pca_dim).fit(Y_pca_fit)
    _log(f"  PCA dim={pca.n_components} shared={bool(args.shared)} preserve_w={args.preserve_w}")

    model = GalleryDual(
        pca,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        shared=bool(args.shared),
    ).to(device)

    _log("Training Gallery-Dual ...")
    model, train_info = train_model(
        model,
        tables["train"]["Y"],
        tables["train"]["kos"],
        tables["val"]["Y"],
        tables["val"]["kos"],
        gal_kos,
        gal_mat,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        preserve_w=args.preserve_w,
        alphas=alphas,
    )
    pd.DataFrame(train_info["history"]).to_csv(out / "train_history.tsv", sep="\t", index=False)
    alpha_star = float(train_info["best"]["alpha"])
    alpha_retune = float(train_info["best"].get("alpha_retune", alpha_star))
    torch.save(
        {
            "model": model.state_dict(),
            "pca": pca.state_dict(),
            "gal_kos": gal_kos,
            "alpha_star": alpha_star,
            "alpha_retune": alpha_retune,
            "args": vars(args),
            "best": train_info["best"],
        },
        out / "gallery_dual_best.pt",
    )
    _log(
        f"Selected epoch={train_info['best']['epoch']} α_train={alpha_star:.3f} "
        f"α_retune={alpha_retune} val_mrr_fuse={train_info['best']['val_mrr']:.4f} "
        f"val_mrr_G={train_info['best'].get('val_mrr_gdual', float('nan')):.4f}"
    )

    # Identity test
    _log("\n=== Identity recovery (test) ===")
    Yte = np.nan_to_num(tables["test"]["Y"], nan=0.0).astype(np.float32)
    Kte = tables["test"]["kos"]
    pear_te = pearson_matrix(Yte, gal_mat)
    learn_te = score_learned_np(model, Yte, gal_mat, device)
    id_rows = []
    for name, a in [
        ("pearson", 0.0),
        ("gallery_dual", 1e9),
        ("fuse_alpha_train", alpha_star),
        ("fuse_alpha_retune", alpha_retune),
    ]:
        r, sm = eval_identity(Yte, Kte, gal_kos, gal_mat, pear_te, learn_te, a)
        id_rows.append({"method": name, "alpha": a, **sm})
        _log(f"  {name:18s} med={sm['median_rank']:.0f} MRR={sm['mrr']:.4f} R@10={sm['recall@10']:.3f}")
    # also scan alphas on test for analysis (not for selection)
    scan = []
    for a in alphas:
        r, sm = eval_identity(Yte, Kte, gal_kos, gal_mat, pear_te, learn_te, a)
        scan.append({"alpha": a, **sm})
    pd.DataFrame(id_rows).to_csv(out / "identity_test_summary.tsv", sep="\t", index=False)
    pd.DataFrame(scan).to_csv(out / "identity_test_alpha_scan.tsv", sep="\t", index=False)

    # Resistance
    _log("\n=== Resistance proving ground ===")
    focus_hepg2 = ["AURKA", "METTL3", "NCOA4", "TFRC", "PKM", "PTEN", "MCL1", "KEAP1", "MYC"]
    focus_pat = ["METTL3", "AURKA", "NCOA4", "TFRC", "PKM", "CCNK"]
    queries = [
        {
            "tag": "gse322742_hepg2_sorafenib",
            "delta": SIG / "hepg2_sorafenib_delta_y_star.tsv",
            "focus": focus_hepg2,
            "gallery": pred_gal,
            "genes": genes,
            "gal_kos": gal_kos,
            "gal_mat": gal_mat,
            "model": model,
        },
        {
            "tag": "gse143233_patient_sr_vs_normal",
            "delta": SIG / "gse143233_resistant_minus_normal_delta_y_star.tsv",
            "focus": focus_pat,
            "gallery": pred_gal,
            "genes": genes,
            "gal_kos": gal_kos,
            "gal_mat": gal_mat,
            "model": model,
        },
    ]

    focus_rows: list[dict] = []
    methods = [
        ("pearson", 0.0),
        ("gallery_dual", 1e9),
        ("fuse_alpha_train", alpha_star),
        ("fuse_alpha_retune", alpha_retune),
    ]
    resist_scan_alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
    for q in queries:
        star = load_vector_tsv(q["delta"])
        query = align_delta_y_star(star, q["genes"]).astype(np.float32)
        qdir = out / q["tag"]
        qdir.mkdir(parents=True, exist_ok=True)
        for mname, a in methods:
            df = dualarm_table(
                query,
                q["genes"],
                q["gallery"],
                q["gal_kos"],
                q["gal_mat"],
                q["model"],
                a,
                device,
            )
            df.to_csv(qdir / f"{mname}_dualarm.tsv", sep="\t", index=False)
            focus_rows.extend(focus_from_df(df, q["focus"], q["tag"], mname))
        sub = pd.DataFrame([r for r in focus_rows if r["tag"] == q["tag"]])
        wide = sub.pivot_table(index="gene", columns="method", values="rank_pref", aggfunc="first")
        _log(f"\n{q['tag']}")
        _log(wide.to_string())
        scan_rows = []
        for a in resist_scan_alphas:
            df = dualarm_table(
                query, q["genes"], q["gallery"], q["gal_kos"], q["gal_mat"], q["model"], a, device
            )
            for g in q["focus"]:
                hit = df[df.ko == g]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                scan_rows.append(
                    {
                        "tag": q["tag"],
                        "alpha": a,
                        "gene": g,
                        "rank_pref": int(r.rank_pref),
                        "pref_arm": r.pref_arm,
                    }
                )
        pd.DataFrame(scan_rows).to_csv(qdir / "resistance_alpha_scan.tsv", sep="\t", index=False)

    # CML / GWPS
    _log("\n=== GWPS CML Gallery-Dual ===")
    gwps_genes, gwps_gal = load_gwps_deltas()
    split_hits = list(
        (_ROOT / "linear_perturbation_prediction-Paper-main/benchmark/working_dir/results").glob(
            "*gwps*split*"
        )
    )
    cml_focus = ["MYB", "STAT5A", "STAT5B", "RUNX1", "BCR", "ABL1"]
    cml_delta = SIG / "gse120932_k562_ir_with_drug_delta_y_star.tsv"
    star = load_vector_tsv(cml_delta)
    query_cml = align_delta_y_star(star, gwps_genes).astype(np.float32)
    qdir = out / "gse120932_k562_ir_with_drug"
    qdir.mkdir(parents=True, exist_ok=True)

    if (not args.skip_gwps) and split_hits:
        from reverse.src.io_gallery import clean_ko

        sp = split_hits[0]
        raw = json.loads(Path(sp).read_text())
        split = {k: [clean_ko(x) for x in raw[k]] for k in ("train", "val", "test")}
        kos = [k for k in dict.fromkeys(split["train"] + split["val"] + split["test"]) if k in gwps_gal]
        mat = np.stack([np.nan_to_num(gwps_gal[k], nan=0.0) for k in kos], axis=0).astype(np.float32)
        tr_kos = [k for k in split["train"] if k in gwps_gal]
        va_kos = [k for k in split["val"] if k in gwps_gal]
        Ytr = np.stack([np.nan_to_num(gwps_gal[k], nan=0.0) for k in tr_kos]).astype(np.float32)
        Yva = np.stack([np.nan_to_num(gwps_gal[k], nan=0.0) for k in va_kos]).astype(np.float32)
        # noisy queries: identity on observed gallery is trivial; add noise
        rng = np.random.default_rng(0)
        Ytr_n = Ytr + rng.normal(0, 0.35, size=Ytr.shape).astype(np.float32)
        Yva_n = Yva + rng.normal(0, 0.35, size=Yva.shape).astype(np.float32)
        pca_g = PCAProjector(n_components=min(256, Ytr.shape[0] - 1)).fit(Ytr)
        model_g = GalleryDual(pca_g, emb_dim=128, hidden=512, shared=True).to(device)
        model_g, info_g = train_model(
            model_g,
            Ytr_n,
            tr_kos,
            Yva_n,
            va_kos,
            kos,
            mat,
            device,
            epochs=25,
            batch_size=128,
            lr=1e-3,
            preserve_w=args.preserve_w,
            alphas=alphas,
        )
        a_g = float(info_g["best"]["alpha"])
        gal_dict = {k: gwps_gal[k] for k in kos}
        for mname, a in [("pearson", 0.0), ("gallery_dual", 1e9), ("fuse_alpha_star", a_g)]:
            df = dualarm_table(query_cml, gwps_genes, gal_dict, kos, mat, model_g, a, device)
            df.to_csv(qdir / f"{mname}_dualarm.tsv", sep="\t", index=False)
            focus_rows.extend(focus_from_df(df, cml_focus, "gse120932_k562_ir_with_drug", mname))
        sub = pd.DataFrame([r for r in focus_rows if r["tag"] == "gse120932_k562_ir_with_drug"])
        _log(sub.pivot_table(index="gene", columns="method", values="rank_pref", aggfunc="first").to_string())
        torch.save({"model": model_g.state_dict(), "pca": pca_g.state_dict(), "alpha": a_g}, out / "gallery_dual_gwps.pt")
    else:
        _log("  no GWPS split; skip")

    foc = pd.DataFrame(focus_rows)
    foc.to_csv(out / "resistance_focus_compare.tsv", sep="\t", index=False)
    wide = foc[foc.in_gallery].pivot_table(
        index=["tag", "gene"], columns="method", values="rank_pref", aggfunc="first"
    )
    wide.to_csv(out / "resistance_pref_rank_wide.tsv", sep="\t")
    _log("\n=== Resistance preferred-rank wide ===")
    _log(wide.to_string())

    summary = {
        "identity_test": id_rows,
        "alpha_star": alpha_star,
        "best": train_info["best"],
        "note": "Gallery-Dual on predicted ΔŶ catalog + Pearson residual fuse",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "README.md").write_text(
        "# Gallery-Dual (G2) / RevPert\n\n"
        "Expression-gallery residual scorer: signed Pearson + learned profile similarity.\n"
        "Same scorer for identity recovery and resistance dual-arm.\n",
        encoding="utf-8",
    )
    _log(f"\nWrote {out}")


if __name__ == "__main__":
    main()
