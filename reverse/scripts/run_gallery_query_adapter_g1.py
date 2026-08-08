#!/usr/bin/env python3
"""G1: evolve reverse retrieval ON the expression gallery.

Plain Pearson:  s(g) = corr(ΔY*, ΔŶ(g))
G1 adapter:     s(g) = corr(Adapter_θ(ΔY*), ΔŶ(g))

Adapter is a residual denoiser in PCA space (fit on train KOs only):
  z = PCA(ΔY)
  ΔY_hat = ΔY + σ · Decode(MLP(z))
trained with InfoNCE so adapted observed queries retrieve the correct
predicted-gallery KO better than raw Pearson.

Evaluates:
  - HepG2 identity recovery (test): Pearson vs Adapter
  - 3 resistance proving-ground queries, signed dual-arm
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
OUT = _ROOT / "reverse/results/gallery_query_adapter_g1"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
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


def _zrows(X: torch.Tensor) -> torch.Tensor:
    mu = X.mean(dim=-1, keepdim=True)
    sd = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd


class QueryAdapter(nn.Module):
    """Residual query adapter in fixed train-fit PCA coordinates."""

    def __init__(self, pca: PCAProjector, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("mean", torch.tensor(pca.mean_))
        self.register_buffer("components", torch.tensor(pca.components_))  # (k, G)
        k = int(pca.n_components)
        self.mlp = nn.Sequential(
            nn.Linear(k, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k),
        )
        # start near identity (tiny residual)
        self.log_scale = nn.Parameter(torch.tensor(-2.0))  # σ ≈ 0.14
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def encode_pca(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, G)
        return (y - self.mean) @ self.components.T

    def decode_pca(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.components

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y0 = torch.nan_to_num(y, nan=0.0)
        z = self.encode_pca(y0)
        dz = self.mlp(z)
        scale = self.log_scale.exp().clamp(max=3.0)
        return y0 + scale * self.decode_pca(dz)


def pearson_scores_torch(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """queries (B,G), gallery (N,G) → (B,N) pearson via z-cosine."""
    q = _zrows(torch.nan_to_num(queries, nan=0.0))
    g = _zrows(torch.nan_to_num(gallery, nan=0.0))
    return q @ g.T / q.shape[1]


@torch.no_grad()
def eval_part_ranks(
    adapter: QueryAdapter | None,
    Y: np.ndarray,
    kos: list[str],
    gal_kos: list[str],
    gal_mat: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    idx = {k: i for i, k in enumerate(gal_kos)}
    keep = [i for i, k in enumerate(kos) if k in idx]
    Y = Y[keep]
    kos = [kos[i] for i in keep]
    q = torch.tensor(Y, dtype=torch.float32, device=device)
    if adapter is not None:
        adapter.eval()
        q = adapter(q)
    g = torch.tensor(gal_mat, dtype=torch.float32, device=device)
    scores = pearson_scores_torch(q, g).cpu().numpy()
    ranks = []
    for i, k in enumerate(kos):
        ranks.append(int(_ranks_desc(scores[i])[idx[k]]))
    return np.asarray(ranks)


def train_adapter(
    Ytr: np.ndarray,
    Ptr_kos: list[str],
    Yva: np.ndarray,
    Pva_kos: list[str],
    gal_kos: list[str],
    gal_mat: np.ndarray,
    pca: PCAProjector,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 512,
) -> tuple[QueryAdapter, dict]:
    # map train queries to gallery indices
    g_index = {k: i for i, k in enumerate(gal_kos)}
    tr_keep = [i for i, k in enumerate(Ptr_kos) if k in g_index]
    Ytr = Ytr[tr_keep]
    tr_labels = np.array([g_index[Ptr_kos[i]] for i in tr_keep], dtype=np.int64)

    adapter = QueryAdapter(pca, hidden=hidden).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gallery_t = torch.tensor(gal_mat, dtype=torch.float32, device=device)

    ds = TensorDataset(
        torch.tensor(np.nan_to_num(Ytr, nan=0.0), dtype=torch.float32),
        torch.tensor(tr_labels, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best = {"val_mrr": -1.0, "state": None, "epoch": 0}
    history = []
    for ep in range(1, epochs + 1):
        adapter.train()
        losses = []
        for yb, lb in loader:
            yb = yb.to(device)
            lb = lb.to(device)
            q = adapter(yb)
            logits = pearson_scores_torch(q, gallery_t) * 20.0  # temperature-ish
            loss = F.cross_entropy(logits, lb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        sched.step()

        # val
        r_ad = eval_part_ranks(adapter, Yva, Pva_kos, gal_kos, gal_mat, device)
        r_pe = eval_part_ranks(None, Yva, Pva_kos, gal_kos, gal_mat, device)
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "val_mrr_adapter": float(np.mean(1.0 / r_ad)),
            "val_median_adapter": float(np.median(r_ad)),
            "val_mrr_pearson": float(np.mean(1.0 / r_pe)),
            "val_median_pearson": float(np.median(r_pe)),
            "scale": float(adapter.log_scale.exp().item()),
        }
        history.append(row)
        _log(
            f"  ep {ep:02d} loss={row['train_loss']:.3f} "
            f"val_MRR A={row['val_mrr_adapter']:.4f} P={row['val_mrr_pearson']:.4f} "
            f"med A={row['val_median_adapter']:.0f} P={row['val_median_pearson']:.0f} "
            f"σ={row['scale']:.3f}"
        )
        if row["val_mrr_adapter"] > best["val_mrr"]:
            best = {
                "val_mrr": row["val_mrr_adapter"],
                "state": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
                "epoch": ep,
            }

    if best["state"] is not None:
        adapter.load_state_dict(best["state"])
    return adapter, {"best_epoch": best["epoch"], "best_val_mrr": best["val_mrr"], "history": history}


def score_dualarm_adapter(
    adapter: QueryAdapter | None,
    query: np.ndarray,
    genes: list[str],
    gallery: dict[str, np.ndarray],
    device: torch.device,
) -> pd.DataFrame:
    """Arm A: adapt(Δ); Arm B: adapt(−Δ); score each vs gallery by Pearson."""
    kos = sorted(gallery.keys())
    gal_mat = np.stack([np.nan_to_num(gallery[k], nan=0.0) for k in kos], axis=0)

    def adapted(vec: np.ndarray) -> np.ndarray:
        t = torch.tensor(np.nan_to_num(vec, nan=0.0)[None], dtype=torch.float32, device=device)
        if adapter is None:
            out = t
        else:
            adapter.eval()
            with torch.no_grad():
                out = adapter(t)
        return out[0].cpu().numpy()

    qa = adapted(query)
    qb = adapted(-query)
    # use score_gallery for manuscript-consistent finite-overlap pearson
    sa = score_gallery(gallery, qa, genes, metric="pearson").set_index("ko")["score"]
    sb = score_gallery(gallery, qb, genes, metric="pearson").set_index("ko")["score"]
    scores_a = np.array([sa.get(k, np.nan) for k in kos], float)
    scores_b = np.array([sb.get(k, np.nan) for k in kos], float)
    ra = _ranks_desc(np.where(np.isfinite(scores_a), scores_a, -np.inf))
    rb = _ranks_desc(np.where(np.isfinite(scores_b), scores_b, -np.inf))
    return pd.DataFrame(
        {
            "ko": kos,
            "score_armA": scores_a,
            "score_armB": scores_b,
            "rank_armA": ra.astype(int),
            "rank_armB": rb.astype(int),
            "pref_arm": np.where(ra <= rb, "A", "B"),
            "rank_pref": np.minimum(ra, rb).astype(int),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_line", default="hepg2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=OUT)
    args = ap.parse_args()

    device = torch.device(
        args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu"
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

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

    # catalog = intersection of all split KOs that have predicted gallery
    all_split_kos = []
    for part in ("train", "val", "test"):
        all_split_kos.extend(tables[part]["kos"])
    gal_kos = [k for k in dict.fromkeys(all_split_kos) if k in pred_gal]
    gal_mat = np.stack([np.nan_to_num(pred_gal[k], nan=0.0) for k in gal_kos], axis=0).astype(np.float32)
    _log(f"  genes={len(genes)} gallery_kos={len(gal_kos)} train={len(tables['train']['kos'])}")

    # PCA on train observed ΔY only
    pca = PCAProjector(n_components=args.pca_dim).fit(tables["train"]["Y"])
    _log(f"  PCA dim={pca.n_components}")

    _log("Training query adapter (InfoNCE on predicted gallery) ...")
    adapter, train_info = train_adapter(
        tables["train"]["Y"],
        tables["train"]["kos"],
        tables["val"]["Y"],
        tables["val"]["kos"],
        gal_kos,
        gal_mat,
        pca,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
    )
    pd.DataFrame(train_info["history"]).to_csv(out / "train_history.tsv", sep="\t", index=False)
    torch.save(
        {"model": adapter.state_dict(), "pca": pca.state_dict(), "gal_kos": gal_kos, "args": vars(args), "train_info": {k: train_info[k] for k in ("best_epoch", "best_val_mrr")}},
        out / "adapter_best.pt",
    )

    # Identity recovery test
    _log("\n=== Identity recovery (test) ===")
    r_p = eval_part_ranks(None, tables["test"]["Y"], tables["test"]["kos"], gal_kos, gal_mat, device)
    r_a = eval_part_ranks(adapter, tables["test"]["Y"], tables["test"]["kos"], gal_kos, gal_mat, device)
    sum_p, sum_a = _summarize(r_p), _summarize(r_a)
    id_tab = pd.DataFrame(
        [
            {"method": "pearson_gallery", **sum_p},
            {"method": "adapter_then_pearson", **sum_a},
        ]
    )
    id_tab.to_csv(out / "identity_test_summary.tsv", sep="\t", index=False)
    pd.DataFrame({"rank_pearson": r_p, "rank_adapter": r_a}).to_csv(
        out / "identity_test_ranks.tsv", sep="\t", index=False
    )
    _log(id_tab.to_string(index=False))

    # Resistance proving ground
    _log("\n=== Resistance proving ground ===")
    queries = [
        {
            "tag": "gse322742_hepg2_sorafenib",
            "delta": SIG / "hepg2_sorafenib_delta_y_star.tsv",
            "gallery": pred_gal,
            "genes": genes,
            "focus": ["AURKA", "METTL3", "NCOA4", "TFRC", "PKM", "PTEN", "MCL1", "KEAP1", "MYC"],
        },
        {
            "tag": "gse143233_patient_sr_vs_normal",
            "delta": SIG / "gse143233_resistant_minus_normal_delta_y_star.tsv",
            "gallery": pred_gal,
            "genes": genes,
            "focus": ["METTL3", "AURKA", "NCOA4", "TFRC", "PKM", "CCNK"],
        },
    ]
    # CML on GWPS observed gallery — train a separate adapter? For G1 minimal: apply HepG2-trained
    # adapter is HepG2-gene-axis specific, cannot apply to GWPS. Score GWPS with pearson only note,
    # OR fit a quick GWPS adapter if we have splits. For now add GWPS pearson-only + try training
    # a GWPS adapter from GWPS observed self-retrieval (oracle-like) is wrong.
    # Better: train GWPS adapter with held-out identity if split exists.
    focus_rows = []
    for q in queries:
        star = load_vector_tsv(q["delta"])
        query = align_delta_y_star(star, q["genes"])
        df_p = score_dualarm_adapter(None, query, q["genes"], q["gallery"], device)
        df_a = score_dualarm_adapter(adapter, query, q["genes"], q["gallery"], device)
        qdir = out / q["tag"]
        qdir.mkdir(parents=True, exist_ok=True)
        df_p.to_csv(qdir / "pearson_dualarm.tsv", sep="\t", index=False)
        df_a.to_csv(qdir / "adapter_dualarm.tsv", sep="\t", index=False)
        for method, df in [("pearson", df_p), ("adapter", df_a)]:
            for g in q["focus"]:
                hit = df[df.ko == g]
                row = {
                    "tag": q["tag"],
                    "method": method,
                    "gene": g,
                    "in_gallery": not hit.empty,
                    "delta": float(star[g]) if g in star.index else np.nan,
                }
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
                focus_rows.append(row)
        sub = pd.DataFrame(focus_rows)
        sub = sub[sub.tag == q["tag"]]
        _log(f"\n{q['tag']}")
        _log(
            sub.pivot_table(index="gene", columns="method", values="rank_pref", aggfunc="first").to_string()
        )

    # GWPS CML: train a small adapter on GWPS observed gallery identity recovery if split exists
    _log("\n=== GWPS CML branch ===")
    gwps_genes, gwps_gal = load_gwps_deltas()
    split_path = Path(paths["split"]).parent / "seed_1_replogle_k562_gwps_split"
    # resolve_cell_paths hepg2 split parent is working_dir/results
    split_path = _ROOT / "linear_perturbation_prediction-Paper-main/benchmark/working_dir/results/seed_1_replogle_k562_gwps_split"
    cml_focus = ["MYB", "STAT5A", "STAT5B", "RUNX1", "BCR", "ABL1"]
    cml_delta = SIG / "gse120932_k562_ir_with_drug_delta_y_star.tsv"
    star = load_vector_tsv(cml_delta)
    query = align_delta_y_star(star, gwps_genes)
    df_p = score_dualarm_adapter(None, query, gwps_genes, gwps_gal, device)
    qdir = out / "gse120932_k562_ir_with_drug"
    qdir.mkdir(parents=True, exist_ok=True)
    df_p.to_csv(qdir / "pearson_dualarm.tsv", sep="\t", index=False)

    gwps_adapter = None
    if split_path.exists() or Path(str(split_path) + ".json").exists() or split_path.is_file():
        # load_split expects JSON file
        sp = split_path if split_path.is_file() else Path(str(split_path))
        if not sp.is_file():
            # try as path without extension used as stem
            candidates = [
                _ROOT
                / "linear_perturbation_prediction-Paper-main/benchmark/working_dir/results/seed_1_replogle_k562_gwps_split"
            ]
            sp = None
            for c in candidates:
                if c.is_file():
                    sp = c
                    break
                if c.with_suffix("").exists() is False and Path(str(c)).exists():
                    # directory? skip
                    pass
            # list
            import glob as _glob

            hits = list(
                Path(
                    _ROOT
                    / "linear_perturbation_prediction-Paper-main/benchmark/working_dir/results"
                ).glob("*gwps*split*")
            )
            _log(f"  GWPS split candidates: {hits[:5]}")
            sp = hits[0] if hits else None
        if sp is not None and sp.is_file():
            from reverse.src.io_gallery import clean_ko

            raw = json.loads(sp.read_text())
            split = {k: [clean_ko(x) for x in raw[k]] for k in ("train", "val", "test")}
            # build matrices from observed GWPS (oracle gallery = observed; for fair reverse use
            # observed-to-observed would be trivial. For adapter training use leave-one-out style:
            # query=obs, gallery=obs is identity oracle. Instead train adapter to improve
            # retrieval when query is noisy: query = obs + noise, gallery = obs.
            kos = [k for k in split["train"] + split["val"] + split["test"] if k in gwps_gal]
            kos = list(dict.fromkeys(kos))
            mat = np.stack([np.nan_to_num(gwps_gal[k], nan=0.0) for k in kos], axis=0)
            Ytr = np.stack([np.nan_to_num(gwps_gal[k], nan=0.0) for k in split["train"] if k in gwps_gal])
            tr_kos = [k for k in split["train"] if k in gwps_gal]
            Yva = np.stack([np.nan_to_num(gwps_gal[k], nan=0.0) for k in split["val"] if k in gwps_gal])
            va_kos = [k for k in split["val"] if k in gwps_gal]
            # Add noise to queries during training so adapter isn't identity map
            pca_g = PCAProjector(n_components=min(256, Ytr.shape[0] - 1)).fit(Ytr)
            _log(f"  Training GWPS noisy-query adapter catalog={len(kos)} ...")

            def _noisy(Y, rng, std=0.5):
                return Y + rng.normal(0, std, size=Y.shape).astype(np.float32)

            rng = np.random.default_rng(0)
            # Custom short train loop with noisy queries
            adapter_g = QueryAdapter(pca_g, hidden=512).to(device)
            opt = torch.optim.AdamW(adapter_g.parameters(), lr=1e-3, weight_decay=1e-4)
            g_index = {k: i for i, k in enumerate(kos)}
            tr_lab = np.array([g_index[k] for k in tr_kos], dtype=np.int64)
            gallery_t = torch.tensor(mat, dtype=torch.float32, device=device)
            best_mrr, best_state = -1.0, None
            for ep in range(1, 26):
                adapter_g.train()
                # resample noise each epoch
                Ytr_n = _noisy(Ytr, rng, std=0.35)
                ds = TensorDataset(
                    torch.tensor(Ytr_n, dtype=torch.float32),
                    torch.tensor(tr_lab, dtype=torch.long),
                )
                loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
                losses = []
                for yb, lb in loader:
                    yb, lb = yb.to(device), lb.to(device)
                    logits = pearson_scores_torch(adapter_g(yb), gallery_t) * 20.0
                    loss = F.cross_entropy(logits, lb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    losses.append(float(loss.item()))
                # val: noisy queries
                Yva_n = _noisy(Yva, rng, std=0.35)
                r = eval_part_ranks(adapter_g, Yva_n, va_kos, kos, mat, device)
                mrr = float(np.mean(1.0 / r))
                _log(f"  GWPS ep {ep:02d} loss={np.mean(losses):.3f} val_MRR={mrr:.4f} med={np.median(r):.0f}")
                if mrr > best_mrr:
                    best_mrr, best_state = mrr, {k: v.detach().cpu().clone() for k, v in adapter_g.state_dict().items()}
            if best_state is not None:
                adapter_g.load_state_dict(best_state)
                gwps_adapter = adapter_g
                df_a = score_dualarm_adapter(gwps_adapter, query, gwps_genes, gwps_gal, device)
                df_a.to_csv(qdir / "adapter_dualarm.tsv", sep="\t", index=False)
                for method, df in [("pearson", df_p), ("adapter", df_a)]:
                    for g in cml_focus:
                        hit = df[df.ko == g]
                        row = {"tag": "gse120932_k562_ir_with_drug", "method": method, "gene": g, "in_gallery": not hit.empty}
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
                        focus_rows.append(row)
                sub = pd.DataFrame([r for r in focus_rows if r["tag"] == "gse120932_k562_ir_with_drug"])
                _log(sub.pivot_table(index="gene", columns="method", values="rank_pref", aggfunc="first").to_string())
            else:
                for g in cml_focus:
                    hit = df_p[df_p.ko == g]
                    row = {"tag": "gse120932_k562_ir_with_drug", "method": "pearson", "gene": g, "in_gallery": not hit.empty}
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
                    focus_rows.append(row)
        else:
            _log("  No GWPS split file; CML pearson-only")
            for g in cml_focus:
                hit = df_p[df_p.ko == g]
                row = {"tag": "gse120932_k562_ir_with_drug", "method": "pearson", "gene": g, "in_gallery": not hit.empty}
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
                focus_rows.append(row)
    else:
        _log("  GWPS split missing; CML pearson-only")
        for g in cml_focus:
            hit = df_p[df_p.ko == g]
            row = {"tag": "gse120932_k562_ir_with_drug", "method": "pearson", "gene": g, "in_gallery": not hit.empty}
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
            focus_rows.append(row)

    foc = pd.DataFrame(focus_rows)
    foc.to_csv(out / "resistance_focus_compare.tsv", sep="\t", index=False)
    wide = foc[foc.in_gallery].pivot_table(
        index=["tag", "gene"], columns="method", values="rank_pref", aggfunc="first"
    )
    wide.to_csv(out / "resistance_pref_rank_wide.tsv", sep="\t")
    _log("\n=== Resistance preferred-rank wide ===")
    _log(wide.to_string())

    summary = {
        "identity_test": id_tab.to_dict(orient="records"),
        "train_info": {k: train_info[k] for k in ("best_epoch", "best_val_mrr")},
        "note": "G1 = learnable query adapter then Pearson vs predicted/observed gallery",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "README.md").write_text(
        "# G1 Gallery Query Adapter\n\n"
        "Evolve reverse retrieval **on the expression gallery**:\n"
        "`s(g)=corr(Adapter(ΔY*), ΔŶ(g))` vs plain Pearson.\n",
        encoding="utf-8",
    )
    _log(f"\nWrote {out}")


if __name__ == "__main__":
    main()
