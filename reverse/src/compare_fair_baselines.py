#!/usr/bin/env python3
"""Fair held-out KO recovery baselines vs dual-encoder v2 (parameterized by cell line).

Prior work mapping
------------------
- CMap / RGES / GenePerturbR-style: match a query signature to a *perturbation gallery*
  → here: pearson / CMap-lite against predicted (or observed) ΔY galleries.
- CRISPR-GEM: scores *simulated edits* for phenotype shift; not KO-ID recovery.
  We include a GEM-lite proxy (see notes), not a full reimplementation.
- GEARS / Scouter: *forward* predictors only; enter as predicted-gallery builders.
- Dual-encoder v2: trained reverse retrieval (this repo).

All methods below use GEARS seed-1 test KOs as queries and the same catalog size.
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

from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_reverse_bundle  # noqa: E402
from reverse.src.reverse_model import PCAProjector, ReverseDualEncoder  # noqa: E402
from reverse.src.train_reverse_retrieval import (  # noqa: E402
    dual_encoder_ranks,
    load_gene_prior_matrix,
)


def _zrows(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def ranks_from_scores(scores: np.ndarray, query_kos: list[str], gallery_kos: list[str], method: str) -> pd.DataFrame:
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


def pearson_gallery_ranks(query_Y: np.ndarray, query_kos: list[str], gallery: dict[str, np.ndarray], gallery_kos: list[str], method: str) -> pd.DataFrame:
    g_mat = np.column_stack([gallery[k] for k in gallery_kos]).T  # n_g x genes
    q = _zrows(query_Y)
    g = _zrows(g_mat)
    scores = (q @ g.T) / q.shape[1]
    return ranks_from_scores(scores, query_kos, gallery_kos, method)


def cmap_lite_ranks(query_Y: np.ndarray, query_kos: list[str], gallery: dict[str, np.ndarray], gallery_kos: list[str], top_n: int = 100) -> pd.DataFrame:
    """CMap-like: up/down gene sets from query, score each gallery profile by mean rank enrichment."""
    g_mat = np.column_stack([gallery[k] for k in gallery_kos])  # genes x G
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
        s_up = -ranks[up, :].mean(axis=0)
        s_down = ranks[down, :].mean(axis=0)
        scores = s_up + s_down
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
                "method": "cmap_lite_pred_gallery",
            }
        )
    return pd.DataFrame(out_rows)


def ridge_delta_to_p_ranks(
    Ytr: np.ndarray,
    Ptr: np.ndarray,
    Yte: np.ndarray,
    query_kos: list[str],
    cat_kos: list[str],
    cat_P: np.ndarray,
) -> pd.DataFrame:
    reg = Ridge(alpha=1.0, random_state=0)
    reg.fit(np.nan_to_num(Ytr, nan=0.0), np.nan_to_num(Ptr, nan=0.0))
    Phat = reg.predict(np.nan_to_num(Yte, nan=0.0))
    Phat = _zrows(Phat)
    Pcat = _zrows(np.nan_to_num(cat_P, nan=0.0))
    scores = Phat @ Pcat.T / Phat.shape[1]
    return ranks_from_scores(scores, query_kos, cat_kos, "ridge_delta_to_P")


def esm_signature_ranks(
    query_Y: np.ndarray,
    query_kos: list[str],
    cat_kos: list[str],
    cat_G: np.ndarray,
    gene_names: list[str],
    esm_by_expr_gene: dict[str, np.ndarray],
    top_n: int = 50,
) -> pd.DataFrame:
    rows_scores = []
    G = _zrows(cat_G)
    for i in range(len(query_kos)):
        q = np.nan_to_num(query_Y[i], nan=0.0)
        top = np.argsort(-np.abs(q))[:top_n]
        vecs = []
        for j in top:
            gname = gene_names[j]
            if gname in esm_by_expr_gene:
                vecs.append(esm_by_expr_gene[gname])
        if not vecs:
            rows_scores.append(np.zeros(len(cat_kos)))
            continue
        qv = np.mean(np.stack(vecs, axis=0), axis=0)
        qv = (qv - qv.mean()) / (qv.std() + 1e-12)
        rows_scores.append(G @ qv / max(len(qv), 1))
    scores = np.stack(rows_scores, axis=0)
    return ranks_from_scores(scores, query_kos, cat_kos, "esm_mean_topDEG_to_KO")


def load_esm_by_symbol(esm_tsv: Path, out_dim: int = 64) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA

    df = pd.read_csv(esm_tsv, sep="\t")
    genes = [str(c) for c in df.columns]
    mat = df.to_numpy(dtype=np.float32).T
    k = min(out_dim, mat.shape[0] - 1, mat.shape[1])
    z = PCA(n_components=k, random_state=0).fit_transform(mat).astype(np.float32)
    return {g: z[i] for i, g in enumerate(genes)}


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
        sub = (sub - sub.mean(axis=1, keepdims=True)) / (sub.std(axis=1, keepdims=True) + 1e-12)
        scores[i] = sub @ qq / len(idx)
    return ranks_from_scores(scores, query_kos, gallery_kos, "gem_lite_topDEG_corr")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_line", type=str, default="hepg2", choices=list_cells())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", type=str, default=None, help="Default: registry retrieval out / best.pt")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    paths = resolve_cell_paths(args.cell_line, seed=args.seed)
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(paths["retrieval_out"]) / "best.pt"
    out = Path(args.out_dir) if args.out_dir else Path(paths["fair_out"])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    genes, tables, meta = load_reverse_bundle(
        Path(paths["pseudobulk_deltas"]),
        Path(paths["pred_dir"]) / "gene_names.json",
        Path(paths["split"]),
        Path(paths["p_tsv"]),
    )
    cat_kos, cat_P = [], []
    for part in ("train", "val", "test"):
        cat_kos.extend(tables[part]["kos"])
        cat_P.append(tables[part]["P"])
    cat_P = np.vstack(cat_P)
    cat_G = load_gene_prior_matrix(Path(paths["esm_tsv"]), cat_kos, out_dim=64)

    Ytr, Ptr = tables["train"]["Y"], tables["train"]["P"]
    Yte = tables["test"]["Y"]
    kte = tables["test"]["kos"]

    genes2, pred_abs = load_prediction_dir(Path(paths["pred_dir"]))
    assert genes2 == genes
    ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
    pred_gal = absolute_to_delta(pred_abs, ctrl)
    g_kos = [k for k in cat_kos if k in pred_gal]
    keep = [i for i, k in enumerate(kte) if k in pred_gal]
    kte = [kte[i] for i in keep]
    Yte = Yte[keep]

    esm_map = load_esm_by_symbol(Path(paths["esm_tsv"]), out_dim=64)

    frames = []
    frames.append(pearson_gallery_ranks(Yte, kte, pred_gal, g_kos, "pearson_pred_gallery"))
    frames.append(cmap_lite_ranks(Yte, kte, pred_gal, g_kos, top_n=100))
    frames.append(gem_lite_ranks(Yte, kte, pred_gal, g_kos, top_deg=200))
    frames.append(ridge_delta_to_p_ranks(Ytr, Ptr, Yte, kte, cat_kos, cat_P))
    frames.append(esm_signature_ranks(Yte, kte, cat_kos, cat_G, genes, esm_map, top_n=50))

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing dual-encoder checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pca = PCAProjector()
    pca.load_state_dict(ckpt["pca"])
    Yte_pca = pca.transform(Yte)
    model = ReverseDualEncoder(
        delta_in_dim=pca.n_components,
        p_dim=cat_P.shape[1],
        gene_dim=cat_G.shape[1],
        emb_dim=ckpt["args"].get("emb_dim", 128),
        hidden=ckpt["args"].get("hidden", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    frames.append(
        dual_encoder_ranks(model, Yte_pca, kte, cat_kos, cat_P, cat_G, device, method="dual_encoder_v2")
    )

    obs_gal = {}
    for part in ("train", "val", "test"):
        for k, y in zip(tables[part]["kos"], tables[part]["Y"]):
            obs_gal[k] = y
    frames.append(
        pearson_gallery_ranks(
            Yte, kte, obs_gal, [k for k in cat_kos if k in obs_gal], "pearson_obs_gallery_oracle"
        )
    )
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "all_method_ranks.tsv", sep="\t", index=False)

    summaries = []
    for m, sub in all_df.groupby("method"):
        s = summarize_recovery(sub)
        s["method"] = m
        s["mrr"] = float((1.0 / sub["rank"]).mean())
        summaries.append(s)
    sum_df = pd.DataFrame(summaries).sort_values("median_rank")
    sum_df.to_csv(out / "fair_compare_summary.tsv", sep="\t", index=False)
    (out / "fair_compare_summary.json").write_text(json.dumps(summaries, indent=2))
    (out / "protocol_notes.json").write_text(
        json.dumps(
            {
                "task": "held-out KO identity recovery from observed ΔY query",
                "cell_line": paths["label"],
                "seed": int(paths["seed"]),
                "split": f"GEARS seed{paths['seed']} {paths['label']} simulation",
                "n_test": len(kte),
                "catalog": len(cat_kos),
                "ckpt": str(ckpt_path),
                "comparable_methods": [
                    "pearson_pred_gallery (~CMap/GenePerturbR with forward-model gallery)",
                    "cmap_lite_pred_gallery (up/down set enrichment)",
                    "gem_lite_topDEG_corr (GEM-inspired DEG subspace corr; not full GEM)",
                    "ridge_delta_to_P (linear invert to atlas P)",
                    "dual_encoder_v2 (this work)",
                ],
                "not_apples_to_apples": [
                    "CRISPR-GEM full (phenotype-shift MLP; different objective)",
                    "GEARS/Scouter alone (forward only)",
                    "pearson_obs_gallery_oracle (uses observed test profiles in gallery)",
                ],
                "meta": meta,
            },
            indent=2,
        )
    )
    print(sum_df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
