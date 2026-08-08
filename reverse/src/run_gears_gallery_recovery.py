#!/usr/bin/env python3
"""scGPT-style reverse: GEARS forward gallery + pearson retrieval (HepG2 held-out)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    clean_ko,
    load_ctrl_from_perturb_processed,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_split_kos  # noqa: E402
from reverse.src.reverse_model import PCAProjector, ReverseDualEncoder  # noqa: E402
from reverse.src.train_reverse_retrieval import (  # noqa: E402
    DEFAULTS,
    dual_encoder_ranks,
    load_gene_prior_matrix,
)
from reverse.src.reverse_data import load_reverse_bundle  # noqa: E402

BENCH = _ROOT / "linear_perturbation_prediction-Paper-main" / "benchmark"
GEARS_DIR = (
    BENCH
    / "working_dir/results/matched_ml_baselines/replogle_hepg2_essential__gears_seed1"
)
LINEAR_DIR = (
    BENCH
    / "working_dir/results/progressive_stack_fulltest"
    / "replogle_hepg2_essential__prog_L3_k562_rpe1_jurkat"
)


def align_gallery_to_genes(
    gallery: dict[str, np.ndarray], src_genes: list[str], dst_genes: list[str]
) -> dict[str, np.ndarray]:
    idx = {g: i for i, g in enumerate(src_genes)}
    out = {}
    for ko, vec in gallery.items():
        out[ko] = np.array(
            [vec[idx[g]] if g in idx else np.nan for g in dst_genes], dtype=float
        )
    return out


def pearson_ranks(
    query_Y: np.ndarray,
    query_kos: list[str],
    gallery: dict[str, np.ndarray],
    gallery_kos: list[str],
    method: str,
) -> pd.DataFrame:
    """Vectorized Pearson ranks; genes with any NaN across queries/gallery dropped."""
    g_mat = np.stack([gallery[k] for k in gallery_kos], axis=0)  # G x genes
    # shared finite genes
    ok = np.isfinite(query_Y).all(axis=0) & np.isfinite(g_mat).all(axis=0)
    q = query_Y[:, ok]
    g = g_mat[:, ok]
    q = q - q.mean(axis=1, keepdims=True)
    g = g - g.mean(axis=1, keepdims=True)
    qsd = q.std(axis=1, keepdims=True)
    gsd = g.std(axis=1, keepdims=True)
    qsd[qsd < 1e-12] = 1.0
    gsd[gsd < 1e-12] = 1.0
    q = q / qsd
    g = g / gsd
    scores = (q @ g.T) / q.shape[1]

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


def build_delta_gallery(pred_dir: Path, dataset_h5ad: Path):
    genes, pred_abs = load_prediction_dir(pred_dir)
    ctrl = load_ctrl_from_perturb_processed(dataset_h5ad, genes)
    return genes, absolute_to_delta(pred_abs, ctrl)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears_dir", type=str, default=str(GEARS_DIR))
    ap.add_argument("--linear_dir", type=str, default=str(LINEAR_DIR))
    ap.add_argument("--ckpt", type=str, default=str(_ROOT / "reverse/results/retrieval_hepg2_v2/best.pt"))
    ap.add_argument("--out_dir", type=str, default=str(_ROOT / "reverse/results/gears_gallery_hepg2"))
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_h5ad = Path(DEFAULTS["dataset_h5ad"])
    split = load_split_kos(Path(DEFAULTS["split"]))
    test_kos = split["test"]

    # Observed ΔY on GEARS gene axis
    gears_genes, gears_gal = build_delta_gallery(Path(args.gears_dir), dataset_h5ad)
    _, obs = load_observed_deltas(Path(DEFAULTS["pseudobulk_deltas"]), gears_genes)

    # Linear gallery aligned to GEARS genes
    lin_genes, lin_gal_raw = build_delta_gallery(Path(args.linear_dir), dataset_h5ad)
    lin_gal = align_gallery_to_genes(lin_gal_raw, lin_genes, gears_genes)

    # Catalog = intersection of obs, gears, linear, and preferably all test queries
    catalog = sorted(set(obs) & set(gears_gal) & set(lin_gal))
    queries = [k for k in test_kos if k in catalog]
    print(json.dumps({"n_catalog": len(catalog), "n_test_queries": len(queries)}))

    Q = np.stack([obs[k] for k in queries], axis=0)
    frames = []
    frames.append(pearson_ranks(Q, queries, gears_gal, catalog, "gears_gallery_pearson"))
    frames.append(pearson_ranks(Q, queries, lin_gal, catalog, "linear_L3_gallery_pearson"))

    # Dual v2 on same queries (uses its own gene/P space from training bundle)
    device = torch.device("cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    genes_b, tables, meta = load_reverse_bundle(
        Path(DEFAULTS["pseudobulk_deltas"]),
        Path(DEFAULTS["pred_dir"]) / "gene_names.json",
        Path(DEFAULTS["split"]),
        Path(DEFAULTS["p_tsv"]),
    )
    cat_kos, cat_P = [], []
    for part in ("train", "val", "test"):
        cat_kos.extend(tables[part]["kos"])
        cat_P.append(tables[part]["P"])
    cat_P = np.vstack(cat_P)
    cat_G = load_gene_prior_matrix(Path(DEFAULTS["esm_tsv"]), cat_kos, out_dim=64)
    # map dual queries: observed Y in bundle gene order
    obs_bundle = {}
    for part in ("train", "val", "test"):
        for k, y in zip(tables[part]["kos"], tables[part]["Y"]):
            obs_bundle[k] = y
    q2 = [k for k in queries if k in obs_bundle and k in set(cat_kos)]
    Yq = np.stack([obs_bundle[k] for k in q2], axis=0)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    pca = PCAProjector()
    pca.load_state_dict(ckpt["pca"])
    model = ReverseDualEncoder(
        delta_in_dim=pca.n_components,
        p_dim=cat_P.shape[1],
        gene_dim=cat_G.shape[1],
        emb_dim=ckpt["args"].get("emb_dim", 128),
        hidden=ckpt["args"].get("hidden", 256),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    df_dual = dual_encoder_ranks(
        model, pca.transform(Yq), q2, cat_kos, cat_P, cat_G, device, method="dual_encoder_v2"
    )
    frames.append(df_dual)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "ranks.tsv", sep="\t", index=False)
    summaries = []
    for m, sub in all_df.groupby("method"):
        s = summarize_recovery(sub)
        s["method"] = m
        summaries.append(s)
    sum_df = pd.DataFrame(summaries).sort_values("median_rank")
    sum_df.to_csv(out / "summary.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(sum_df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
