#!/usr/bin/env python3
"""scGPT-style reverse: scGPT forward gallery + pearson, vs dual/GEARS/linear."""

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
    load_ctrl_from_perturb_processed,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_reverse_bundle, load_split_kos  # noqa: E402
from reverse.src.reverse_model import PCAProjector, ReverseDualEncoder  # noqa: E402
from reverse.src.run_gears_gallery_recovery import (  # noqa: E402
    align_gallery_to_genes,
    build_delta_gallery,
    pearson_ranks,
)
from reverse.src.train_reverse_retrieval import (  # noqa: E402
    DEFAULTS,
    dual_encoder_ranks,
    load_gene_prior_matrix,
)

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
SCGPT_DIR = _ROOT / "reverse/results/scgpt_hepg2_seed1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scgpt_dir", type=str, default=str(SCGPT_DIR))
    ap.add_argument("--gears_dir", type=str, default=str(GEARS_DIR))
    ap.add_argument("--linear_dir", type=str, default=str(LINEAR_DIR))
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(_ROOT / "reverse/results/retrieval_hepg2_v2/best.pt"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_ROOT / "reverse/results/scgpt_gallery_hepg2"),
    )
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_h5ad = Path(DEFAULTS["dataset_h5ad"])
    split = load_split_kos(Path(DEFAULTS["split"]))
    test_kos = split["test"]

    scgpt_genes, scgpt_gal = build_delta_gallery(Path(args.scgpt_dir), dataset_h5ad)
    _, obs = load_observed_deltas(Path(DEFAULTS["pseudobulk_deltas"]), scgpt_genes)

    gears_genes, gears_gal_raw = build_delta_gallery(Path(args.gears_dir), dataset_h5ad)
    gears_gal = align_gallery_to_genes(gears_gal_raw, gears_genes, scgpt_genes)

    lin_genes, lin_gal_raw = build_delta_gallery(Path(args.linear_dir), dataset_h5ad)
    lin_gal = align_gallery_to_genes(lin_gal_raw, lin_genes, scgpt_genes)

    catalog = sorted(set(obs) & set(scgpt_gal) & set(gears_gal) & set(lin_gal))
    queries = [k for k in test_kos if k in catalog]
    print(
        json.dumps(
            {
                "n_scgpt_gallery": len(scgpt_gal),
                "n_catalog": len(catalog),
                "n_test_queries": len(queries),
            }
        )
    )

    Q = np.stack([obs[k] for k in queries], axis=0)
    frames = [
        pearson_ranks(Q, queries, scgpt_gal, catalog, "scgpt_gallery_pearson"),
        pearson_ranks(Q, queries, gears_gal, catalog, "gears_gallery_pearson"),
        pearson_ranks(Q, queries, lin_gal, catalog, "linear_L3_gallery_pearson"),
    ]

    device = torch.device(
        "cuda"
        if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
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
    frames.append(
        dual_encoder_ranks(
            model,
            pca.transform(Yq),
            q2,
            cat_kos,
            cat_P,
            cat_G,
            device,
            method="dual_encoder_v2",
        )
    )

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "ranks.tsv", sep="\t", index=False)
    summaries = []
    for m, sub in all_df.groupby("method"):
        s = summarize_recovery(sub)
        s["method"] = m
        # note dual may have fewer queries
        s["n"] = int(len(sub))
        summaries.append(s)
    sum_df = pd.DataFrame(summaries).sort_values("median_rank")
    sum_df.to_csv(out / "summary.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(sum_df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
