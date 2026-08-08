#!/usr/bin/env python3
"""Phase-2 disease / state signature helpers: demo proxy, score, gene-set enrichment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.benchmarks import random_null_max_score, sign_flip_scores  # noqa: E402
from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402
from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv, save_delta_y_star  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.score import score_gallery, top_k  # noqa: E402

# Small curated LIHC / liver-cancer related genes present in many Essential panels (demo only)
_LIHC_PROXY_KOS = [
    "MYC",
    "CTNNB1",
    "TP53",
    "AXIN1",
    "ARID1A",
    "KEAP1",
    "NFE2L2",
    "TERT",
    "MET",
    "CCND1",
]


def _gallery(cell: str):
    paths = resolve_cell_paths(cell)
    genes, pred_abs = load_prediction_dir(paths["pred_dir"])
    ctrl = load_ctrl_from_perturb_processed(paths["dataset_h5ad"], genes)
    gal = absolute_to_delta(pred_abs, ctrl)
    return paths, genes, gal


def cmd_demo_lihc_proxy(args: argparse.Namespace) -> None:
    """Build synthetic ΔY* = mean observed ΔY of available LIHC-related KOs (smoke test)."""
    paths, genes, gal = _gallery(args.cell_line)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, obs = load_observed_deltas(paths["pseudobulk_deltas"], genes)
    used = [k for k in _LIHC_PROXY_KOS if k in obs]
    if len(used) < 3:
        # fallback: top-variance train-like KOs alphabetically present
        used = sorted(obs.keys())[:10]
    mat = np.column_stack([obs[k] for k in used])
    star = np.nanmean(mat, axis=1)
    star_series = pd.Series(star, index=genes, dtype=float)
    save_delta_y_star(out / "delta_y_star.tsv", star_series)
    (out / "proxy_source_kos.json").write_text(json.dumps({"used_kos": used, "note": "synthetic mean obs ΔY"}, indent=2))

    scored = score_gallery(gal, star, genes, metric=args.metric)
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=args.top_k, covered_only=False).to_csv(out / f"top{args.top_k}.tsv", sep="\t", index=False)

    # vs DEG: top-|Δ| genes in the signature itself
    abs_star = np.abs(star)
    deg_idx = np.argsort(-abs_star)[: args.top_k]
    deg = pd.DataFrame({"gene": [genes[i] for i in deg_idx], "abs_delta": abs_star[deg_idx], "rank_deg": np.arange(1, len(deg_idx) + 1)})
    deg.to_csv(out / "deg_top_from_signature.tsv", sep="\t", index=False)
    top_rev = set(top_k(scored, k=args.top_k, covered_only=False)["ko"])
    top_deg = set(deg["gene"])
    overlap = sorted(top_rev & top_deg)
    (out / "vs_deg_overlap.json").write_text(
        json.dumps(
            {
                "top_k": args.top_k,
                "n_overlap": len(overlap),
                "jaccard": len(overlap) / max(len(top_rev | top_deg), 1),
                "overlap_genes": overlap,
            },
            indent=2,
        )
    )

    flip = sign_flip_scores(gal, star, genes, metric=args.metric)
    flip.to_csv(out / "reverse_scores_signflip.tsv", sep="\t", index=False)
    if args.n_random > 0:
        null = random_null_max_score(gal, star, genes, n_rand=args.n_random, metric=args.metric, seed=0)
        null.to_csv(out / "random_null_max_score.tsv", sep="\t", index=False)

    print(f"demo proxy used KOs={used}")
    print(f"Wrote {out}")


def cmd_score(args: argparse.Namespace) -> None:
    paths, genes, gal = _gallery(args.cell_line)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    star_series = load_vector_tsv(Path(args.delta_y_star))
    star = align_delta_y_star(star_series, genes)
    save_delta_y_star(out / "delta_y_star.aligned.tsv", pd.Series(star, index=genes, dtype=float))
    scored = score_gallery(gal, star, genes, metric=args.metric)
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=args.top_k, covered_only=False).to_csv(out / f"top{args.top_k}.tsv", sep="\t", index=False)
    flip = sign_flip_scores(gal, star, genes, metric=args.metric)
    flip.to_csv(out / "reverse_scores_signflip.tsv", sep="\t", index=False)
    print(f"Wrote {out}")


def cmd_enrich(args: argparse.Namespace) -> None:
    """Hypergeometric enrichment of top reverse hits in provided gene sets (one gene per line)."""
    from scipy.stats import hypergeom

    ranks = pd.read_csv(args.ranks, sep="\t")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    top = set(ranks.nsmallest(args.top_k, "rank")["ko"].astype(str))
    background = set(ranks["ko"].astype(str))
    set_path = Path(args.gene_sets)
    if set_path.is_dir():
        files = sorted(set_path.glob("*.txt"))
    else:
        files = [set_path]
    rows = []
    M = len(background)
    n = len(top)
    for f in files:
        geneset = {ln.strip() for ln in f.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
        K = len(geneset & background)
        x = len(geneset & top)
        # P(X >= x)
        p = float(hypergeom.sf(x - 1, M, K, n)) if K and n else 1.0
        rows.append(
            {
                "gene_set": f.stem,
                "n_set_in_background": K,
                "n_top": n,
                "n_overlap": x,
                "overlap": ",".join(sorted(geneset & top)),
                "hypergeom_pvalue": p,
            }
        )
    edf = pd.DataFrame(rows).sort_values("hypergeom_pvalue")
    edf.to_csv(out / "gene_set_enrichment.tsv", sep="\t", index=False)
    print(edf.to_string(index=False))
    print(f"Wrote {out / 'gene_set_enrichment.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-2 disease signature reverse scoring")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("demo_lihc_proxy", help="Synthetic LIHC-like ΔY* from HepG2 KO means")
    p1.add_argument("--cell_line", default="hepg2", choices=list_cells())
    p1.add_argument("--out_dir", default=str(_ROOT / "reverse/results/disease_lihc_proxy"))
    p1.add_argument("--metric", default="pearson")
    p1.add_argument("--top_k", type=int, default=50)
    p1.add_argument("--n_random", type=int, default=50)
    p1.set_defaults(func=cmd_demo_lihc_proxy)

    p2 = sub.add_parser("score", help="Score a real delta_y_star.tsv")
    p2.add_argument("--delta_y_star", required=True)
    p2.add_argument("--cell_line", default="hepg2", choices=list_cells())
    p2.add_argument("--out_dir", default=str(_ROOT / "reverse/results/disease_custom"))
    p2.add_argument("--metric", default="pearson")
    p2.add_argument("--top_k", type=int, default=50)
    p2.set_defaults(func=cmd_score)

    p3 = sub.add_parser("enrich", help="Hypergeometric enrichment vs gene-set TXT files")
    p3.add_argument("--ranks", required=True, help="reverse_scores.tsv")
    p3.add_argument("--gene_sets", required=True, help="TXT file or directory of TXT gene lists")
    p3.add_argument("--out_dir", required=True)
    p3.add_argument("--top_k", type=int, default=50)
    p3.set_defaults(func=cmd_enrich)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
