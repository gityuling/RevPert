#!/usr/bin/env python3
"""K562 GWPS reverse helpers: split, Task-1 oracle, disease scoring on large bulk KO gallery.

Data: replogle_k562_gwps/all_pseudobulk_deltas.h5ad (~9866 KOs, bulk/pseudobulk).
Claim boundary: GWPS is K562 (not hepatocyte); LIHC scoring is cross-lineage usage /
catalog expansion, not a liver-matched gallery replacement for HepG2 Essential.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv  # noqa: E402
from reverse.src.io_gallery import clean_ko  # noqa: E402
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.score import score_gallery, top_k  # noqa: E402

BENCH = _ROOT / "linear_perturbation_prediction-Paper-main" / "benchmark"
GWPS_DELTAS = BENCH / "data/gears_pert_data/replogle_k562_gwps/all_pseudobulk_deltas.h5ad"
SPLIT_PATH = BENCH / "working_dir/results/seed_1_replogle_k562_gwps_split"
DEFAULT_OUT = _ROOT / "reverse/results/gwps_k562"


def load_gwps_deltas(path: Path = GWPS_DELTAS) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return (gene_axis, {ko: delta_vector})."""
    adata = ad.read_h5ad(path)
    genes = list(map(str, adata.var_names))
    gal: dict[str, np.ndarray] = {}
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    for i, name in enumerate(adata.obs_names.astype(str)):
        ko = clean_ko(name)
        if "perturbed_gene" in adata.obs.columns:
            ko = clean_ko(str(adata.obs["perturbed_gene"].iloc[i]))
        if ko.lower() in {"ctrl", "control", "non-targeting", "nt"}:
            continue
        gal[ko] = X[i].astype(float)
    return genes, gal


def make_split(
    *,
    seed: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    out_path: Path = SPLIT_PATH,
) -> dict:
    """GEARS-like JSON split over GWPS KO names (condition style: GENE+ctrl)."""
    genes, gal = load_gwps_deltas()
    kos = sorted(gal)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(kos))
    rng.shuffle(idx)
    n = len(kos)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = [f"{kos[i]}+ctrl" for i in idx[:n_train]]
    val = [f"{kos[i]}+ctrl" for i in idx[n_train : n_train + n_val]]
    test = [f"{kos[i]}+ctrl" for i in idx[n_train + n_val :]]
    split = {
        "train": train,
        "val": val,
        "test": test,
        "dataset": "replogle_k562_gwps",
        "seed": seed,
        "n_genes_axis": len(genes),
        "note": "Random split for reverse GWPS pilots; not an official GEARS simulation split.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split) + "\n")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in split.items()}, indent=2))
    print(f"Wrote {out_path}")
    return split


def _zrows(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def task1_oracle(*, split_path: Path = SPLIT_PATH, out_dir: Path = DEFAULT_OUT / "task1_oracle") -> pd.DataFrame:
    """Observed-gallery Pearson on held-out test (SI upper bound; gallery includes test obs)."""
    split = json.loads(Path(split_path).read_text())
    genes, gal = load_gwps_deltas()
    test_kos = [clean_ko(c) for c in split["test"] if clean_ko(c) in gal]
    gallery_kos = sorted(gal)
    q = np.vstack([gal[k] for k in test_kos])
    g = np.vstack([gal[k] for k in gallery_kos])
    scores = (_zrows(q) @ _zrows(g).T) / q.shape[1]
    g_index = {k: i for i, k in enumerate(gallery_kos)}
    rows = []
    for i, ko in enumerate(test_kos):
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
                "method": "pearson_obs_gallery_oracle",
            }
        )
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ranks.tsv", sep="\t", index=False)
    summary = summarize_recovery(df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta = {
        "n_test": len(test_kos),
        "n_gallery": len(gallery_kos),
        "n_genes": len(genes),
        "claim": "oracle upper bound; observed gallery includes test KOs",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"summary": summary, **meta}, indent=2))
    return df


def score_disease(
    *,
    delta_y_star: Path,
    out_dir: Path,
    genes_file: Path | None = None,
    top_k_n: int = 50,
    metric: str = "pearson",
) -> pd.DataFrame:
    """Score disease ΔY* against full GWPS observed ΔY gallery."""
    genes, gal = load_gwps_deltas()
    star = align_delta_y_star(load_vector_tsv(Path(delta_y_star)), genes)
    scored = score_gallery(gal, star, genes, metric=metric)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=top_k_n).to_csv(out_dir / f"top{top_k_n}.tsv", sep="\t", index=False)

    meta = {
        "gallery": "k562_gwps_observed_deltas",
        "n_gallery": int(len(gal)),
        "n_genes": len(genes),
        "delta_y_star": str(delta_y_star),
        "claim_boundary": (
            "K562 GWPS bulk KO gallery × LIHC signature = cross-lineage catalog expansion; "
            "not hepatocyte-matched; not driver discovery."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if genes_file and Path(genes_file).is_file():
        focus = []
        for line in Path(genes_file).read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                focus.append(s.split("\t")[0])
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
        focus_df = pd.DataFrame(rows)
        focus_df.to_csv(out_dir / "focus_gene_ranks.tsv", sep="\t", index=False)
        print(focus_df.sort_values("rank", na_position="last").to_string(index=False))
    print(f"Wrote {out_dir} gallery={len(gal)}")
    return scored


def compare_to_hepg2(
    *,
    gwps_scores: Path,
    hepg2_scores: Path,
    out_dir: Path,
    genes_file: Path | None = None,
) -> dict:
    """Spearman on shared KOs + focus-gene rank table."""
    g = pd.read_csv(gwps_scores, sep="\t").set_index("ko")
    h = pd.read_csv(hepg2_scores, sep="\t").set_index("ko")
    common = sorted(set(g.index) & set(h.index))
    rho, p = spearmanr(g.loc[common, "score"], h.loc[common, "score"])
    out = {
        "n_common_kos": len(common),
        "n_gwps_only": int(len(set(g.index) - set(h.index))),
        "n_hepg2_only": int(len(set(h.index) - set(g.index))),
        "spearman_score_gwps_vs_hepg2": float(rho),
        "spearman_pvalue": float(p),
        "gwps_top10": list(g.sort_values("rank").head(10).index),
        "hepg2_top10": list(h.sort_values("rank").head(10).index),
        "top50_overlap": int(len(set(g.sort_values("rank").head(50).index) & set(h.sort_values("rank").head(50).index))),
    }
    if genes_file and Path(genes_file).is_file():
        focus = [
            line.strip().split("\t")[0]
            for line in Path(genes_file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        rows = []
        for gene in focus:
            rows.append(
                {
                    "ko": gene,
                    "rank_gwps": int(g.loc[gene, "rank"]) if gene in g.index else None,
                    "rank_hepg2": int(h.loc[gene, "rank"]) if gene in h.index else None,
                    "in_gwps": gene in g.index,
                    "in_hepg2": gene in h.index,
                }
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "focus_ranks_gwps_vs_hepg2.tsv", sep="\t", index=False)
        out["focus_in_gwps"] = sum(1 for r in rows if r["in_gwps"])
        out["focus_in_hepg2"] = sum(1 for r in rows if r["in_hepg2"])
        out["focus_in_gwps_top200"] = sum(
            1 for r in rows if r["rank_gwps"] is not None and r["rank_gwps"] <= 200
        )
        out["focus_in_hepg2_top200"] = sum(
            1 for r in rows if r["rank_hepg2"] is not None and r["rank_hepg2"] <= 200
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="K562 GWPS reverse pilots")
    sub = p.add_subparsers(dest="cmd", required=True)

    s0 = sub.add_parser("make_split")
    s0.add_argument("--seed", type=int, default=1)
    s0.add_argument("--out", default=str(SPLIT_PATH))

    s1 = sub.add_parser("task1_oracle")
    s1.add_argument("--split", default=str(SPLIT_PATH))
    s1.add_argument("--out_dir", default=str(DEFAULT_OUT / "task1_oracle"))

    s2 = sub.add_parser("score_disease")
    s2.add_argument("--delta_y_star", required=True)
    s2.add_argument("--out_dir", required=True)
    s2.add_argument("--genes_file", default=None)
    s2.add_argument("--top_k", type=int, default=50)

    s3 = sub.add_parser("compare_hepg2")
    s3.add_argument("--gwps_scores", required=True)
    s3.add_argument("--hepg2_scores", required=True)
    s3.add_argument("--out_dir", required=True)
    s3.add_argument("--genes_file", default=None)

    args = p.parse_args()
    if args.cmd == "make_split":
        make_split(seed=args.seed, out_path=Path(args.out))
    elif args.cmd == "task1_oracle":
        task1_oracle(split_path=Path(args.split), out_dir=Path(args.out_dir))
    elif args.cmd == "score_disease":
        score_disease(
            delta_y_star=Path(args.delta_y_star),
            out_dir=Path(args.out_dir),
            genes_file=Path(args.genes_file) if args.genes_file else None,
            top_k_n=args.top_k,
        )
    elif args.cmd == "compare_hepg2":
        compare_to_hepg2(
            gwps_scores=Path(args.gwps_scores),
            hepg2_scores=Path(args.hepg2_scores),
            out_dir=Path(args.out_dir),
            genes_file=Path(args.genes_file) if args.genes_file else None,
        )


if __name__ == "__main__":
    main()
