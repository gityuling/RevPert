#!/usr/bin/env python3
"""Exploratory UniPert→forward-gallery→Pearson reverse on Essential (NOT for manuscript).

Protocol (honest proxy of UniPert genetic side, not full G2CP chemical transfer):
  1. Encode catalog KO genes with pretrained UniPert.
  2. Fit multi-output Ridge: UniPert emb(g) → observed ΔY(g) on train KOs.
  3. Predict ΔY for the full catalog → Pearson reverse on held-out test queries.
  4. Compare to linear L3 / GEARS predicted-gallery Pearson and RevPert seed-1 ranks.

This is deliberately offline / scratch-results only.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
    if present_a.any():
        ok = np.isfinite(query_Y).all(axis=0) & np.isfinite(g_mat[present_a]).all(axis=0)
    else:
        ok = np.isfinite(query_Y).all(axis=0)
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
    scores = (q @ g.T) / max(q.shape[1], 1)
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
                "gallery_missing": bool(not present_a[j]),
            }
        )
    return pd.DataFrame(rows)


def load_emb_dict(path: Path) -> dict[str, np.ndarray]:
    """Load UniPert-style emb dump (pkl dict) or TSV."""
    path = Path(path)
    if path.suffix == ".pkl":
        raw = pickle.loads(path.read_bytes())
        out: dict[str, np.ndarray] = {}
        for k, v in raw.items():
            vec = np.asarray(v, dtype=float).ravel()
            if vec.size == 0:
                continue
            ks = str(k)
            out[ks] = vec
            out[clean_ko(ks)] = vec
            out[ks.upper()] = vec
            out[clean_ko(ks).upper()] = vec
        return out
    kos, mat = load_emb_matrix(path)
    return {k: mat[i] for i, k in enumerate(kos)}


def load_emb_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    """TSV (kos × dims or dims × kos) or pkl dict[str→vec]."""
    path = Path(path)
    if path.suffix == ".pkl":
        raw = pickle.loads(path.read_bytes())
        kos = sorted(raw.keys())
        mat = np.stack([np.asarray(raw[k], dtype=float).ravel() for k in kos], axis=0)
        return [clean_ko(k) for k in kos], mat
    df = pd.read_csv(path, sep="\t", index_col=0)
    # prefer rows=KOs
    if df.shape[0] >= df.shape[1]:
        kos = [clean_ko(i) for i in df.index.astype(str)]
        mat = df.to_numpy(dtype=float)
    else:
        kos = [clean_ko(c) for c in df.columns.astype(str)]
        mat = df.to_numpy(dtype=float).T
    return kos, mat


def encode_with_unipert(
    gene_names: list[str],
    unipert_root: Path,
    data_dir: Path,
    model_dir: Path,
    device: str,
    cache_pkl: Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    if cache_pkl.is_file():
        print(f"[cache] load embeddings: {cache_pkl}")
        reps = pickle.loads(cache_pkl.read_bytes())
        missing = [g for g in gene_names if g not in reps and g.upper() not in reps]
        return reps, missing

    sys.path.insert(0, str(unipert_root))
    from unipert import UniPert  # noqa: WPS440

    model = UniPert(
        data_dir=str(data_dir),
        model_hparams={"save_dir": str(model_dir)},
        device=device,
    )
    reps, invalid = model.encode_genetic_perturbagens_from_gene_names(gene_names, save=False)
    reps = {clean_ko(k): np.asarray(v, dtype=float).ravel() for k, v in (reps or {}).items()}
    cache_pkl.parent.mkdir(parents=True, exist_ok=True)
    cache_pkl.write_bytes(pickle.dumps(reps))
    print(f"[save] UniPert embeddings → {cache_pkl} (n={len(reps)}, invalid={len(invalid or [])})")
    return reps, list(invalid or [])


def fit_emb_to_delta_gallery(
    emb: dict[str, np.ndarray],
    obs: dict[str, np.ndarray],
    train_kos: list[str],
    catalog: list[str],
    alpha: float,
) -> dict[str, np.ndarray]:
    train = [k for k in train_kos if k in emb and k in obs]
    if len(train) < 20:
        raise SystemExit(f"Too few train KOs with emb+obs: {len(train)}")
    Xtr = np.stack([emb[k] for k in train], axis=0)
    Ytr = np.stack([obs[k] for k in train], axis=0)
    Ytr = np.nan_to_num(Ytr, nan=0.0)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(Xtr, Ytr)
    out = {}
    for k in catalog:
        if k not in emb:
            continue
        pred = model.predict(emb[k].reshape(1, -1))[0]
        out[k] = np.asarray(pred, dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell_line", default="hepg2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--unipert_root",
        type=str,
        default=str(
            _ROOT
            / "reverse/external/unipert_probe/UniPert-G2CP_reproduce-code/UniPert-G2CP_reproduce/UniPert"
        ),
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=str(_ROOT / "reverse/external/unipert_probe/assets/data"),
    )
    ap.add_argument(
        "--model_dir",
        type=str,
        default=str(_ROOT / "reverse/external/unipert_probe/assets/current_model"),
    )
    ap.add_argument(
        "--emb_cache",
        type=str,
        default="",
        help="Optional precomputed emb pkl/tsv; skips UniPert encode if set",
    )
    ap.add_argument("--ridge_alpha", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_ROOT / "reverse/results/revpert/unipert_probe"),
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = resolve_cell_paths(args.cell_line, seed=args.seed)
    split = load_split_kos(Path(paths["split"]))
    genes, l3_gal = build_delta_gallery(Path(paths["pred_dir"]), Path(paths["dataset_h5ad"]))
    _, obs = load_observed_deltas(Path(paths["pseudobulk_deltas"]), genes)

    # GEARS gallery if present (align to L3 gene axis)
    gears_gal = None
    gears_dir = Path(paths["gears_dir"])
    if (gears_dir / "all_predictions.json").is_file():
        g_genes, gears_raw = build_delta_gallery(gears_dir, Path(paths["dataset_h5ad"]))
        idx = {g: i for i, g in enumerate(g_genes)}
        gears_gal = {
            k: np.array([vec[idx[g]] if g in idx else np.nan for g in genes], dtype=float)
            for k, vec in gears_raw.items()
        }

    catalog = sorted(set(obs) & set(l3_gal))
    if gears_gal is not None:
        catalog = sorted(set(catalog) & set(gears_gal))
    queries = [k for k in split["test"] if k in catalog]
    train_kos = [clean_ko(k) for k in split["train"]]

    # embeddings
    cache_pkl = out / f"{args.cell_line}_seed{args.seed}_unipert_emb.pkl"
    if args.emb_cache:
        emb = load_emb_dict(Path(args.emb_cache))
        invalid: list[str] = []
        # peek dim from first gene-like key
        sample = next(iter(emb.values()))
        print(f"[emb_cache] n_keys={len(emb)} dim={sample.size}")
    else:
        need = sorted(set(catalog) | set(train_kos) | set(queries))
        emb, invalid = encode_with_unipert(
            need,
            Path(args.unipert_root),
            Path(args.data_dir),
            Path(args.model_dir),
            args.device,
            cache_pkl,
        )
        # normalize keys
        emb2 = {}
        for k, v in emb.items():
            emb2[clean_ko(k)] = np.asarray(v, dtype=float).ravel()
            emb2[clean_ko(k).upper()] = emb2[clean_ko(k)]
        emb = emb2

    def _get_emb(k: str) -> np.ndarray | None:
        if k in emb:
            return emb[k]
        if k.upper() in emb:
            return emb[k.upper()]
        return None

    emb_clean = {k: _get_emb(k) for k in catalog if _get_emb(k) is not None}
    emb_clean = {k: v for k, v in emb_clean.items() if v is not None}

    unipert_gal = fit_emb_to_delta_gallery(
        emb_clean, obs, train_kos, catalog, alpha=args.ridge_alpha
    )
    # Restrict shared catalog to KOs UniPert can cover
    catalog_u = sorted(set(catalog) & set(unipert_gal))
    queries_u = [k for k in queries if k in catalog_u]
    Q = np.stack([obs[k] for k in queries_u], axis=0)

    frames = [
        pearson_ranks(Q, queries_u, unipert_gal, catalog_u, "unipert_ridge_gallery_pearson"),
        pearson_ranks(Q, queries_u, {k: l3_gal[k] for k in catalog_u}, catalog_u, "linear_L3_gallery_pearson"),
    ]
    if gears_gal is not None:
        frames.append(
            pearson_ranks(
                Q,
                queries_u,
                {k: gears_gal[k] for k in catalog_u},
                catalog_u,
                "gears_gallery_pearson",
            )
        )

    # RevPert seed-1 ranks if available (same queries intersection)
    rev_path = (
        _ROOT
        / "reverse/results/revpert/essential/seed1_per_query_ranks"
        / f"{args.cell_line}_seed1_ranks.tsv"
    )
    if rev_path.is_file():
        rdf = pd.read_csv(rev_path, sep="\t")
        # expect columns true_ko + pearson/revpert ranks — keep flexible
        cols = {c.lower(): c for c in rdf.columns}
        ko_col = cols.get("true_ko") or cols.get("ko") or rdf.columns[0]
        sub = rdf[rdf[ko_col].map(clean_ko).isin(queries_u)].copy()
        sub.to_csv(out / f"{args.cell_line}_seed{args.seed}_revpert_ref_ranks.tsv", sep="\t", index=False)

    ranks = pd.concat(frames, ignore_index=True)
    ranks.to_csv(out / f"{args.cell_line}_seed{args.seed}_ranks.tsv", sep="\t", index=False)
    summary_rows = []
    for method, sub in ranks.groupby("method"):
        row = summarize_recovery(sub)
        row["method"] = method
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / f"{args.cell_line}_seed{args.seed}_summary.tsv", sep="\t", index=False)

    meta = {
        "cell_line": paths["label"],
        "seed": int(args.seed),
        "n_catalog": len(catalog_u),
        "n_test_queries": len(queries_u),
        "n_emb": len(emb_clean),
        "n_invalid_encode": len(invalid),
        "ridge_alpha": args.ridge_alpha,
        "protocol": (
            "UniPert gene embeddings → Ridge(emb→ΔY) predicted gallery → Pearson reverse; "
            "exploratory only, not manuscript."
        ),
        "note": (
            "This is NOT full UniPert-G2CP (no chemical transfer / LINCS G2CP). "
            "It uses UniPert as the genetic perturbagen encoder feeding a linear forward head "
            "trained on Essential train KOs — closest lightweight 'their encoder on our data' probe."
        ),
    }
    (out / f"{args.cell_line}_seed{args.seed}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
