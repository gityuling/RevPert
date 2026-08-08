#!/usr/bin/env python3
"""Compare gallery-style reverse methods on the 3 primary resistance proving-ground queries.

Methods (all signed dual-arm: Arm A on Δ, Arm B on −Δ):
  - pearson      whole-profile Pearson (manuscript baseline)
  - spearman     whole-profile Spearman
  - cosine       whole-profile cosine
  - cmap_lite    up/down set enrichment vs gallery ranks
  - gem_lite     Pearson on top-|Δ| DEG subspace
  - ridge_P      (HepG2 Essential only) Ridge ΔY→P, cosine to catalog P

Primary queries:
  1) GSE322742 HepG2 SR − parental   → HepG2 Essential L3 gallery
  2) GSE143233 patient SR − normal    → HepG2 Essential L3 gallery
  3) GSE120932 K562-IR +drug − parental → K562 GWPS observed gallery
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

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

SIG = _ROOT / "reverse/data/signatures"
OUT = _ROOT / "reverse/results/resistance_gallery_method_compare"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _pair_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    x, y = a[mask], b[mask]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if metric == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    if metric == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    if metric == "cosine":
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    raise ValueError(metric)


def score_profile_metrics(
    gallery: dict[str, np.ndarray],
    query: np.ndarray,
    kos: list[str],
    metric: str,
) -> np.ndarray:
    return np.array([_pair_metric(gallery[k], query, metric) for k in kos], dtype=float)


def score_cmap_lite(
    gallery: dict[str, np.ndarray],
    query: np.ndarray,
    kos: list[str],
    top_n: int = 100,
) -> np.ndarray:
    """Higher = better match to query up/down sets."""
    q = np.nan_to_num(query, nan=0.0)
    up = np.argsort(-q)[:top_n]
    down = np.argsort(q)[:top_n]
    # precompute per-KO gene ranks (1=highest expression change)
    scores = np.empty(len(kos), dtype=float)
    for j, k in enumerate(kos):
        v = np.nan_to_num(gallery[k], nan=0.0)
        order = np.argsort(-v)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        s_up = -ranks[up].mean()
        s_down = ranks[down].mean()
        scores[j] = s_up + s_down
    return scores


def score_gem_lite(
    gallery: dict[str, np.ndarray],
    query: np.ndarray,
    kos: list[str],
    top_deg: int = 200,
) -> np.ndarray:
    q = np.nan_to_num(query, nan=0.0)
    idx = np.argsort(-np.abs(q))[:top_deg]
    qq = q[idx]
    qq = (qq - qq.mean()) / (qq.std() + 1e-12)
    scores = np.empty(len(kos), dtype=float)
    for j, k in enumerate(kos):
        v = np.nan_to_num(gallery[k], nan=0.0)[idx]
        if v.std() < 1e-12:
            scores[j] = 0.0
        else:
            vv = (v - v.mean()) / (v.std() + 1e-12)
            scores[j] = float(np.dot(vv, qq) / len(idx))
    return scores


def score_ridge_p(
    query: np.ndarray,
    kos: list[str],
    ridge: Ridge,
    cat_P: np.ndarray,
    ko_to_i: dict[str, int],
) -> np.ndarray:
    phat = ridge.predict(np.nan_to_num(query, nan=0.0)[None, :])[0]
    phat = (phat - phat.mean()) / (phat.std() + 1e-12)
    P = np.nan_to_num(cat_P, nan=0.0)
    Pz = (P - P.mean(axis=1, keepdims=True)) / (P.std(axis=1, keepdims=True) + 1e-12)
    scores = np.full(len(kos), np.nan)
    for j, k in enumerate(kos):
        i = ko_to_i[k]
        scores[j] = float(np.dot(phat, Pz[i]) / max(len(phat), 1))
    return scores


def dualarm_table(
    method: str,
    kos: list[str],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> pd.DataFrame:
    # missing scores → worst
    sa = np.where(np.isfinite(scores_a), scores_a, -np.inf)
    sb = np.where(np.isfinite(scores_b), scores_b, -np.inf)
    ra, rb = _ranks_desc(sa), _ranks_desc(sb)
    df = pd.DataFrame(
        {
            "method": method,
            "ko": kos,
            "score_armA": scores_a,
            "score_armB": scores_b,
            "rank_armA": ra.astype(int),
            "rank_armB": rb.astype(int),
        }
    )
    df["pref_arm"] = np.where(df["rank_armA"] <= df["rank_armB"], "A", "B")
    df["rank_pref"] = df[["rank_armA", "rank_armB"]].min(axis=1)
    return df


def load_hepg2_gallery():
    paths = resolve_cell_paths("hepg2", seed=1)
    genes, pred_abs = load_prediction_dir(Path(paths["pred_dir"]))
    ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
    gal = absolute_to_delta(pred_abs, ctrl)
    kos = sorted(gal.keys())
    # ridge assets
    g2, tables, _ = load_reverse_bundle(
        Path(paths["pseudobulk_deltas"]),
        Path(paths["pred_dir"]) / "gene_names.json",
        Path(paths["split"]),
        Path(paths["p_tsv"]),
    )
    assert g2 == genes
    cat_kos, cat_P = [], []
    for part in ("train", "val", "test"):
        cat_kos.extend(tables[part]["kos"])
        cat_P.append(tables[part]["P"])
    cat_P = np.vstack(cat_P)
    ridge = Ridge(alpha=1.0, random_state=0)
    ridge.fit(
        np.nan_to_num(tables["train"]["Y"], nan=0.0),
        np.nan_to_num(tables["train"]["P"], nan=0.0),
    )
    return {
        "name": "hepg2_essential_L3",
        "genes": genes,
        "gallery": gal,
        "kos": kos,
        "ridge": ridge,
        "cat_kos": cat_kos,
        "cat_P": cat_P,
        "ko_to_i": {k: i for i, k in enumerate(cat_kos)},
    }


def load_gwps_gallery():
    genes, gal = load_gwps_deltas()
    kos = sorted(gal.keys())
    return {"name": "k562_gwps_observed", "genes": genes, "gallery": gal, "kos": kos}


def run_query(
    tag: str,
    delta_path: Path,
    pack: dict,
    focus: list[str],
    methods: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    star = load_vector_tsv(delta_path)
    genes = pack["genes"]
    query = align_delta_y_star(star, genes)
    neg = -query
    gal, kos = pack["gallery"], pack["kos"]
    # restrict to KOs present
    kos = [k for k in kos if k in gal]

    deg = np.nan_to_num(np.abs(query), nan=0.0)
    # DEG rank among genes that appear as catalog KOs and have finite Δ
    deg_rank = {}
    # rank all finite genes on axis by |Δ|
    finite_idx = np.where(np.isfinite(query))[0]
    order = finite_idx[np.argsort(-np.abs(query[finite_idx]))]
    gene_deg_rank = {genes[i]: r + 1 for r, i in enumerate(order)}

    frames = []
    for m in methods:
        if m in {"pearson", "spearman", "cosine"}:
            sa = score_profile_metrics(gal, query, kos, m)
            sb = score_profile_metrics(gal, neg, kos, m)
        elif m == "cmap_lite":
            sa = score_cmap_lite(gal, query, kos)
            sb = score_cmap_lite(gal, neg, kos)
        elif m == "gem_lite":
            sa = score_gem_lite(gal, query, kos)
            sb = score_gem_lite(gal, neg, kos)
        elif m == "ridge_P":
            if "ridge" not in pack:
                continue
            # only score KOs in P catalog
            kos_r = [k for k in kos if k in pack["ko_to_i"]]
            sa = score_ridge_p(query, kos_r, pack["ridge"], pack["cat_P"], pack["ko_to_i"])
            sb = score_ridge_p(neg, kos_r, pack["ridge"], pack["cat_P"], pack["ko_to_i"])
            df = dualarm_table(m, kos_r, sa, sb)
            frames.append(df)
            continue
        else:
            raise ValueError(m)
        frames.append(dualarm_table(m, kos, sa, sb))

    all_df = pd.concat(frames, ignore_index=True)
    qdir = out_dir / tag
    qdir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(qdir / "all_method_dualarm.tsv", sep="\t", index=False)

    # focus summary
    rows = []
    for m, sub in all_df.groupby("method"):
        for g in focus:
            hit = sub[sub.ko == g]
            row = {
                "tag": tag,
                "gallery": pack["name"],
                "method": m,
                "gene": g,
                "in_gallery": not hit.empty,
                "delta": float(star[g]) if g in star.index else np.nan,
                "deg_rank_abs": gene_deg_rank.get(g, np.nan),
            }
            if not hit.empty:
                r = hit.iloc[0]
                row.update(
                    {
                        "rank_armA": int(r.rank_armA),
                        "rank_armB": int(r.rank_armB),
                        "pref_arm": r.pref_arm,
                        "rank_pref": int(r.rank_pref),
                        "score_armA": float(r.score_armA) if np.isfinite(r.score_armA) else np.nan,
                        "score_armB": float(r.score_armB) if np.isfinite(r.score_armB) else np.nan,
                    }
                )
            rows.append(row)
    foc = pd.DataFrame(rows)
    foc.to_csv(qdir / "focus_compare.tsv", sep="\t", index=False)

    # Top-10 per method preferred arm
    top_rows = []
    for m, sub in all_df.groupby("method"):
        top = sub.nsmallest(10, "rank_pref")
        for _, r in top.iterrows():
            top_rows.append(
                {
                    "tag": tag,
                    "method": m,
                    "rank_pref": int(r.rank_pref),
                    "pref_arm": r.pref_arm,
                    "ko": r.ko,
                    "rank_armA": int(r.rank_armA),
                    "rank_armB": int(r.rank_armB),
                }
            )
    pd.DataFrame(top_rows).to_csv(qdir / "top10_pref.tsv", sep="\t", index=False)
    return foc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    _log("Loading HepG2 Essential L3 gallery ...")
    hepg2 = load_hepg2_gallery()
    _log(f"  n_kos={len(hepg2['kos'])} n_genes={len(hepg2['genes'])}")
    _log("Loading K562 GWPS observed gallery ...")
    gwps = load_gwps_gallery()
    _log(f"  n_kos={len(gwps['kos'])} n_genes={len(gwps['genes'])}")

    queries = [
        {
            "tag": "gse322742_hepg2_sorafenib",
            "delta": SIG / "hepg2_sorafenib_delta_y_star.tsv",
            "pack": hepg2,
            "focus": ["AURKA", "METTL3", "NCOA4", "TFRC", "PKM", "PTEN", "MCL1", "KEAP1", "MYC"],
            "methods": ["pearson", "spearman", "cosine", "cmap_lite", "gem_lite", "ridge_P"],
        },
        {
            "tag": "gse143233_patient_sr_vs_normal",
            "delta": SIG / "gse143233_resistant_minus_normal_delta_y_star.tsv",
            "pack": hepg2,
            "focus": ["METTL3", "AURKA", "NCOA4", "TFRC", "PKM", "CCNK", "FOXO3"],
            "methods": ["pearson", "spearman", "cosine", "cmap_lite", "gem_lite", "ridge_P"],
        },
        {
            "tag": "gse120932_k562_ir_with_drug",
            "delta": SIG / "gse120932_k562_ir_with_drug_delta_y_star.tsv",
            "pack": gwps,
            "focus": ["MYB", "STAT5A", "STAT5B", "RUNX1", "BCR", "ABL1", "MYC"],
            "methods": ["pearson", "spearman", "cosine", "cmap_lite", "gem_lite"],
        },
    ]

    all_foc = []
    for q in queries:
        _log(f"\n=== {q['tag']} ===")
        foc = run_query(q["tag"], q["delta"], q["pack"], q["focus"], q["methods"], out)
        all_foc.append(foc)
        # print headline anchors vs pearson
        anchors = {
            "gse322742_hepg2_sorafenib": ["AURKA"],
            "gse143233_patient_sr_vs_normal": ["METTL3"],
            "gse120932_k562_ir_with_drug": ["MYB", "STAT5A", "STAT5B", "RUNX1", "BCR"],
        }[q["tag"]]
        sub = foc[foc.gene.isin(anchors) & foc.in_gallery]
        cols = ["method", "gene", "rank_armA", "rank_armB", "pref_arm", "rank_pref", "deg_rank_abs"]
        _log(sub[cols].sort_values(["gene", "rank_pref"]).to_string(index=False))

    tab = pd.concat(all_foc, ignore_index=True)
    tab.to_csv(out / "all_focus_compare.tsv", sep="\t", index=False)

    # wide: preferred rank by method for key anchors
    key = tab[tab.in_gallery].copy()
    wide = key.pivot_table(
        index=["tag", "gene"], columns="method", values="rank_pref", aggfunc="first"
    )
    if "pearson" in wide.columns:
        for c in wide.columns:
            if c == "pearson":
                continue
            wide[f"vs_pearson_{c}"] = wide[c] - wide["pearson"]
    wide.to_csv(out / "anchor_pref_rank_wide.tsv", sep="\t")
    _log("\n=== preferred-rank wide (lower better) ===")
    _log(wide.to_string())

    (out / "summary.json").write_text(
        json.dumps(
            {
                "queries": [q["tag"] for q in queries],
                "methods_hepg2": queries[0]["methods"],
                "methods_gwps": queries[2]["methods"],
                "note": "Arm A = score(Δ); Arm B = score(−Δ); preferred = min rank",
            },
            indent=2,
        )
    )
    (out / "README.md").write_text(
        "# Resistance proving ground — gallery method comparison\n\n"
        "Signed dual-arm ranking of catalog KOs for three primary queries.\n"
        "Pearson is the manuscript baseline; others are fair gallery-style alternatives.\n",
        encoding="utf-8",
    )
    _log(f"\nWrote {out}")


if __name__ == "__main__":
    main()
