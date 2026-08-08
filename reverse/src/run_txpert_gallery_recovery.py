#!/usr/bin/env python3
"""Forward-gallery reverse recovery: TxPert / GEARS / linear L3 + dual-encoder reference.

TxPert itself is forward-only; here it enters ReversePerturb-Bench as a *gallery builder*
(same protocol as GEARS/scGPT gallery ablations).
"""

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

from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.recovery import summarize_recovery  # noqa: E402
from reverse.src.reverse_data import load_reverse_bundle, load_split_kos  # noqa: E402
from reverse.src.reverse_model import PCAProjector, ReverseDualEncoder  # noqa: E402
from reverse.src.train_reverse_retrieval import dual_encoder_ranks, load_gene_prior_matrix  # noqa: E402


def align_gallery_to_genes(
    gallery: dict[str, np.ndarray], src_genes: list[str], dst_genes: list[str]
) -> dict[str, np.ndarray]:
    idx = {g: i for i, g in enumerate(src_genes)}
    out = {}
    for ko, vec in gallery.items():
        out[ko] = np.array([vec[idx[g]] if g in idx else np.nan for g in dst_genes], dtype=float)
    return out


def pearson_ranks(
    query_Y: np.ndarray,
    query_kos: list[str],
    gallery: dict[str, np.ndarray],
    gallery_kos: list[str],
    method: str,
) -> pd.DataFrame:
    """Pearson reverse recovery. Missing gallery KOs get -inf scores → worst ranks.

    Incomplete forward models (e.g. scGPT) are scored on the same catalog/queries
    as complete galleries; coverage failure is baked into rank.
    """
    g_rows = []
    present = []
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


def build_delta_gallery(pred_dir: Path, dataset_h5ad: Path):
    genes, pred_abs = load_prediction_dir(pred_dir)
    ctrl = load_ctrl_from_perturb_processed(dataset_h5ad, genes)
    return genes, absolute_to_delta(pred_abs, ctrl)


def main() -> None:
    ap = argparse.ArgumentParser(description="TxPert/GEARS/scGPT/linear gallery reverse recovery")
    ap.add_argument("--cell_line", type=str, default="hepg2", choices=list_cells())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--galleries",
        nargs="+",
        default=["txpert_gat", "txpert_xcell", "linear_L3", "gears", "scgpt", "dual_encoder_v2"],
        help="Which methods to evaluate",
    )
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    paths = resolve_cell_paths(args.cell_line, seed=args.seed)
    out = Path(args.out_dir) if args.out_dir else Path(paths["gallery_compare_out"])
    out.mkdir(parents=True, exist_ok=True)

    split = load_split_kos(Path(paths["split"]))
    test_kos = split["test"]
    h5ad = Path(paths["dataset_h5ad"])

    # Reference gene axis = linear L3
    ref_genes, lin_gal = build_delta_gallery(Path(paths["pred_dir"]), h5ad)
    _, obs = load_observed_deltas(Path(paths["pseudobulk_deltas"]), ref_genes)

    want = set(args.galleries)
    # Primary shared-catalog methods (exclude scgpt from intersection — fewer KOs)
    primary = []
    if "linear_L3" in want:
        primary.append(("linear_L3_gallery_pearson", Path(paths["pred_dir"])))
    if "txpert_gat" in want:
        primary.append(("txpert_gat_gallery_pearson", Path(paths["txpert_gat_dir"])))
    if "txpert_xcell" in want:
        primary.append(("txpert_xcell_gallery_pearson", Path(paths["txpert_xcell_dir"])))
    if "gears" in want:
        primary.append(("gears_gallery_pearson", Path(paths["gears_dir"])))

    loaded: dict[str, dict[str, np.ndarray]] = {}
    available_methods: list[str] = []
    for method, pred_dir in primary:
        if pred_dir is None or not (pred_dir / "all_predictions.json").is_file():
            print(f"[skip] missing gallery for {method}: {pred_dir}")
            continue
        if method == "linear_L3_gallery_pearson":
            loaded[method] = lin_gal
        else:
            src_genes, gal_raw = build_delta_gallery(pred_dir, h5ad)
            loaded[method] = align_gallery_to_genes(gal_raw, src_genes, ref_genes)
        available_methods.append(method)

    catalog = set(obs.keys())
    for gal in loaded.values():
        catalog &= set(gal.keys())
    catalog = sorted(catalog)
    queries = [k for k in test_kos if k in catalog]
    meta = {
        "cell_line": paths["label"],
        "seed": int(paths["seed"]),
        "n_catalog": len(catalog),
        "n_test_queries": len(queries),
        "n_ref_genes": len(ref_genes),
        "methods_primary": available_methods,
        "note": "Primary galleries share one catalog intersection. "
        "scGPT (fewer predicted KOs) is scored on a restricted cover subset.",
    }
    print(json.dumps({k: meta[k] for k in meta if k != "note"}, indent=2))
    if len(queries) < 10:
        raise SystemExit(f"Too few queries after intersection: {len(queries)}")

    Q = np.stack([obs[k] for k in queries], axis=0)
    frames = []
    for method in available_methods:
        frames.append(pearson_ranks(Q, queries, loaded[method], catalog, method))

    # scGPT on the *same* catalog/queries; missing KO preds → worst rank
    if "scgpt" in want:
        scgpt_path = Path(paths["scgpt_dir"])
        if not (scgpt_path / "all_predictions.json").is_file():
            print(f"[skip] missing scGPT gallery: {scgpt_path}")
        else:
            sg_genes, sg_raw = build_delta_gallery(scgpt_path, h5ad)
            sg_gal = align_gallery_to_genes(sg_raw, sg_genes, ref_genes)
            n_present = sum(1 for k in catalog if k in sg_gal)
            meta["scgpt_n_present_in_catalog"] = n_present
            meta["scgpt_coverage"] = float(n_present) / max(len(catalog), 1)
            frames.append(pearson_ranks(Q, queries, sg_gal, catalog, "scgpt_gallery_pearson"))
            available_methods.append("scgpt_gallery_pearson")

    if "dual_encoder_v2" in want:
        ckpt_path = Path(paths["retrieval_out"]) / "best.pt"
        if not ckpt_path.is_file():
            print(f"[skip] dual ckpt missing: {ckpt_path}")
        else:
            device = torch.device(
                "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
            )
            genes_b, tables, _ = load_reverse_bundle(
                Path(paths["pseudobulk_deltas"]),
                Path(paths["pred_dir"]) / "gene_names.json",
                Path(paths["split"]),
                Path(paths["p_tsv"]),
            )
            assert genes_b == ref_genes
            cat_kos, cat_P = [], []
            for part in ("train", "val", "test"):
                cat_kos.extend(tables[part]["kos"])
                cat_P.append(tables[part]["P"])
            cat_P = np.vstack(cat_P)
            cat_G = load_gene_prior_matrix(Path(paths["esm_tsv"]), cat_kos, out_dim=64)
            obs_bundle = {}
            for part in ("train", "val", "test"):
                for k, y in zip(tables[part]["kos"], tables[part]["Y"]):
                    obs_bundle[k] = y
            q2 = [k for k in queries if k in obs_bundle and k in set(cat_kos)]
            Yq = np.stack([obs_bundle[k] for k in q2], axis=0)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
            meta["n_dual_queries"] = len(q2)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(out / "ranks.tsv", sep="\t", index=False)
    summaries = []
    for m, sub in all_df.groupby("method"):
        s = summarize_recovery(sub)
        s["method"] = m
        s["mrr"] = float((1.0 / sub["rank"]).mean())
        s["cell_line"] = paths["label"]
        s["seed"] = int(paths["seed"])
        summaries.append(s)
    sum_df = pd.DataFrame(summaries).sort_values("median_rank")
    sum_df.to_csv(out / "summary.tsv", sep="\t", index=False)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2))
    (out / "protocol_notes.json").write_text(json.dumps(meta, indent=2))
    print(sum_df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
