#!/usr/bin/env python3
"""
Re-eval official PDGrapher checkpoints with Task-1-style metrics (incl. median rank),
then build unified main + SI tables vs dual / Pearson on the same genetic splits.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1] / "external" / "PDGrapher" / "repo"
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from pdgrapher import Dataset, PDGrapher  # noqa: E402
from pdgrapher._utils import get_thresholds  # noqa: E402

CELLS = [
    "A549",
    "A375",
    "AGS",
    "BICR6",
    "ES2",
    "HT29",
    "MCF7",
    "PC3",
    "U251MG",
    "YAPC",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _idcg(num_correct: int, num_nodes: int) -> float:
    idcg = 0.0
    for rank in range(1, num_correct + 1):
        idcg += (1.0 - rank / num_nodes) / np.log2(rank + 1)
    return idcg


@torch.no_grad()
def eval_fold(model: PDGrapher, dataset: Dataset, device: torch.device) -> dict:
    thresholds = get_thresholds(dataset)
    thresholds = {k: (v.to(device) if v is not None else v) for k, v in thresholds.items()}
    model.perturbation_discovery.edge_index = model.perturbation_discovery.edge_index.to(device)
    model.response_prediction.edge_index = model.response_prediction.edge_index.to(device)
    model.perturbation_discovery = model.perturbation_discovery.to(device).eval()
    model.response_prediction = model.response_prediction.to(device).eval()

    (
        _tf,
        _tb,
        _vf,
        _vb,
        _tef,
        test_loader_backward,
    ) = dataset.get_dataloaders(num_workers=0, batch_size=1, shuffle=False)

    r1, r10, r100, r1000 = [], [], [], []
    rankings, ndcgs, med_ranks = [], [], []
    n_partial = 0
    n_test = 0

    for data in test_loader_backward:
        n_test += 1
        pred = model.perturbation_discovery(
            torch.concat(
                [data.diseased.view(-1, 1).to(device), data.treated.view(-1, 1).to(device)],
                1,
            ),
            data.batch.to(device),
            mutilate_mutations=data.mutations.to(device),
            threshold_input=thresholds,
        )
        num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
        correct = [
            int(x)
            for x in torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist()
        ]
        order = torch.argsort(pred.detach().cpu().view(-1, num_nodes), descending=True)[0].tolist()
        rank_of = {g: i for i, g in enumerate(order)}  # 0-based
        ranks0 = [rank_of[c] for c in correct]
        n_c = max(len(ranks0), 1)
        med_ranks.append(min(ranks0) + 1)

        def hit(k: int) -> float:
            return sum(1 for r in ranks0 if r < k) / n_c

        r1.append(hit(1))
        r10.append(hit(10))
        r100.append(hit(100))
        r1000.append(hit(1000))
        dcg = 0.0
        for r0 in ranks0:
            rankings.append(1.0 - (r0 / num_nodes))
            rank1 = r0 + 1
            dcg += (1.0 - rank1 / num_nodes) / np.log2(rank1 + 1)
        idcg = _idcg(len(ranks0), num_nodes)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        if any(r < len(ranks0) for r in ranks0):
            n_partial += 1

    return {
        "n_test": n_test,
        "recall@1": float(np.mean(r1)),
        "recall@10": float(np.mean(r10)),
        "recall@100": float(np.mean(r100)),
        "recall@1000": float(np.mean(r1000)),
        "pct_partially_accurate": 100.0 * n_partial / max(n_test, 1),
        "ranking_score": float(np.mean(rankings)) if rankings else float("nan"),
        "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "median_rank_true": float(np.median(med_ranks)) if med_ranks else float("nan"),
    }


def reeval_official(
    cells: list[str],
    data_dir: Path,
    splits_dir: Path,
    official_dir: Path,
    n_layers_gnn: int,
) -> pd.DataFrame:
    rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cell in cells:
        cell_out = official_dir / cell / f"n_gnn_{n_layers_gnn}"
        existing = cell_out / "fold_metrics.tsv"
        if existing.exists():
            prev = pd.read_csv(existing, sep="\t")
            if "median_rank_true" in prev.columns and prev["median_rank_true"].notna().all():
                _log(f"[{cell}] median_rank already present — reuse")
                prev = prev.copy()
                prev["method"] = "pdgrapher_official"
                rows.append(prev)
                continue

        use_forward = cell not in {"ES2", "BICR6", "YAPC", "AGS", "U251MG", "HT29", "A375"}
        fp = data_dir / f"data_forward_{cell}.pt"
        dataset = Dataset(
            forward_path=str(fp) if fp.exists() else None,
            backward_path=str(data_dir / f"data_backward_{cell}.pt"),
            splits_path=str(splits_dir / "genetic" / cell / "random" / "5fold" / "splits.pt"),
        )
        edge_index = torch.load(
            data_dir / f"edge_index_{cell}.pt", map_location="cpu", weights_only=False
        )
        base = PDGrapher(
            edge_index,
            model_kwargs={
                "n_layers_nn": 1,
                "n_layers_gnn": n_layers_gnn,
                "num_vars": dataset.get_num_vars(),
            },
            response_kwargs={"train": False},
            perturbation_kwargs={"train": False},
        )
        cell_rows = []
        for fold_idx in range(1, dataset.num_of_folds + 1):
            fold_log = cell_out / f"fold_{fold_idx}"
            rp_path = fold_log / "response_prediction.pt"
            pd_path = fold_log / "perturbation_discovery.pt"
            if not (rp_path.exists() and pd_path.exists()):
                raise FileNotFoundError(f"missing ckpt for {cell} fold {fold_idx}")
            dataset.prepare_fold(fold_idx)
            model = base  # structure shared; weights loaded below
            # fresh module copies via state_dict load into a deepcopy-like rebuild
            model = PDGrapher(
                edge_index,
                model_kwargs={
                    "n_layers_nn": 1,
                    "n_layers_gnn": n_layers_gnn,
                    "num_vars": dataset.get_num_vars(),
                },
                response_kwargs={"train": False},
                perturbation_kwargs={"train": False},
            )
            rp = torch.load(rp_path, map_location="cpu", weights_only=False)
            pd_ckpt = torch.load(pd_path, map_location="cpu", weights_only=False)
            model.response_prediction.load_state_dict(rp["model_state_dict"])
            model.perturbation_discovery.load_state_dict(pd_ckpt["model_state_dict"])
            del rp, pd_ckpt
            _log(f"[{cell}] fold {fold_idx}: re-eval (+median_rank)")
            metrics = eval_fold(model, dataset, device)
            metrics.update(
                {
                    "method": "pdgrapher_official",
                    "cell": cell,
                    "fold": fold_idx,
                    "n_layers_gnn": n_layers_gnn,
                    "use_forward": use_forward,
                }
            )
            cell_rows.append(metrics)
            _log(
                f"[{cell}] fold {fold_idx}: r@10={metrics['recall@10']:.4f} "
                f"median_rank={metrics['median_rank_true']:.1f} "
                f"partial%={metrics['pct_partially_accurate']:.2f}"
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()

        df_cell = pd.DataFrame(cell_rows)
        # keep previous columns if present, overwrite with full metrics
        out_cols = [
            "n_test",
            "recall@1",
            "recall@10",
            "recall@100",
            "recall@1000",
            "pct_partially_accurate",
            "ranking_score",
            "ndcg",
            "median_rank_true",
            "cell",
            "fold",
            "n_layers_gnn",
        ]
        df_cell[out_cols].to_csv(cell_out / "fold_metrics.tsv", sep="\t", index=False)
        rows.append(df_cell)
    return pd.concat(rows, ignore_index=True)


def build_tables(official: pd.DataFrame, dual_path: Path, out_dir: Path) -> None:
    dual = pd.read_csv(dual_path, sep="\t")
    dual = dual[dual["method"].isin(["dual_encoder", "pearson_train_gallery"])].copy()
    keep = [
        "method",
        "cell",
        "fold",
        "n_test",
        "recall@1",
        "recall@10",
        "recall@100",
        "recall@1000",
        "pct_partially_accurate",
        "ranking_score",
        "ndcg",
        "median_rank_true",
    ]
    off = official.copy()
    if "method" not in off.columns:
        off["method"] = "pdgrapher_official"
    all_folds = pd.concat([off[keep], dual[keep]], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_folds.to_csv(out_dir / "all_fold_metrics.tsv", sep="\t", index=False)

    # Main table: median rank + R@10 (Task-1 style)
    g = all_folds.groupby(["cell", "method"], as_index=False).agg(
        median_rank_mean=("median_rank_true", "mean"),
        median_rank_std=("median_rank_true", "std"),
        recall_at_10_mean=("recall@10", "mean"),
        recall_at_10_std=("recall@10", "std"),
        n_folds=("fold", "count"),
    )
    method_order = ["pearson_train_gallery", "pdgrapher_official", "dual_encoder"]
    cell_order = CELLS
    g["method"] = pd.Categorical(g["method"], method_order, ordered=True)
    g["cell"] = pd.Categorical(g["cell"], cell_order, ordered=True)
    g = g.sort_values(["cell", "method"])
    g.to_csv(out_dir / "main_table_median_rank_r10.tsv", sep="\t", index=False)

    # Wide main for manuscript
    wide_rank = g.pivot(index="cell", columns="method", values="median_rank_mean")
    wide_r10 = g.pivot(index="cell", columns="method", values="recall_at_10_mean")
    wide = pd.DataFrame(
        {
            "pearson_median_rank": wide_rank.get("pearson_train_gallery"),
            "pdgrapher_median_rank": wide_rank.get("pdgrapher_official"),
            "dual_median_rank": wide_rank.get("dual_encoder"),
            "pearson_r@10": 100 * wide_r10.get("pearson_train_gallery"),
            "pdgrapher_r@10": 100 * wide_r10.get("pdgrapher_official"),
            "dual_r@10": 100 * wide_r10.get("dual_encoder"),
        }
    )
    wide.to_csv(out_dir / "main_table_wide.tsv", sep="\t")

    # SI: PDGrapher-paper metrics
    si = all_folds.groupby(["cell", "method"], as_index=False).agg(
        partial_pct_mean=("pct_partially_accurate", "mean"),
        partial_pct_std=("pct_partially_accurate", "std"),
        recall_at_1_mean=("recall@1", "mean"),
        recall_at_1_std=("recall@1", "std"),
        recall_at_100_mean=("recall@100", "mean"),
        ndcg_mean=("ndcg", "mean"),
        ndcg_std=("ndcg", "std"),
    )
    si["method"] = pd.Categorical(si["method"], method_order, ordered=True)
    si["cell"] = pd.Categorical(si["cell"], cell_order, ordered=True)
    si = si.sort_values(["cell", "method"])
    si.to_csv(out_dir / "si_table_partial_ndcg.tsv", sep="\t", index=False)

    readme = out_dir / "README.md"
    readme.write_text(
        "# PDGrapher genetic as Task-1 extra data source\n\n"
        "Same reverse-retrieval task; PDGrapher genetic 10 lines as an additional resource.\n\n"
        "- **Main:** `main_table_wide.tsv` / `main_table_median_rank_r10.tsv` "
        "(median rank + Recall@10)\n"
        "- **SI:** `si_table_partial_ndcg.tsv` (partial%, R@1/100, nDCG — PDGrapher-paper style)\n"
        "- **Folds:** `all_fold_metrics.tsv`\n"
        "- Methods: `pearson_train_gallery`, `pdgrapher_official`, `dual_encoder`\n",
        encoding="utf-8",
    )
    _log(f"Wrote {out_dir}")
    _log("\nMain wide:\n" + wide.to_string(float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    torch.set_float32_matmul_precision("high")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=Path,
        default=_ROOT
        / "reverse/external/PDGrapher/data/processed/torch_data/real_lognorm",
    )
    p.add_argument(
        "--splits_dir",
        type=Path,
        default=_ROOT / "reverse/external/PDGrapher/data/processed/splits",
    )
    p.add_argument(
        "--official_dir",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_official_genetic",
    )
    p.add_argument(
        "--dual_metrics",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_genetic_dual/all_fold_metrics.tsv",
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_genetic_task1_unified",
    )
    p.add_argument("--cells", nargs="+", default=CELLS)
    p.add_argument("--n_layers_gnn", type=int, default=2)
    p.add_argument(
        "--skip_reeval",
        action="store_true",
        help="Only assemble tables from existing fold_metrics (must already have median_rank).",
    )
    args = p.parse_args()

    if args.skip_reeval:
        frames = []
        for cell in args.cells:
            fp = args.official_dir / cell / f"n_gnn_{args.n_layers_gnn}" / "fold_metrics.tsv"
            d = pd.read_csv(fp, sep="\t")
            d["method"] = "pdgrapher_official"
            frames.append(d)
        official = pd.concat(frames, ignore_index=True)
    else:
        official = reeval_official(
            args.cells, args.data_dir, args.splits_dir, args.official_dir, args.n_layers_gnn
        )
        official.to_csv(args.official_dir / "all_fold_metrics.tsv", sep="\t", index=False)

    build_tables(official, args.dual_metrics, args.out_dir)


if __name__ == "__main__":
    main()
