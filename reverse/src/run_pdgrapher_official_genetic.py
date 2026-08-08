#!/usr/bin/env python3
"""
Train official PDGrapher on genetic splits and report paper-aligned metrics
(recall@k, partial%, nDCG) on the test folds.
"""

from __future__ import annotations

import argparse
import gc
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[1] / "external" / "PDGrapher" / "repo"
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from pdgrapher import Dataset, PDGrapher, Trainer  # noqa: E402
from pdgrapher._utils import get_thresholds  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _idcg(num_correct: int, num_nodes: int) -> float:
    idcg = 0.0
    for rank in range(1, num_correct + 1):
        gain = 1.0 - (rank / num_nodes)
        discount = 1.0 / np.log2(rank + 1)
        idcg += gain * discount
    return idcg


@torch.no_grad()
def eval_test_metrics(model: PDGrapher, dataset: Dataset, device: torch.device) -> dict:
    thresholds = get_thresholds(dataset)
    thresholds = {k: (v.to(device) if v is not None else v) for k, v in thresholds.items()}
    model.response_prediction.edge_index = model.response_prediction.edge_index.to(device)
    model.perturbation_discovery.edge_index = model.perturbation_discovery.edge_index.to(device)
    model.perturbation_discovery = model.perturbation_discovery.to(device)
    model.perturbation_discovery.eval()
    model.response_prediction.eval()

    (
        _tf,
        _tb,
        _vf,
        _vb,
        _tef,
        test_loader_backward,
    ) = dataset.get_dataloaders(num_workers=0, batch_size=1, shuffle=False)

    recall_at_1, recall_at_10, recall_at_100, recall_at_1000 = [], [], [], []
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
        rank_of = {g: i for i, g in enumerate(order)}
        ranks0 = [rank_of[c] for c in correct]
        n_c = max(len(ranks0), 1)
        med_ranks.append(min(ranks0) + 1)
        dcg = 0.0
        for r0 in ranks0:
            rankings.append(1 - (r0 / num_nodes))
            rank1 = r0 + 1
            dcg += (1 - rank1 / num_nodes) / np.log2(rank1 + 1)
        ndcgs.append(dcg / _idcg(len(correct), num_nodes) if correct else 0.0)
        recall_at_1.append(sum(1 for r in ranks0 if r < 1) / n_c)
        recall_at_10.append(sum(1 for r in ranks0 if r < 10) / n_c)
        recall_at_100.append(sum(1 for r in ranks0 if r < 100) / n_c)
        recall_at_1000.append(sum(1 for r in ranks0 if r < 1000) / n_c)
        if any(r < len(ranks0) for r in ranks0):
            n_partial += 1

    return {
        "n_test": n_test,
        "recall@1": float(np.mean(recall_at_1)),
        "recall@10": float(np.mean(recall_at_10)),
        "recall@100": float(np.mean(recall_at_100)),
        "recall@1000": float(np.mean(recall_at_1000)),
        "pct_partially_accurate": 100.0 * n_partial / max(n_test, 1),
        "ranking_score": float(np.mean(rankings)) if rankings else float("nan"),
        "ndcg": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "median_rank_true": float(np.median(med_ranks)) if med_ranks else float("nan"),
    }


def run_cell(
    cell: str,
    data_dir: Path,
    splits_dir: Path,
    out_dir: Path,
    n_layers_gnn: int,
    n_epochs: int,
) -> pd.DataFrame:
    use_forward = cell not in {"ES2", "BICR6", "YAPC", "AGS", "U251MG", "HT29", "A375"}
    forward_path = str(data_dir / f"data_forward_{cell}.pt") if use_forward else None
    # Dataset still accepts path; for no-forward cells notebook passes path but flag False
    if forward_path is None:
        # some cells still have forward file; genetic.py always passes path
        fp = data_dir / f"data_forward_{cell}.pt"
        forward_path = str(fp) if fp.exists() else None

    dataset = Dataset(
        forward_path=forward_path if (data_dir / f"data_forward_{cell}.pt").exists() else None,
        backward_path=str(data_dir / f"data_backward_{cell}.pt"),
        splits_path=str(splits_dir / "genetic" / cell / "random" / "5fold" / "splits.pt"),
    )
    edge_index = torch.load(data_dir / f"edge_index_{cell}.pt", map_location="cpu", weights_only=False)

    cell_out = out_dir / cell / f"n_gnn_{n_layers_gnn}"
    cell_out.mkdir(parents=True, exist_ok=True)

    model = PDGrapher(
        edge_index,
        model_kwargs={
            "n_layers_nn": 1,
            "n_layers_gnn": n_layers_gnn,
            "num_vars": dataset.get_num_vars(),
        },
        response_kwargs={"train": True},
        perturbation_kwargs={"train": True},
    )

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True,
        use_forward_data=use_forward and forward_path is not None,
        use_backward_data=True,
        use_supervision=True,
        use_intervention_data=True,
        supervision_multiplier=0.01,
        log_train=False,
        # Skip per-epoch test metrics (we re-eval paper recall/nDCG after each fold).
        log_test=False,
        logging_dir=str(cell_out),
    )

    _log(
        f"[{cell}] PDGrapher train n_gnn={n_layers_gnn} epochs={n_epochs} "
        f"use_forward={use_forward} n_vars={dataset.get_num_vars()}"
    )
    metrics_path = cell_out / "fold_metrics.tsv"
    done: dict[int, dict] = {}
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path, sep="\t")
        for _, r in prev.iterrows():
            done[int(r["fold"])] = r.to_dict()
        _log(f"[{cell}] resume: loaded {len(done)} finished folds from {metrics_path}")

    rows = []
    for fold_idx in range(1, dataset.num_of_folds + 1):
        if fold_idx in done:
            rows.append(done[fold_idx])
            _log(f"[{cell}] fold {fold_idx}: skip (already in fold_metrics.tsv)")
            continue

        dataset.prepare_fold(fold_idx)
        fold_log = cell_out / f"fold_{fold_idx}"
        fold_log.mkdir(parents=True, exist_ok=True)
        rp_path = fold_log / "response_prediction.pt"
        pd_path = fold_log / "perturbation_discovery.pt"
        have_ckpt = rp_path.exists() and pd_path.exists()

        model_tmp = deepcopy(model)
        if have_ckpt:
            _log(f"[{cell}] fold {fold_idx}/{dataset.num_of_folds}: eval existing checkpoints")
        else:
            trainer.logging_paths(path=str(fold_log), name="")
            _log(f"[{cell}] fold {fold_idx}/{dataset.num_of_folds} training...")
            trainer.train(model_tmp, dataset, n_epochs=n_epochs)

        rp = torch.load(rp_path, map_location="cpu", weights_only=False)
        pd_ckpt = torch.load(pd_path, map_location="cpu", weights_only=False)
        model_tmp.response_prediction.load_state_dict(rp["model_state_dict"])
        model_tmp.perturbation_discovery.load_state_dict(pd_ckpt["model_state_dict"])
        del rp, pd_ckpt
        metrics = eval_test_metrics(model_tmp, dataset, torch.device("cuda"))
        metrics.update(
            {
                "cell": cell,
                "fold": fold_idx,
                "n_layers_gnn": n_layers_gnn,
                "trainer_test_topk": None,
            }
        )
        rows.append(metrics)
        _log(
            f"[{cell}] fold {fold_idx}: recall@1={metrics['recall@1']:.4f} "
            f"recall@10={metrics['recall@10']:.4f} partial%={metrics['pct_partially_accurate']:.2f} "
            f"ndcg={metrics['ndcg']:.4f}"
        )
        pd.DataFrame(rows).to_csv(metrics_path, sep="\t", index=False)
        del model_tmp
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, sep="\t", index=False)
    return df


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
        "--out_dir",
        type=Path,
        default=_ROOT / "reverse/results/pdgrapher_official_genetic",
    )
    p.add_argument("--cells", nargs="+", default=["ES2", "A549"])
    p.add_argument("--n_layers_gnn", type=int, default=2)
    p.add_argument("--n_epochs", type=int, default=50)
    args = p.parse_args()

    frames = []
    for cell in args.cells:
        frames.append(
            run_cell(
                cell,
                args.data_dir,
                args.splits_dir,
                args.out_dir,
                n_layers_gnn=args.n_layers_gnn,
                n_epochs=args.n_epochs,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    # Merge with any previously finished cells (e.g. ES2) in out_dir.
    cell_files = sorted(args.out_dir.glob("*/n_gnn_*/fold_metrics.tsv"))
    all_df = pd.concat([pd.read_csv(p, sep="\t") for p in cell_files], ignore_index=True)
    all_df.to_csv(args.out_dir / "all_fold_metrics.tsv", sep="\t", index=False)
    summary = (
        all_df.groupby("cell")[
            ["recall@1", "recall@10", "recall@100", "pct_partially_accurate", "ndcg"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(args.out_dir / "summary_by_cell.tsv", sep="\t")
    _log(summary.to_string())
    _log(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
