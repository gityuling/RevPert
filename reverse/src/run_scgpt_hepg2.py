#!/usr/bin/env python3
"""Fine-tune scGPT on HepG2 Essential (seed1 split) and export forward gallery.

Requires conda env ``scgpt_env`` (scgpt + cell-gears 0.1.2).
Pretrained checkpoint: reverse/models/scGPT_human/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

try:
    import torchtext

    torchtext.disable_torchtext_deprecation_warning()
except Exception:
    pass
warnings.filterwarnings("ignore", message="flash_attn is not installed")

import scgpt as scg
from gears import PertData
from gears.utils import create_cell_graph_dataset_for_prediction
from scgpt.loss import masked_mse_loss, masked_relative_error
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import map_raw_id_to_vocab_id, set_seed

_ROOT = Path(__file__).resolve().parents[2]
BENCH = _ROOT / "linear_perturbation_prediction-Paper-main" / "benchmark"
DEFAULT_MODEL = _ROOT / "reverse" / "models" / "scGPT_human"
DEFAULT_SPLIT = BENCH / "working_dir" / "results" / "seed_1_replogle_hepg2_essential_split"
DEFAULT_GENE_NAMES = (
    BENCH
    / "working_dir"
    / "results"
    / "matched_ml_baselines"
    / "replogle_hepg2_essential__gears_seed1"
    / "gene_names.json"
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="replogle_hepg2_essential")
    ap.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT))
    ap.add_argument("--gene_names_json", type=str, default=str(DEFAULT_GENE_NAMES))
    ap.add_argument("--load_model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--pool_size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_seq_len", type=int, default=1536)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--max_cells_per_condition",
        type=int,
        default=0,
        help="Cap cells per condition before building graphs (0 = no cap). Use ~30 for Jurkat.",
    )
    ap.add_argument("--smoke", action="store_true", help="1 epoch, tiny pool, few predicts")
    return ap.parse_args()


def _subsample_adata_by_condition(adata, max_cells: int, seed: int = 1):
    """Cap cells per perturbation condition to reduce pyg / dataloader memory."""
    if max_cells <= 0:
        return adata
    rng = np.random.default_rng(seed)
    keep_chunks: list[np.ndarray] = []
    conditions = adata.obs["condition"].astype(str).to_numpy()
    for cond in np.unique(conditions):
        idx = np.flatnonzero(conditions == cond)
        if idx.size <= max_cells:
            keep_chunks.append(idx)
        else:
            keep_chunks.append(rng.choice(idx, size=max_cells, replace=False))
    keep = np.sort(np.concatenate(keep_chunks))
    out = adata[keep].copy()
    print(
        f"Subsampled {adata.n_obs} -> {out.n_obs} cells "
        f"(max {max_cells} cells/condition, seed={seed})",
        flush=True,
    )
    return out


def _fix_gene_names(pert_data: PertData, gene_names_json: Path) -> list[str]:
    genes = json.loads(Path(gene_names_json).read_text())
    if len(genes) != pert_data.adata.n_vars:
        raise ValueError(
            f"gene_names length {len(genes)} != n_vars {pert_data.adata.n_vars}"
        )
    pert_data.adata.var["gene_name"] = genes
    pert_data.gene_names = pert_data.adata.var["gene_name"]
    return genes


def _filter_good_conditions(pert_data: PertData) -> list[str]:
    col = pert_data.adata.obs["condition"]
    if str(col.dtype) == "category":
        conds = col.cat.remove_unused_categories().cat.categories.astype(str).tolist()
    else:
        conds = sorted(set(col.astype(str)))
    gene_names = list(pert_data.adata.var["gene_name"].astype(str).values) + ["ctrl"]
    gene_set = set(gene_names)
    good = []
    for c in conds:
        parts = c.split("+")
        if len(parts) == 1 or (parts[0] in gene_set and parts[1] in gene_set):
            good.append(c)
    return good


def _load_pert_names(pert_data: PertData) -> np.ndarray:
    """GO-graph pert name list used by gears 0.1.x pert_idx."""
    if getattr(pert_data, "pert_names", None) is not None and len(pert_data.pert_names):
        return np.asarray(pert_data.pert_names)
    path_ = os.path.join(pert_data.data_path, "essential_all_data_pert_genes.pkl")
    with open(path_, "rb") as f:
        essential_genes = pickle.load(f)
    with open(os.path.join(pert_data.data_path, "gene2go_all.pkl"), "rb") as f:
        lookup_gene2go = pickle.load(f)
    gene2go = {i: lookup_gene2go[i] for i in essential_genes if i in lookup_gene2go}
    return np.unique(list(gene2go.keys()))


def _remap_flash_attn_keys(state: dict) -> dict:
    return {k.replace("Wqkv.", "in_proj_"): v for k, v in state.items()}


def _build_pred_graphs(pert, ctrl_adata, gene_list, device, pool_size):
    """Build gears-0.0.2-style graphs (x: [n_genes, 2]) that scGPT.pred_perturb expects."""
    graphs = create_cell_graph_dataset_for_prediction(
        pert, ctrl_adata, gene_list, device, num_samples=pool_size
    )
    # 0.1.2 graphs only have expression in x; attach pert flags as column 1
    gene_arr = np.asarray(gene_list)
    pert_idx = [int(np.where(p == gene_arr)[0][0]) for p in pert]
    out = []
    for g in graphs:
        x = g.x.detach().cpu().numpy()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] == 1:
            flags = np.zeros((x.shape[0], 1), dtype=np.float32)
            for pi in pert_idx:
                flags[pi, 0] = 1.0
            x2 = np.concatenate([x, flags], axis=1)
            g.x = torch.tensor(x2, dtype=torch.float32, device=device)
        out.append(g)
    return out


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.pool_size = 4
        args.batch_size = 8
        args.eval_batch_size = 8

    set_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(BENCH)
    logger = scg.logger

    # ---- data ----
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    pad_value = 0
    pert_pad_id = 2
    include_zero_gene = "all"
    max_seq_len = args.max_seq_len
    MLM = True
    CLS = CCE = MVC = ECS = False
    cell_emb_style = "cls"
    mvc_decoder_style = "inner product, detach"
    amp = True
    load_param_prefixs = ["encoder", "value_encoder", "transformer_encoder"]
    use_fast_transformer = False  # no flash-attn
    dropout = 0.2
    schedule_interval = 1
    early_stop = args.patience
    epochs = args.epochs
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    lr = args.lr

    pert_data = PertData(Path("data/gears_pert_data/"))
    data_path = Path("data/gears_pert_data") / args.dataset_name
    max_cells = int(args.max_cells_per_condition)

    if max_cells > 0:
        # Avoid loading the full-cell pyg cache into RAM (Jurkat OOM).
        import scanpy as sc
        from gears.utils import filter_pert_in_go, print_sys

        logger.info(
            f"Low-memory load: max_cells_per_condition={max_cells} "
            f"(skip full cell_graphs.pkl)"
        )
        pert_data.dataset_name = args.dataset_name
        pert_data.dataset_path = str(data_path)
        pert_data.adata = sc.read_h5ad(data_path / "perturb_processed.h5ad")
        genes = _fix_gene_names(pert_data, Path(args.gene_names_json))
        pert_data.set_pert_genes()
        print_sys(
            "These perturbations are not in the GO graph and their "
            "perturbation can thus not be predicted"
        )
        not_in_go = np.array(
            pert_data.adata.obs[
                pert_data.adata.obs.condition.apply(
                    lambda x: not filter_pert_in_go(x, pert_data.pert_names)
                )
            ].condition.unique()
        )
        print_sys(not_in_go)
        filter_go = pert_data.adata.obs[
            pert_data.adata.obs.condition.apply(
                lambda x: filter_pert_in_go(x, pert_data.pert_names)
            )
        ]
        pert_data.adata = pert_data.adata[filter_go.index.values, :].copy()
        pert_data.adata = _subsample_adata_by_condition(
            pert_data.adata, max_cells, seed=args.seed
        )
        # Dense AnnData rows yield 1-D ArrayView.toarray(); GEARS create_cell_graph
        # then stores x as [n_genes] and training's x[:, 0] crashes. CSR rows
        # produce x as [n_genes, 1], matching the full pyg cache format.
        import scipy.sparse as sp

        if not sp.issparse(pert_data.adata.X):
            pert_data.adata.X = sp.csr_matrix(pert_data.adata.X)
        pert_data.ctrl_adata = pert_data.adata[
            pert_data.adata.obs["condition"] == "ctrl"
        ].copy()
        pert_data.gene_names = pert_data.adata.var["gene_name"]
        logger.info("Building cell graphs from subsampled AnnData ...")
        pert_data.create_dataset_file()
        logger.info(
            f"Built graphs for {len(pert_data.dataset_processed)} conditions "
            f"({pert_data.adata.n_obs} cells)"
        )
    else:
        pert_data.load(data_path=str(data_path))
        genes = _fix_gene_names(pert_data, Path(args.gene_names_json))

    good_conds = set(_filter_good_conditions(pert_data))
    pert_data.adata = pert_data.adata[
        [c in good_conds for c in pert_data.adata.obs["condition"].astype(str)], :
    ].copy()
    # refresh ctrl after filter
    pert_data.ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]

    with open(args.split_json) as f:
        set2conditions = json.load(f)
    for split_name in ("train", "val", "test"):
        set2conditions[split_name] = [
            c for c in set2conditions[split_name] if c in good_conds
        ]
    logger.info(
        f"Filtered split sizes train/val/test = "
        f"{len(set2conditions['train'])}/{len(set2conditions['val'])}/{len(set2conditions['test'])}"
    )
    if args.smoke:
        set2conditions["train"] = set2conditions["train"][:32]
        set2conditions["val"] = set2conditions["val"][:8]
        set2conditions["test"] = set2conditions["test"][:8]

    # Keep only graph keys we need (and ctrl)
    keep = set(sum((set2conditions[s] for s in ("train", "val", "test")), [])) | {"ctrl"}
    pert_data.dataset_processed = {
        k: v for k, v in pert_data.dataset_processed.items() if k in keep
    }

    pert_data.set2conditions = set2conditions
    pert_data.split = "custom"
    pert_data.subgroup = None
    pert_data.seed = args.seed
    pert_data.train_gene_set_size = 0.75
    pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

    pert_names = _load_pert_names(pert_data)
    gene_name_series = pert_data.adata.var["gene_name"]

    # ---- model ----
    model_dir = Path(args.load_model)
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if g in vocab else -1 for g in pert_data.adata.var["gene_name"]
    ]
    logger.info(
        f"match {int((pert_data.adata.var['id_in_vocab'] >= 0).sum())}/"
        f"{pert_data.adata.n_vars} genes in vocab"
    )

    with open(model_dir / "args.json") as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int
    )
    n_genes = len(genes)
    ntokens = len(vocab)

    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        do_mvc=MVC,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        use_fast_transformer=use_fast_transformer,
    )
    model_dict = model.state_dict()
    pretrained = torch.load(model_dir / "best_model.pt", map_location="cpu")
    pretrained = _remap_flash_attn_keys(pretrained)
    pretrained = {
        k: v
        for k, v in pretrained.items()
        if any(k.startswith(p) for p in load_param_prefixs)
    }
    missing = [k for k in pretrained if k not in model_dict]
    shape_mismatch = [
        k
        for k, v in pretrained.items()
        if k in model_dict and model_dict[k].shape != v.shape
    ]
    logger.info(f"pretrained keys={len(pretrained)} missing={len(missing)} shape_mismatch={len(shape_mismatch)}")
    pretrained = {
        k: v
        for k, v in pretrained.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    model.to(device)

    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def _pert_flags_from_batch(batch_data, batch_size_):
        flags = torch.zeros((batch_size_, n_genes), dtype=torch.long, device=device)
        for idx in range(batch_size_):
            for pi in batch_data.pert_idx[idx]:
                if pi == -1:
                    continue
                pi = int(pi)
                if pi < 0 or pi >= len(pert_names):
                    continue
                pert_name = pert_names[pi]
                hits = np.where(gene_name_series.values == pert_name)[0]
                if len(hits):
                    flags[idx, int(hits[0])] = 1
        return flags

    def train_one_epoch(model_, loader):
        model_.train()
        total_loss = 0.0
        start = time.time()
        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            bs = len(batch_data.y)
            batch_data.to(device)
            x = batch_data.x
            if x.dim() == 1:
                ori = x.view(bs, n_genes)
            else:
                ori = x[:, 0].view(bs, n_genes)
            pert_flags = _pert_flags_from_batch(batch_data, bs)
            target = batch_data.y

            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target[:, input_gene_ids]
            mapped = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(bs, 1)
            src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool)

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model_(
                    mapped,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                output_values = output_dict["mlm_output"]
                masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                loss = criterion(output_values, target_values, masked_positions)

            model_.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if batch % 50 == 0 and batch > 0:
                ms = (time.time() - start) * 1000 / 50
                logger.info(
                    f"| {batch:4d}/{num_batches:4d} batches | ms/batch {ms:5.2f} | "
                    f"loss {total_loss / (batch + 1):5.4f}"
                )
                start = time.time()
        return total_loss / max(num_batches, 1)

    def evaluate(model_, loader):
        model_.eval()
        total_loss = 0.0
        total_error = 0.0
        with torch.no_grad():
            for batch_data in loader:
                bs = len(batch_data.y)
                batch_data.to(device)
                x = batch_data.x
                if x.dim() == 1:
                    ori = x.view(bs, n_genes)
                else:
                    ori = x[:, 0].view(bs, n_genes)
                pert_flags = _pert_flags_from_batch(batch_data, bs)
                target = batch_data.y
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = input_gene_ids[:max_seq_len]
                input_values = ori[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target[:, input_gene_ids]
                mapped = map_raw_id_to_vocab_id(input_gene_ids, gene_ids).repeat(bs, 1)
                src_key_padding_mask = torch.zeros_like(input_values, dtype=torch.bool)
                with torch.cuda.amp.autocast(enabled=amp):
                    output_dict = model_(
                        mapped,
                        input_values,
                        input_pert_flags,
                        src_key_padding_mask=src_key_padding_mask,
                        CLS=CLS,
                        CCE=CCE,
                        MVC=MVC,
                        ECS=ECS,
                    )
                    output_values = output_dict["mlm_output"]
                    masked_positions = torch.ones_like(input_values, dtype=torch.bool)
                    loss = criterion(output_values, target_values, masked_positions)
                total_loss += loss.item()
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item()
        n = max(len(loader), 1)
        return total_loss / n, total_error / n

    best_val = float("inf")
    best_model = copy.deepcopy(model)
    patience = 0
    for epoch in range(epochs):
        t0 = time.time()
        train_one_epoch(model, pert_data.dataloader["train_loader"])
        val_loss, val_mre = evaluate(model, pert_data.dataloader["val_loader"])
        logger.info(
            f"epoch {epoch} time={time.time() - t0:.1f}s val_loss={val_loss:.4f} mre={val_mre:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_model = copy.deepcopy(model)
            patience = 0
            torch.save(best_model.state_dict(), out_dir / "best_model.pt")
            logger.info(f"new best val_loss={best_val:.4f}")
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(f"early stop at epoch {epoch}")
                break
        scheduler.step()

    # ---- predict gallery ----
    conds = sorted(keep - {"ctrl"})
    if args.smoke:
        conds = conds[:16]
    split_conds = [
        list(filter(lambda y: y != "ctrl", x.split("+"))) for x in conds
    ]
    # drop empty
    pairs = [(c, p) for c, p in zip(conds, split_conds) if p]
    logger.info(f"Predicting {len(pairs)} conditions, pool_size={args.pool_size}")

    gene_list = genes
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    best_model.eval()
    results_pred = {}
    with torch.no_grad():
        for i, (cond, pert) in enumerate(pairs):
            # skip KO genes missing from panel
            if any(g not in gene_list for g in pert):
                continue
            try:
                cell_graphs = _build_pred_graphs(
                    pert, ctrl_adata, gene_list, device, args.pool_size
                )
            except Exception as e:
                logger.info(f"skip {cond}: {e}")
                continue
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred = best_model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred)
            preds = torch.cat(preds, dim=0)
            key = "_".join(pert)
            results_pred[key] = np.mean(preds.detach().cpu().numpy(), axis=0).tolist()
            if (i + 1) % 50 == 0 or i == 0:
                logger.info(f"predicted {i + 1}/{len(pairs)}")

    with open(out_dir / "all_predictions.json", "w", encoding="utf8") as f:
        json.dump(results_pred, f)
    with open(out_dir / "gene_names.json", "w", encoding="utf8") as f:
        json.dump(genes, f, indent=2)
    meta = {
        "dataset": args.dataset_name,
        "split": str(args.split_json),
        "n_pred": len(results_pred),
        "best_val_loss": best_val,
        "epochs": epochs,
        "pool_size": args.pool_size,
        "max_cells_per_condition": int(args.max_cells_per_condition),
        "batch_size": batch_size,
        "filtered_split": {k: len(v) for k, v in set2conditions.items()},
        "smoke": bool(args.smoke),
        "n_cells": int(pert_data.adata.n_obs),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Wrote gallery to {out_dir} n_pred={len(results_pred)}")


if __name__ == "__main__":
    main()
