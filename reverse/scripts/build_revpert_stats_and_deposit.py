#!/usr/bin/env python3
"""Build RevPert statistics, PDGrapher fairness audit, and anonymous deposit pack.

1) Essential seed-1 per-query ranks (RevPert vs Pearson) + Wilcoxon/bootstrap
2) Genetic protocol fairness audit table (matched vs non-matched dimensions)
3) Anonymous reproducible deposit under reverse/deposit/anonymous_revpert/

Does not retrain models; reloads frozen RevPert checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.scripts.run_gallery_dual_g2 import (  # noqa: E402
    GalleryDual,
    eval_identity,
    pearson_matrix,
    score_learned_np,
)
from reverse.src.cell_lines import resolve_cell_paths  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_prediction_dir,
)
from reverse.src.reverse_data import load_reverse_bundle  # noqa: E402
from reverse.src.reverse_model import PCAProjector  # noqa: E402
from reverse.src.summarize_fair_all import (  # noqa: E402
    bootstrap_median_diff,
    wilcoxon_dual_vs_pearson,
)

REV = _ROOT / "reverse/results/revpert"
CELLS = ["hepg2", "k562", "rpe1", "jurkat"]
LABEL = {"hepg2": "HepG2", "k562": "K562", "rpe1": "RPE1", "jurkat": "Jurkat"}
GEN_CELLS = [
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_essential_seed1_ranks(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reload checkpoints; write per-query ranks and Wilcoxon/bootstrap table."""
    out_ranks = REV / "essential" / "seed1_per_query_ranks"
    out_ranks.mkdir(parents=True, exist_ok=True)
    tables = REV / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (REV / "figures_source").mkdir(parents=True, exist_ok=True)

    all_rows = []
    stats_rows = []
    for cell in CELLS:
        ckpt_path = REV / "essential" / cell / "seed1" / "gallery_dual_best.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        paths = resolve_cell_paths(cell, seed=1)
        genes, tables_bundle, _meta = load_reverse_bundle(
            Path(paths["pseudobulk_deltas"]),
            Path(paths["pred_dir"]) / "gene_names.json",
            Path(paths["split"]),
            Path(paths["p_tsv"]),
        )
        genes2, pred_abs = load_prediction_dir(Path(paths["pred_dir"]))
        assert genes2 == genes
        ctrl = load_ctrl_from_perturb_processed(Path(paths["dataset_h5ad"]), genes)
        pred_gal = absolute_to_delta(pred_abs, ctrl)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        gal_kos = list(ckpt["gal_kos"])
        gal_mat = np.stack(
            [np.nan_to_num(pred_gal[k], nan=0.0) for k in gal_kos], axis=0
        ).astype(np.float32)

        pca = PCAProjector(n_components=int(ckpt["pca"]["n_components"]))
        pca.load_state_dict(ckpt["pca"])
        model = GalleryDual(
            pca,
            emb_dim=int(ckpt["args"].get("emb_dim", 128)),
            hidden=int(ckpt["args"].get("hidden", 512)),
            dropout=float(ckpt["args"].get("dropout", 0.1)),
            shared=bool(ckpt["args"].get("shared", 1)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        alpha = float(ckpt["alpha_star"])

        Yte = np.nan_to_num(tables_bundle["test"]["Y"], nan=0.0).astype(np.float32)
        Kte = list(tables_bundle["test"]["kos"])
        pear_te = pearson_matrix(Yte, gal_mat)
        learn_te = score_learned_np(model, Yte, gal_mat, device)
        idx = {k: i for i, k in enumerate(gal_kos)}
        keep = [i for i, k in enumerate(Kte) if k in idx]
        kos_k = [Kte[i] for i in keep]

        r_p, sm_p = eval_identity(Yte, Kte, gal_kos, gal_mat, pear_te, learn_te, 0.0)
        r_r, sm_r = eval_identity(Yte, Kte, gal_kos, gal_mat, pear_te, learn_te, alpha)
        assert len(r_p) == len(kos_k) == len(r_r)

        df = pd.DataFrame(
            {
                "true_ko": kos_k,
                "rank_pearson": r_p.astype(int),
                "rank_revpert": r_r.astype(int),
                "cell": cell,
                "cell_line": LABEL[cell],
                "seed": 1,
                "alpha_star": alpha,
                "n_catalog": len(gal_kos),
            }
        )
        df.to_csv(out_ranks / f"{cell}_seed1_ranks.tsv", sep="\t", index=False)
        all_rows.append(df)

        a = df["rank_revpert"].to_numpy(float)
        b = df["rank_pearson"].to_numpy(float)
        boot = bootstrap_median_diff(a, b, n_boot=2000, seed=0)
        # rename dual→revpert for clarity
        boot_named = {
            "median_rank_revpert": boot["median_rank_dual"],
            "median_rank_pearson": boot["median_rank_pearson"],
            "median_diff_revpert_minus_pearson": boot["median_diff_dual_minus_pearson"],
            "boot_mean_diff": boot["boot_mean_diff"],
            "boot_ci95_low": boot["boot_ci95_low"],
            "boot_ci95_high": boot["boot_ci95_high"],
            "n_query": boot["n_query"],
            "n_boot": boot["n_boot"],
        }
        wil = wilcoxon_dual_vs_pearson(a, b)
        wil_named = {
            "wilcoxon_test": wil["test"],
            "wilcoxon_alternative": "pearson_rank > revpert_rank (RevPert better)",
            "wilcoxon_statistic": wil.get("statistic", float("nan")),
            "wilcoxon_pvalue": wil["pvalue"],
            "wilcoxon_n_revpert_better": wil["n_dual_better"],
            "wilcoxon_n_pearson_better": wil["n_pearson_better"],
            "wilcoxon_n_tie": wil["n_tie"],
        }
        stats_rows.append({"cell_line": LABEL[cell], **boot_named, **wil_named})
        _log(
            f"[{LABEL[cell]}] n={len(df)} Pearson med={sm_p['median_rank']:.1f} "
            f"RevPert med={sm_r['median_rank']:.1f} α={alpha:.3f} "
            f"p={wil_named['wilcoxon_pvalue']:.2e}"
        )

    ranks_all = pd.concat(all_rows, ignore_index=True)
    ranks_all.to_csv(out_ranks / "all_seed1_ranks.tsv", sep="\t", index=False)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(tables / "essential_wilcoxon_bootstrap.tsv", sep="\t", index=False)
    stats.to_csv(
        REV / "figures_source" / "essential_wilcoxon_bootstrap.tsv", sep="\t", index=False
    )
    (tables / "essential_wilcoxon_bootstrap.json").write_text(
        json.dumps(stats_rows, indent=2)
    )
    return ranks_all, stats


def build_pdgrapher_fairness_audit() -> pd.DataFrame:
    """Document matched / unmatched protocol dimensions for Resource B."""
    tables = REV / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    gen_sum = pd.read_csv(REV / "genetic" / "summary_by_cell.tsv", sep="\t")
    fuse = gen_sum[gen_sum["method"] == "gallery_fuse"].copy()
    old = pd.read_csv(
        _ROOT
        / "reverse/results/pdgrapher_genetic_task1_unified/main_table_median_rank_r10.tsv",
        sep="\t",
    )
    pdg = old[old["method"] == "pdgrapher_official"].set_index("cell")
    pear = gen_sum[gen_sum["method"] == "pearson_train_gallery"].set_index("cell")

    # Protocol checklist (shared for all lines)
    protocol = pd.DataFrame(
        [
            {
                "dimension": "Data resource",
                "matched": "yes",
                "detail": "PDGrapher genetic LINCS L1000 screens (10 lines); not Perturb-seq",
            },
            {
                "dimension": "Splits",
                "matched": "yes",
                "detail": "Official within-line random 5-fold genetic splits (backward indices)",
            },
            {
                "dimension": "Primary metrics",
                "matched": "yes",
                "detail": "Median rank of true intervention + Recall@10 on the same ranking interface",
            },
            {
                "dimension": "Query definition",
                "matched": "yes",
                "detail": "ΔY = treated − diseased on the 10,716-gene PPI-intersected axis",
            },
            {
                "dimension": "Catalog restriction",
                "matched": "yes",
                "detail": "Within each fold, genes restricted to those available to all compared methods before ranking",
            },
            {
                "dimension": "Gallery for matching baselines / RevPert",
                "matched": "yes",
                "detail": "Train-fold mean ΔY by intervention gene (no held-out leakage into gallery)",
            },
            {
                "dimension": "Official PDGrapher training recipe",
                "matched": "partial",
                "detail": "Published genetic model / checkpoints scored under the identity-recovery ranking interface; training recipe unchanged beyond runtime stability settings",
            },
            {
                "dimension": "Model class / objective",
                "matched": "no",
                "detail": "RevPert is gallery-native residual retrieval; PDGrapher is a GNN perturbation-discovery model — fair on ranking metrics, not architecture-matched",
            },
            {
                "dimension": "Pooled leaderboard with Essential",
                "matched": "no",
                "detail": "Resources never pooled; Essential primary baseline is Pearson, genetic primary baseline is official PDGrapher",
            },
        ]
    )
    protocol.to_csv(tables / "pdgrapher_fairness_protocol.tsv", sep="\t", index=False)

    rows = []
    for cell in GEN_CELLS:
        fr = float(fuse.loc[fuse.cell == cell, "median_rank_mean"].iloc[0])
        pr = float(pear.loc[cell, "median_rank_mean"])
        pg = float(pdg.loc[cell, "median_rank_mean"])
        fr10 = float(fuse.loc[fuse.cell == cell, "recall_at_10_mean"].iloc[0])
        pg10 = float(pdg.loc[cell, "recall_at_10_mean"])
        rows.append(
            {
                "cell": cell,
                "n_folds": 5,
                "gene_axis": 10716,
                "split": "official_random_5fold",
                "revpert_median_rank_mean": fr,
                "pearson_median_rank_mean": pr,
                "pdgrapher_official_median_rank_mean": pg,
                "fold_vs_pdgrapher": pg / max(fr, 1e-9),
                "fold_vs_pearson": pr / max(fr, 1e-9),
                "revpert_recall10_mean": fr10,
                "pdgrapher_recall10_mean": pg10,
                "revpert_better_median_than_pdgrapher": fr < pg,
                "revpert_better_median_than_pearson": fr < pr,
            }
        )
    per = pd.DataFrame(rows)
    per.to_csv(tables / "pdgrapher_fairness_perline.tsv", sep="\t", index=False)

    summary = {
        "n_lines": int(len(per)),
        "revpert_beats_pdgrapher_median": int(per["revpert_better_median_than_pdgrapher"].sum()),
        "revpert_beats_pearson_median": int(per["revpert_better_median_than_pearson"].sum()),
        "fold_vs_pdgrapher_min": float(per["fold_vs_pdgrapher"].min()),
        "fold_vs_pdgrapher_max": float(per["fold_vs_pdgrapher"].max()),
        "fold_vs_pdgrapher_median": float(per["fold_vs_pdgrapher"].median()),
        "protocol_tsv": "tables/pdgrapher_fairness_protocol.tsv",
        "perline_tsv": "tables/pdgrapher_fairness_perline.tsv",
        "claim_boundary": (
            "Comparison is fair on splits, gene axis restriction, ΔY definition and "
            "ranking metrics; model classes differ (gallery residual scorer vs GNN discovery)."
        ),
    }
    (tables / "pdgrapher_fairness_summary.json").write_text(json.dumps(summary, indent=2))
    _log(
        f"Fairness: RevPert < PDGrapher on {summary['revpert_beats_pdgrapher_median']}/10; "
        f"fold range {summary['fold_vs_pdgrapher_min']:.1f}–{summary['fold_vs_pdgrapher_max']:.1f}×"
    )
    return per


def build_anonymous_deposit(stats: pd.DataFrame) -> Path:
    """Stage an anonymous, double-blind-safe deposit tree with checksums."""
    deposit = _ROOT / "reverse/deposit/anonymous_revpert"
    if deposit.exists():
        shutil.rmtree(deposit)
    for sub in ("tables", "figures_source", "essential_ranks", "genetic", "scripts", "docs"):
        (deposit / sub).mkdir(parents=True, exist_ok=True)

    copies = [
        (REV / "tables" / "essential_wilcoxon_bootstrap.tsv", deposit / "tables" / "essential_wilcoxon_bootstrap.tsv"),
        (REV / "tables" / "essential_fair_median_rank.tsv", deposit / "tables" / "essential_fair_median_rank.tsv"),
        (REV / "tables" / "fourteen_dataset_board.tsv", deposit / "tables" / "fourteen_dataset_board.tsv"),
        (REV / "tables" / "genetic_main_table.tsv", deposit / "tables" / "genetic_main_table.tsv"),
        (REV / "tables" / "leaderboard_snapshot.json", deposit / "tables" / "leaderboard_snapshot.json"),
        (REV / "tables" / "pdgrapher_fairness_protocol.tsv", deposit / "tables" / "pdgrapher_fairness_protocol.tsv"),
        (REV / "tables" / "pdgrapher_fairness_perline.tsv", deposit / "tables" / "pdgrapher_fairness_perline.tsv"),
        (REV / "tables" / "pdgrapher_fairness_summary.json", deposit / "tables" / "pdgrapher_fairness_summary.json"),
        (REV / "essential" / "identity_all.tsv", deposit / "tables" / "essential_identity_all.tsv"),
        (REV / "essential" / "identity_multiseed_mean_std.tsv", deposit / "tables" / "essential_identity_multiseed_mean_std.tsv"),
        (REV / "genetic" / "summary_by_cell.tsv", deposit / "genetic" / "summary_by_cell.tsv"),
        (REV / "genetic" / "all_fold_metrics.tsv", deposit / "genetic" / "all_fold_metrics.tsv"),
        (REV / "genetic" / "pdgrapher_metrics_macro.tsv", deposit / "genetic" / "pdgrapher_metrics_macro.tsv"),
        (
            REV / "essential" / "seed1_per_query_ranks" / "all_seed1_ranks.tsv",
            deposit / "essential_ranks" / "all_seed1_ranks.tsv",
        ),
        (
            _ROOT / "reverse/scripts/build_revpert_stats_and_deposit.py",
            deposit / "scripts" / "build_revpert_stats_and_deposit.py",
        ),
        (
            _ROOT / "reverse/scripts/build_revpert_manuscript_tables.py",
            deposit / "scripts" / "build_revpert_manuscript_tables.py",
        ),
        (
            _ROOT / "reverse/benchmark/METHODS_REGISTRY.md",
            deposit / "docs" / "METHODS_REGISTRY.md",
        ),
        (
            _ROOT / "reverse/manuscript/TASK1_PROTOCOL.md",
            deposit / "docs" / "TASK1_PROTOCOL.md",
        ),
    ]
    for cell in CELLS:
        src = REV / "essential" / "seed1_per_query_ranks" / f"{cell}_seed1_ranks.tsv"
        if src.is_file():
            copies.append((src, deposit / "essential_ranks" / f"{cell}_seed1_ranks.tsv"))

    for src, dst in copies:
        if not src.is_file():
            _log(f"[warn] missing {src}")
            continue
        shutil.copy2(src, dst)

    readme = deposit / "README.md"
    readme.write_text(
        """# Anonymous RevPert reproducibility deposit

Double-blind peer-review package for **RevPert** (gallery-native residual reverse
perturbation scoring). Author names and institutional paths are withheld.

## Contents

| Path | Description |
|------|-------------|
| `tables/essential_wilcoxon_bootstrap.tsv` | Seed-1 paired Wilcoxon + bootstrap CI (RevPert vs Pearson) |
| `tables/pdgrapher_fairness_*.tsv` | Protocol audit + per-line fold vs official PDGrapher |
| `tables/fourteen_dataset_board.tsv` | Main 14-dataset board (Essential + genetic) |
| `essential_ranks/` | Per-query ranks for Essential seed 1 |
| `genetic/` | Fold metrics and cell-line summaries |
| `scripts/` | Table / stats builders (no private credentials) |
| `docs/` | Method registry and Task-1 protocol |

## Recompute Essential Wilcoxon (requires full analysis environment)

```bash
# From the full analysis repository (not this deposit alone):
PYTHONPATH=. python reverse/scripts/build_revpert_stats_and_deposit.py
```

This deposit ships the **frozen numeric outputs** used in the manuscript SI.
Model checkpoints are not deposited here (stated in Data availability).

## Claim boundaries

- Essential primary baseline: Pearson on linear L3 predicted gallery.
- Genetic primary baseline: official PDGrapher genetic model under the same
  median-rank / Recall@10 interface.
- Resources are **not** pooled into one leaderboard.
- PDGrapher comparison is fair on splits/metrics/catalog restriction; model
  classes differ (residual gallery scorer vs GNN discovery).

## Checksums

See `SHA256SUMS` in this directory.
""",
        encoding="utf-8",
    )

    # SHA256 of all deposited files except the sums file itself
    sums = []
    for path in sorted(deposit.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"SHA256SUMS", ".DS_Store"}:
            continue
        rel = path.relative_to(deposit).as_posix()
        sums.append(f"{_sha256(path)}  {rel}")
    (deposit / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    # Tar.gz for upload
    archive = _ROOT / "reverse/deposit/anonymous_revpert_v1"
    shutil.make_archive(str(archive), "gztar", root_dir=deposit.parent, base_dir=deposit.name)
    tar = Path(str(archive) + ".tar.gz")
    (deposit / "ARCHIVE_SHA256.txt").write_text(
        f"{_sha256(tar)}  {tar.name}\n", encoding="utf-8"
    )
    _log(f"Deposit: {deposit}")
    _log(f"Archive: {tar} ({tar.stat().st_size/1e6:.2f} MB)")
    return deposit


def latex_wilcoxon_snippet(stats: pd.DataFrame) -> str:
    """Emit a ready-to-paste tabular body for tab:wilcox."""
    lines = []
    for _, r in stats.iterrows():
        p = float(r["wilcoxon_pvalue"])
        if p < 1e-4:
            mant, exp = f"{p:.1e}".split("e")
            p_s = f"{mant}\\times10^{{{int(exp)}}}"
        else:
            p_s = f"{p:.4f}"
        ci = f"$[{r['boot_ci95_low']:.1f},{r['boot_ci95_high']:.1f}]$"
        lines.append(
            f"{r['cell_line']} & {r['median_rank_revpert']:g} & {r['median_rank_pearson']:g} & "
            f"{r['median_diff_revpert_minus_pearson']:g} & {ci} & {int(r['n_query'])} & "
            f"${p_s}$ & {int(r['wilcoxon_n_revpert_better'])}/{int(r['n_query'])} \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device={device}")
    _ranks, stats = export_essential_seed1_ranks(device)
    build_pdgrapher_fairness_audit()
    deposit = build_anonymous_deposit(stats)
    snip = latex_wilcoxon_snippet(stats)
    (REV / "tables" / "wilcoxon_latex_rows.txt").write_text(snip + "\n", encoding="utf-8")
    _log("\n=== Wilcoxon LaTeX rows ===\n" + snip)
    _log(f"\nDone. Deposit root: {deposit}")


if __name__ == "__main__":
    main()
