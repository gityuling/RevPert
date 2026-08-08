#!/usr/bin/env python3
"""Curate Task-1 tables and export publication figures (Python / matplotlib).

Figures
-------
Fig.1  Task-1 schematic
Fig.2  Fair method comparison (median rank + Top-10)
Fig.3  Gallery-builder ablation (incl. scGPT; same catalog)
Fig.4  Multi-seed stability (dual vs Pearson, mean±s.d.)

Source data are written under ``results/tables_task1/``; figures under
``results/figures_task1/`` (symlinked into ``manuscript/figures/``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS = _ROOT / "reverse/results"
FIG_OUT = RESULTS / "figures_task1"
TAB_OUT = RESULTS / "tables_task1"
MS_FIG = _ROOT / "reverse/manuscript/figures"

CELLS = ["HepG2", "K562", "RPE1", "Jurkat"]
CELL_COLORS = {
    "HepG2": "#0072B2",
    "K562": "#E69F00",
    "RPE1": "#009E73",
    "Jurkat": "#D55E00",
}

FAIR_METHODS = [
    "dual_encoder_v2",
    "pearson_pred_gallery",
    "ridge_delta_to_P",
    "gem_lite_topDEG_corr",
    "cmap_lite_pred_gallery",
]
FAIR_LABELS = {
    "dual_encoder_v2": "Dual-encoder v2",
    "pearson_pred_gallery": "Pearson\npred. gallery",
    "ridge_delta_to_P": "Ridge\nΔY→P",
    "gem_lite_topDEG_corr": "GEM-lite",
    "cmap_lite_pred_gallery": "CMap-lite",
}

GALLERY_METHODS = [
    "dual_encoder_v2",
    "txpert_gat_gallery_pearson",
    "linear_L3_gallery_pearson",
    "txpert_xcell_gallery_pearson",
    "gears_gallery_pearson",
    "scgpt_gallery_pearson",
]
GALLERY_LABELS = {
    "dual_encoder_v2": "Dual-encoder",
    "txpert_gat_gallery_pearson": "TxPert-GAT\ngallery",
    "linear_L3_gallery_pearson": "Linear L3\ngallery",
    "txpert_xcell_gallery_pearson": "TxPert x-cell\ngallery",
    "gears_gallery_pearson": "GEARS\ngallery",
    "scgpt_gallery_pearson": "scGPT\ngallery",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


def _symlink_ms(name: str) -> None:
    MS_FIG.mkdir(parents=True, exist_ok=True)
    src = FIG_OUT / f"{name}.pdf"
    dst = MS_FIG / f"{name}.pdf"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(Path("../../results/figures_task1") / f"{name}.pdf")


def curate_tables() -> dict[str, pd.DataFrame]:
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    fair = pd.read_csv(RESULTS / "fair_compare_all/summary.tsv", sep="\t")
    fair = fair[fair["method"].isin(FAIR_METHODS)].copy()
    fair["cell_line"] = pd.Categorical(fair["cell_line"], CELLS, ordered=True)
    fair = fair.sort_values(["method", "cell_line"])
    keep = [
        "cell_line",
        "method",
        "n_query",
        "median_rank",
        "mean_rank",
        "pct_top1",
        "pct_top10",
        "pct_top50",
        "pct_top100",
        "mrr",
    ]
    tables["fair_seed1"] = fair[keep]
    fair[keep].to_csv(TAB_OUT / "table_fair_seed1_metrics.tsv", sep="\t", index=False)
    fair.pivot_table(index="method", columns="cell_line", values="median_rank").reindex(
        FAIR_METHODS
    )[CELLS].to_csv(TAB_OUT / "table_fair_seed1_median_rank.tsv", sep="\t")
    fair.pivot_table(index="method", columns="cell_line", values="pct_top10").reindex(
        FAIR_METHODS
    )[CELLS].to_csv(TAB_OUT / "table_fair_seed1_top10.tsv", sep="\t")

    stats = pd.read_csv(RESULTS / "fair_compare_all/dual_vs_pearson_stats.tsv", sep="\t")
    tables["dual_vs_pearson"] = stats
    stats.to_csv(TAB_OUT / "table_dual_vs_pearson_stats.tsv", sep="\t", index=False)

    gal = pd.read_csv(RESULTS / "gallery_compare_all/summary.tsv", sep="\t")
    gal = gal[gal["method"].isin(GALLERY_METHODS)].copy()
    gal["cell_line"] = pd.Categorical(gal["cell_line"], CELLS, ordered=True)
    gal = gal.sort_values(["method", "cell_line"])
    tables["gallery"] = gal[keep]
    gal[keep].to_csv(TAB_OUT / "table_gallery_seed1_metrics.tsv", sep="\t", index=False)
    gal.pivot_table(index="method", columns="cell_line", values="median_rank").reindex(
        GALLERY_METHODS
    )[CELLS].to_csv(TAB_OUT / "table_gallery_seed1_median_rank.tsv", sep="\t")

    ms = pd.read_csv(RESULTS / "fair_compare_multiseed/summary_seed_mean_std.tsv", sep="\t")
    tables["multiseed"] = ms
    ms.to_csv(TAB_OUT / "table_multiseed_mean_std.tsv", sep="\t", index=False)

    # Compact leaderboard for README / SI
    board = {
        "fair_median_rank": fair.pivot_table(
            index="method", columns="cell_line", values="median_rank"
        )
        .reindex(FAIR_METHODS)[CELLS]
        .to_dict(),
        "gallery_median_rank": gal.pivot_table(
            index="method", columns="cell_line", values="median_rank"
        )
        .reindex(GALLERY_METHODS)[CELLS]
        .to_dict(),
        "multiseed_dual_median_rank_mean_std": {
            r["cell_line"]: {
                "mean": float(r["dual_median_rank_mean"]),
                "std": float(r["dual_median_rank_std"]),
            }
            for _, r in ms.iterrows()
        },
    }
    (TAB_OUT / "leaderboard_snapshot.json").write_text(json.dumps(board, indent=2))
    (TAB_OUT / "README.md").write_text(
        "\n".join(
            [
                "# Task-1 curated tables",
                "",
                "| File | Content |",
                "|------|---------|",
                "| `table_fair_seed1_metrics.tsv` | Fair methods, all rank metrics (seed-1) |",
                "| `table_fair_seed1_median_rank.tsv` | Wide median rank |",
                "| `table_fair_seed1_top10.tsv` | Wide Top-10 % |",
                "| `table_dual_vs_pearson_stats.tsv` | Wilcoxon + bootstrap CI |",
                "| `table_gallery_seed1_metrics.tsv` | Gallery builders + dual (same catalog) |",
                "| `table_gallery_seed1_median_rank.tsv` | Wide gallery median rank |",
                "| `table_multiseed_mean_std.tsv` | Seeds 1–5 dual vs Pearson |",
                "| `leaderboard_snapshot.json` | Compact JSON for scripts |",
                "",
                "Generated by `python -m reverse.src.plot_task1_figures`.",
                "",
            ]
        )
    )
    return tables


def fig1_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#F7F7F7"):
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#333333", linewidth=1.1, zorder=2)
        )
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontweight="bold", fontsize=8)
        ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=7, color="#333333")

    box(0.3, 0.9, 3.0, 1.8, "Query", "Observed ΔY\nheld-out test KO")
    box(4.2, 0.9, 3.2, 1.8, "Catalog", "Forward gallery ΔŶ(g)\nor learned prototypes P(g)")
    box(8.4, 0.9, 3.2, 1.8, "Recovery", "Rank catalog genes\nrecover true KO g*")
    for x0, x1 in [(3.3, 4.1), (7.4, 8.3)]:
        ax.annotate(
            "",
            xy=(x1, 1.8),
            xytext=(x0, 1.8),
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2),
            zorder=3,
        )
    ax.text(
        6.0,
        0.25,
        "Task-1 · GEARS simulation splits · Essential lines HepG2 / K562 / RPE1 / Jurkat",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    _save(fig, FIG_OUT / "fig1_task1_schematic")
    _symlink_ms("fig1_task1_schematic")


def _grouped_bars(ax, methods, labels, wide, ylabel, log_y=False):
    x = np.arange(len(methods))
    width = 0.18
    for i, cell in enumerate(CELLS):
        if cell not in wide.columns:
            continue
        vals = wide[cell].reindex(methods).to_numpy(dtype=float)
        ax.bar(
            x + (i - (len(CELLS) - 1) / 2) * width,
            vals,
            width=width,
            label=cell,
            color=CELL_COLORS[cell],
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in methods])
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend(ncol=4, loc="upper right", bbox_to_anchor=(1.0, 1.02))


def fig2_fair(tables: dict[str, pd.DataFrame]) -> None:
    fair = tables["fair_seed1"]
    med = fair.pivot_table(index="method", columns="cell_line", values="median_rank").reindex(
        FAIR_METHODS
    )[CELLS]
    top = fair.pivot_table(index="method", columns="cell_line", values="pct_top10").reindex(
        FAIR_METHODS
    )[CELLS]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})
    _grouped_bars(axes[0], FAIR_METHODS, FAIR_LABELS, med, "Median rank (↓ better)")
    axes[0].text(-0.12, 1.05, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10)
    _grouped_bars(axes[1], FAIR_METHODS, FAIR_LABELS, top, "Top-10 recovery (%)")
    axes[1].text(-0.12, 1.05, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)
    axes[1].get_legend().remove()
    _save(fig, FIG_OUT / "fig2_fair_median_rank")
    _symlink_ms("fig2_fair_median_rank")
    med.to_csv(FIG_OUT / "fig2_fair_median_rank_source.tsv", sep="\t")
    top.to_csv(FIG_OUT / "fig2_fair_top10_source.tsv", sep="\t")


def fig3_gallery(tables: dict[str, pd.DataFrame]) -> None:
    gal = tables["gallery"]
    med = gal.pivot_table(index="method", columns="cell_line", values="median_rank").reindex(
        GALLERY_METHODS
    )[CELLS]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    _grouped_bars(
        ax,
        GALLERY_METHODS,
        GALLERY_LABELS,
        med,
        "Median rank (↓ better; log scale)",
        log_y=True,
    )
    _save(fig, FIG_OUT / "fig3_gallery_source_hepg2")
    # Keep alias used by older notes
    shutil.copy(FIG_OUT / "fig3_gallery_source_hepg2.pdf", FIG_OUT / "fig3_gallery_source_four_lines.pdf")
    shutil.copy(FIG_OUT / "fig3_gallery_source_hepg2.png", FIG_OUT / "fig3_gallery_source_four_lines.png")
    _symlink_ms("fig3_gallery_source_hepg2")
    med.to_csv(FIG_OUT / "fig3_gallery_source_four_lines.tsv", sep="\t")


def fig4_multiseed(tables: dict[str, pd.DataFrame]) -> None:
    ms = tables["multiseed"].set_index("cell_line").reindex(CELLS)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": 0.32})
    x = np.arange(len(CELLS))
    w = 0.36

    axes[0].bar(
        x - w / 2,
        ms["dual_median_rank_mean"],
        w,
        yerr=ms["dual_median_rank_std"],
        color="#0072B2",
        label="Dual-encoder",
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    axes[0].bar(
        x + w / 2,
        ms["pearson_median_rank_mean"],
        w,
        yerr=ms["pearson_median_rank_std"],
        color="#999999",
        label="Pearson pred. gallery",
        capsize=2,
        error_kw={"elinewidth": 0.8},
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CELLS)
    axes[0].set_ylabel("Median rank (mean ± s.d.)")
    axes[0].legend(loc="upper right")
    axes[0].text(-0.12, 1.05, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10)

    axes[1].bar(
        x - w / 2,
        ms["dual_top10_mean"],
        w,
        color="#0072B2",
        label="Dual-encoder",
    )
    axes[1].bar(
        x + w / 2,
        ms["pearson_top10_mean"],
        w,
        color="#999999",
        label="Pearson pred. gallery",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CELLS)
    axes[1].set_ylabel("Top-10 recovery (%, mean)")
    axes[1].text(-0.12, 1.05, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)

    _save(fig, FIG_OUT / "fig4_multiseed_stability")
    _symlink_ms("fig4_multiseed_stability")
    ms.to_csv(FIG_OUT / "fig4_multiseed_source.tsv", sep="\t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_plot", action="store_true")
    args = ap.parse_args()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    tables = curate_tables()
    print(f"Tables → {TAB_OUT}")
    if args.skip_plot:
        return
    fig1_schematic()
    fig2_fair(tables)
    fig3_gallery(tables)
    fig4_multiseed(tables)
    print(f"Figures → {FIG_OUT}")
    print("Manuscript symlinks →", MS_FIG)


if __name__ == "__main__":
    main()
