#!/usr/bin/env python3
"""Aggregate fair_compare across cells and GEARS seeds 1–5."""

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

from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402
from reverse.src.summarize_fair_all import bootstrap_median_diff, wilcoxon_dual_vs_pearson  # noqa: E402

MAIN_METHODS = [
    "dual_encoder_v2",
    "pearson_pred_gallery",
    "cmap_lite_pred_gallery",
    "gem_lite_topDEG_corr",
    "ridge_delta_to_P",
    "esm_mean_topDEG_to_KO",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list_cells())
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--out_dir", type=str, default=str(_ROOT / "reverse/results/fair_compare_multiseed"))
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    stats_rows = []
    for cell in args.cells:
        for seed in args.seeds:
            paths = resolve_cell_paths(cell, seed=seed)
            fair = Path(paths["fair_out"])
            sum_path = fair / "fair_compare_summary.tsv"
            ranks_path = fair / "all_method_ranks.tsv"
            if not sum_path.is_file():
                print(f"[skip] missing {sum_path}")
                continue
            sdf = pd.read_csv(sum_path, sep="\t")
            sdf["cell_line"] = paths["label"]
            sdf["seed"] = seed
            rows.append(sdf)

            if ranks_path.is_file():
                rdf = pd.read_csv(ranks_path, sep="\t")
                dual = rdf.loc[rdf["method"] == "dual_encoder_v2", ["true_ko", "rank"]].rename(
                    columns={"rank": "rank_dual"}
                )
                pear = rdf.loc[rdf["method"] == "pearson_pred_gallery", ["true_ko", "rank"]].rename(
                    columns={"rank": "rank_pearson"}
                )
                m = dual.merge(pear, on="true_ko")
                a = m["rank_dual"].to_numpy(dtype=float)
                b = m["rank_pearson"].to_numpy(dtype=float)
                boot = bootstrap_median_diff(a, b, n_boot=args.n_boot, seed=seed)
                wil = wilcoxon_dual_vs_pearson(a, b)
                stats_rows.append(
                    {
                        "cell_line": paths["label"],
                        "seed": seed,
                        **boot,
                        **{f"wilcoxon_{k}": v for k, v in wil.items()},
                    }
                )

    if not rows:
        raise SystemExit("No fair_compare summaries found.")

    all_sum = pd.concat(rows, ignore_index=True)
    all_sum.to_csv(out / "summary_long.tsv", sep="\t", index=False)

    main = all_sum[all_sum["method"].isin(MAIN_METHODS)].copy()
    main.to_csv(out / "summary.tsv", sep="\t", index=False)

    # mean ± std over seeds for dual and pearson
    focus = main[main["method"].isin(["dual_encoder_v2", "pearson_pred_gallery"])]
    agg = (
        focus.groupby(["cell_line", "method"])
        .agg(
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
            pct_top10_mean=("pct_top10", "mean"),
            pct_top10_std=("pct_top10", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    agg.to_csv(out / "dual_vs_pearson_seed_mean_std.tsv", sep="\t", index=False)

    # wide: dual mean median_rank by cell
    dual_wide = agg[agg["method"] == "dual_encoder_v2"].set_index("cell_line")
    pear_wide = agg[agg["method"] == "pearson_pred_gallery"].set_index("cell_line")
    report = pd.DataFrame(
        {
            "dual_median_rank_mean": dual_wide["median_rank_mean"],
            "dual_median_rank_std": dual_wide["median_rank_std"],
            "pearson_median_rank_mean": pear_wide["median_rank_mean"],
            "pearson_median_rank_std": pear_wide["median_rank_std"],
            "dual_top10_mean": dual_wide["pct_top10_mean"],
            "pearson_top10_mean": pear_wide["pct_top10_mean"],
            "n_seeds": dual_wide["n_seeds"],
        }
    )
    report.to_csv(out / "summary_seed_mean_std.tsv", sep="\t")

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(out / "dual_vs_pearson_stats_per_seed.tsv", sep="\t", index=False)
        # how many seeds have p<0.05
        sdf = pd.DataFrame(stats_rows)
        sig = (
            sdf.groupby("cell_line")
            .agg(
                n_seeds=("seed", "count"),
                n_sig_p05=("wilcoxon_pvalue", lambda s: int((s < 0.05).sum())),
                median_diff_mean=("median_diff_dual_minus_pearson", "mean"),
            )
            .reset_index()
        )
        sig.to_csv(out / "dual_vs_pearson_sig_summary.tsv", sep="\t", index=False)
        (out / "dual_vs_pearson_stats_per_seed.json").write_text(json.dumps(stats_rows, indent=2))

    print(report.to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
