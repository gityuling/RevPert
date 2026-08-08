#!/usr/bin/env python3
"""Aggregate fair_compare_* summaries + dual vs pearson stats across cell lines."""

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


def bootstrap_median_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI for median(a) - median(b); a,b are paired ranks (dual, pearson)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(float(np.median(a[idx]) - np.median(b[idx])))
    diffs = np.asarray(diffs)
    return {
        "median_rank_dual": float(np.median(a)),
        "median_rank_pearson": float(np.median(b)),
        "median_diff_dual_minus_pearson": float(np.median(a) - np.median(b)),
        "boot_mean_diff": float(diffs.mean()),
        "boot_ci95_low": float(np.quantile(diffs, 0.025)),
        "boot_ci95_high": float(np.quantile(diffs, 0.975)),
        "n_query": int(n),
        "n_boot": n_boot,
    }


def wilcoxon_dual_vs_pearson(dual: np.ndarray, pearson: np.ndarray) -> dict:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        # sign test fallback
        d = dual - pearson
        n_pos = int((d < 0).sum())  # dual better when rank lower
        n_neg = int((d > 0).sum())
        n = n_pos + n_neg
        # two-sided binomial approx
        from math import comb

        p = sum(comb(n, k) for k in range(0, min(n_pos, n_neg) + 1)) / (2 ** n) if n else 1.0
        return {"test": "sign_test_fallback", "n_dual_better": n_pos, "n_pearson_better": n_neg, "pvalue": float(p)}

    # lower rank is better → compare pearson - dual so positive means dual better
    stat, p = wilcoxon(pearson, dual, zero_method="wilcox", alternative="greater")
    return {
        "test": "wilcoxon_signed_rank",
        "alternative": "pearson_rank > dual_rank (dual better)",
        "statistic": float(stat),
        "pvalue": float(p),
        "n_dual_better": int((dual < pearson).sum()),
        "n_pearson_better": int((dual > pearson).sum()),
        "n_tie": int((dual == pearson).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list_cells())
    ap.add_argument("--out_dir", type=str, default=str(_ROOT / "reverse/results/fair_compare_all"))
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    stats_rows = []
    for cell in args.cells:
        paths = resolve_cell_paths(cell)
        fair = paths["fair_out"]
        sum_path = fair / "fair_compare_summary.tsv"
        ranks_path = fair / "all_method_ranks.tsv"
        if not sum_path.is_file():
            print(f"[skip] missing {sum_path}")
            continue
        sdf = pd.read_csv(sum_path, sep="\t")
        sdf["cell_line"] = paths["label"]
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
            boot = bootstrap_median_diff(a, b, n_boot=args.n_boot)
            wil = wilcoxon_dual_vs_pearson(a, b)
            stats_rows.append({"cell_line": paths["label"], **boot, **{f"wilcoxon_{k}": v for k, v in wil.items()}})

    if not rows:
        raise SystemExit("No fair_compare summaries found. Run compare_fair_baselines first.")

    all_sum = pd.concat(rows, ignore_index=True)
    all_sum.to_csv(out / "summary_long.tsv", sep="\t", index=False)

    # Wide median_rank / Top10 tables for main text
    for metric, fname in (("median_rank", "summary_median_rank.tsv"), ("pct_top10", "summary_top10.tsv")):
        if metric not in all_sum.columns:
            continue
        wide = all_sum.pivot_table(index="method", columns="cell_line", values=metric, aggfunc="first")
        wide.to_csv(out / fname, sep="\t")

    # Leaderboard-friendly: exclude oracle from "main"
    main_methods = [
        "dual_encoder_v2",
        "pearson_pred_gallery",
        "cmap_lite_pred_gallery",
        "gem_lite_topDEG_corr",
        "ridge_delta_to_P",
        "esm_mean_topDEG_to_KO",
    ]
    main = all_sum[all_sum["method"].isin(main_methods)].copy()
    main.to_csv(out / "summary.tsv", sep="\t", index=False)

    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        sdf.to_csv(out / "dual_vs_pearson_stats.tsv", sep="\t", index=False)
        (out / "dual_vs_pearson_stats.json").write_text(json.dumps(stats_rows, indent=2))

    print(main.pivot_table(index="method", columns="cell_line", values="median_rank").to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
