#!/usr/bin/env python3
"""Aggregate gallery_compare_* summaries (TxPert / GEARS / linear / scGPT / dual)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import list_cells, resolve_cell_paths  # noqa: E402

PRIMARY = {
    "dual_encoder_v2",
    "linear_L3_gallery_pearson",
    "txpert_gat_gallery_pearson",
    "txpert_xcell_gallery_pearson",
    "gears_gallery_pearson",
    "scgpt_gallery_pearson",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list_cells())
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(_ROOT / "reverse/results/gallery_compare_all"),
    )
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in args.cells:
        paths = resolve_cell_paths(cell)
        p = paths["gallery_compare_out"] / "summary.tsv"
        if not p.is_file():
            print(f"[skip] {p}")
            continue
        df = pd.read_csv(p, sep="\t")
        if "cell_line" not in df.columns:
            df["cell_line"] = paths["label"]
        rows.append(df)
    if not rows:
        raise SystemExit("No gallery_compare summaries found.")
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out / "summary.tsv", sep="\t", index=False)

    primary = all_df[all_df["method"].isin(PRIMARY)].copy()
    primary.to_csv(out / "summary_primary.tsv", sep="\t", index=False)
    # Drop legacy cover-only rows if any remain in old runs
    primary = primary[~primary["method"].str.contains("on_scgpt_cover", na=False)]

    wide = primary.pivot_table(index="method", columns="cell_line", values="median_rank", aggfunc="first")
    wide.to_csv(out / "summary_median_rank.tsv", sep="\t")
    wide.to_csv(out / "summary_one_table_median_rank.tsv", sep="\t")
    if "pct_top10" in primary.columns:
        primary.pivot_table(index="method", columns="cell_line", values="pct_top10", aggfunc="first").to_csv(
            out / "summary_top10.tsv", sep="\t"
        )
    (out / "summary.json").write_text(json.dumps(all_df.to_dict(orient="records"), indent=2))
    print("=== gallery (same catalog / n_query; scGPT missing → worst rank) ===")
    print(wide.to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
