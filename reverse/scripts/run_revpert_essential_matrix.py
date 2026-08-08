#!/usr/bin/env python3
"""End-to-end RevPert (gallery-native residual) on Essential 4 lines × seeds 1–5.

RevPert := fuse_alpha_train from run_gallery_dual_g2
  s = z(Pearson) + α z(GalleryDual)

Writes:
  reverse/results/revpert/essential/{cell}/seed{N}/
  reverse/results/revpert/essential/summary_*.tsv
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
PY = Path(sys.executable)
SCRIPT = _ROOT / "reverse/scripts/run_gallery_dual_g2.py"
OUT_ROOT = _ROOT / "reverse/results/revpert"
CELLS = ["hepg2", "k562", "rpe1", "jurkat"]
SEEDS = [1, 2, 3, 4, 5]


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_one(cell: str, seed: int, epochs: int, preserve_w: float, device: str) -> Path:
    out = OUT_ROOT / "essential" / cell / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "identity_test_summary.tsv"
    if marker.exists() and (out / "gallery_dual_best.pt").exists():
        _log(f"[skip] {cell} seed{seed} already done")
        return out
    cmd = [
        str(PY),
        str(SCRIPT),
        "--cell_line",
        cell,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--preserve_w",
        str(preserve_w),
        "--skip_gwps",
        "1",
        "--device",
        device,
        "--out_dir",
        str(out),
    ]
    _log(f"[run] {' '.join(cmd)}")
    log = out / "run.log"
    with log.open("w") as f:
        r = subprocess.run(cmd, cwd=str(_ROOT), env={**dict(**{k: __import__('os').environ.get(k, '') for k in []}), "PYTHONPATH": str(_ROOT), "PATH": __import__("os").environ.get("PATH", "")}, stdout=f, stderr=subprocess.STDOUT)
    # simpler env
    return out if r.returncode == 0 else out


def run_one_fixed(cell: str, seed: int, epochs: int, preserve_w: float, device: str) -> Path:
    import os

    out = OUT_ROOT / "essential" / cell / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "identity_test_summary.tsv").exists() and (out / "gallery_dual_best.pt").exists():
        _log(f"[skip] {cell} seed{seed}")
        return out
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    cmd = [
        str(PY),
        str(SCRIPT),
        "--cell_line",
        cell,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--preserve_w",
        str(preserve_w),
        "--skip_gwps",
        "1",
        "--device",
        device,
        "--out_dir",
        str(out),
    ]
    _log(f"[run] {cell} seed{seed}")
    log = out / "run.log"
    with log.open("w") as f:
        r = subprocess.run(cmd, cwd=str(_ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        _log(f"[FAIL] {cell} seed{seed} rc={r.returncode}; see {log}")
        raise SystemExit(r.returncode)
    return out


METHOD_MAP = {
    "pearson": "pearson_gallery",
    "gallery_dual": "revpert_learn_only",
    "fuse_alpha_train": "revpert",
    "fuse_alpha_retune": "revpert_retune",
}


def rename_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].map(lambda m: METHOD_MAP.get(m, m))
    return out


def aggregate(cells, seeds) -> None:
    id_rows = []
    res_rows = []
    for cell in cells:
        for seed in seeds:
            d = OUT_ROOT / "essential" / cell / f"seed{seed}"
            idf = d / "identity_test_summary.tsv"
            if not idf.exists():
                continue
            df = rename_identity(pd.read_csv(idf, sep="\t"))
            df["cell"] = cell
            df["seed"] = seed
            id_rows.append(df)
            rf = d / "resistance_focus_compare.tsv"
            if rf.exists():
                r = pd.read_csv(rf, sep="\t")
                r["method"] = r["method"].map(
                    {
                        "pearson": "pearson_gallery",
                        "gallery_dual": "revpert_learn_only",
                        "fuse_alpha_train": "revpert",
                        "fuse_alpha_retune": "revpert_retune",
                    }
                ).fillna(r["method"])
                r["cell_model"] = cell
                r["seed"] = seed
                res_rows.append(r)

    if not id_rows:
        _log("No identity results to aggregate")
        return
    id_all = pd.concat(id_rows, ignore_index=True)
    ess = OUT_ROOT / "essential"
    ess.mkdir(parents=True, exist_ok=True)
    id_all.to_csv(ess / "identity_all.tsv", sep="\t", index=False)

    # primary methods for leaderboard
    primary = id_all[id_all.method.isin(["pearson_gallery", "revpert"])]
    seed1 = primary[primary.seed == 1].pivot_table(
        index="cell", columns="method", values=["median_rank", "mrr", "recall@10"], aggfunc="first"
    )
    seed1.to_csv(ess / "identity_seed1_wide.tsv", sep="\t")

    multi = (
        primary.groupby(["cell", "method"])
        .agg(
            median_rank_mean=("median_rank", "mean"),
            median_rank_std=("median_rank", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
            recall10_mean=("recall@10", "mean"),
            recall10_std=("recall@10", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    multi.to_csv(ess / "identity_multiseed_mean_std.tsv", sep="\t", index=False)

    # fair-style: fold vs pearson
    wins = []
    for cell in cells:
        sub = primary[primary.cell == cell]
        for seed in seeds:
            s = sub[sub.seed == seed]
            if s.empty:
                continue
            p = float(s[s.method == "pearson_gallery"]["median_rank"].iloc[0])
            r = float(s[s.method == "revpert"]["median_rank"].iloc[0])
            wins.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "pearson_median": p,
                    "revpert_median": r,
                    "fold_improvement": p / max(r, 1e-9),
                    "revpert_better": r < p,
                }
            )
    wdf = pd.DataFrame(wins)
    wdf.to_csv(ess / "revpert_vs_pearson_by_seed.tsv", sep="\t", index=False)
    _log("\n=== Essential seed1 (RevPert vs Pearson) ===")
    _log(wdf[wdf.seed == 1].to_string(index=False))
    _log("\n=== Multiseed mean median rank ===")
    piv = multi.pivot(index="cell", columns="method", values="median_rank_mean")
    _log(piv.to_string())
    _log(f"RevPert better than Pearson: {int(wdf.revpert_better.sum())}/{len(wdf)}")

    if res_rows:
        rr = pd.concat(res_rows, ignore_index=True)
        rr.to_csv(ess / "resistance_all.tsv", sep="\t", index=False)
        # HepG2 seed1 headline
        h = rr[(rr.cell_model == "hepg2") & (rr.seed == 1) & (rr.method.isin(["pearson_gallery", "revpert"]))]
        if not h.empty:
            wide = h.pivot_table(index=["tag", "gene"], columns="method", values="rank_pref", aggfunc="first")
            wide.to_csv(ess / "resistance_hepg2_seed1_wide.tsv", sep="\t")
            _log("\n=== Resistance HepG2 seed1 ===")
            _log(wide.to_string())

    (ess / "README.md").write_text(
        "# RevPert Essential identity recovery\n\n"
        "Primary method: **revpert** = Pearson residual + gallery dual (fused InfoNCE).\n"
        "Ablations: `pearson_gallery`, `revpert_learn_only`, `revpert_retune`.\n"
        "Legacy prototype dual-encoder: see `../ARCHIVED_DUAL_ENCODER.md`.\n",
        encoding="utf-8",
    )


def link_genetic() -> None:
    src = _ROOT / "reverse/results/pdgrapher_gallery_fuse"
    dst = OUT_ROOT / "genetic"
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "all_fold_metrics.tsv",
        "summary_by_cell.tsv",
        "vs_pdgrapher_dual.tsv",
        "pdgrapher_metrics_compare.tsv",
        "pdgrapher_metrics_macro.tsv",
    ]:
        s = src / name
        if s.exists():
            t = dst / name
            if t.exists() or t.is_symlink():
                t.unlink()
            shutil.copy2(s, t)
    # rewrite method names in a leaderboard view
    summ = dst / "summary_by_cell.tsv"
    if summ.exists():
        df = pd.read_csv(summ, sep="\t")
        df["method"] = df["method"].replace(
            {
                "gallery_fuse": "revpert",
                "gallery_dual": "revpert_learn_only",
                "pearson_train_gallery": "pearson_gallery",
            }
        )
        df.to_csv(dst / "summary_by_cell_revpert.tsv", sep="\t", index=False)
    (dst / "README.md").write_text(
        "# RevPert on PDGrapher genetic protocol\n\n"
        "Copied from `pdgrapher_gallery_fuse/`. Primary method renamed **revpert**.\n",
        encoding="utf-8",
    )
    _log(f"Linked genetic results → {dst}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--preserve_w", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "README.md").write_text(
        "# RevPert — gallery-native reverse perturbation\n\n"
        "**RevPert** learns residual similarity on an expression KO gallery while "
        "preserving signed Pearson connectivity.\n\n"
        "- `essential/` — Replogle Essential 4 lines × seeds\n"
        "- `genetic/` — PDGrapher genetic 10 lines (official folds)\n"
        "- Legacy dual-encoder archived: `../ARCHIVED_DUAL_ENCODER.md`\n",
        encoding="utf-8",
    )

    if not args.aggregate_only:
        for cell in args.cells:
            for seed in args.seeds:
                run_one_fixed(cell, seed, args.epochs, args.preserve_w, args.device)

    aggregate(args.cells, args.seeds)
    link_genetic()
    _log(f"\nDone. Root: {OUT_ROOT}")


if __name__ == "__main__":
    main()
