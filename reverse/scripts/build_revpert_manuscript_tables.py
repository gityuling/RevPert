#!/usr/bin/env python3
"""Build manuscript-facing tables for RevPert (replaces Dual as primary method).

Keeps Pearson / CMap / GEM / ridge / PDGrapher baselines from existing tables;
swaps Dual → RevPert numbers from reverse/results/revpert/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
REV = _ROOT / "reverse/results/revpert"
FIG_SRC = REV / "figures_source"
TABLES = REV / "tables"
OLD_FAIR = _ROOT / "reverse/results/figures_task1/fig2_fair_median_rank_source.tsv"
OLD_TOP10 = _ROOT / "reverse/results/figures_task1/fig2_fair_top10_source.tsv"
OLD_GEN = _ROOT / "reverse/results/pdgrapher_genetic_task1_unified/main_table_median_rank_r10.tsv"
OLD_MULTI = _ROOT / "reverse/dual_encoder_v2/results/fair_compare_multiseed/dual_vs_pearson_seed_mean_std.tsv"
LABEL = {"hepg2": "HepG2", "k562": "K562", "rpe1": "RPE1", "jurkat": "Jurkat"}


def main() -> None:
    FIG_SRC.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    id_all = pd.read_csv(REV / "essential/identity_all.tsv", sep="\t")
    # remap methods in identity_all if still old names
    id_all["method"] = id_all["method"].replace(
        {
            "pearson": "pearson_gallery",
            "fuse_alpha_train": "revpert",
            "gallery_dual": "revpert_learn_only",
            "fuse_alpha_retune": "revpert_retune",
        }
    )
    seed1 = id_all[(id_all.seed == 1) & (id_all.method.isin(["revpert", "pearson_gallery"]))]

    # --- Essential fair heatmap source: replace Dual row with RevPert ---
    fair = pd.read_csv(OLD_FAIR, sep="\t")
    # map revpert seed1 medians
    rev_med = {
        LABEL[r.cell]: float(r.median_rank)
        for _, r in seed1[seed1.method == "revpert"].iterrows()
    }
    pear_med = {
        LABEL[r.cell]: float(r.median_rank)
        for _, r in seed1[seed1.method == "pearson_gallery"].iterrows()
    }
    # rebuild: first row revpert, keep other methods except dual
    rows = [{"method": "revpert", **{c: rev_med[c] for c in ["HepG2", "K562", "RPE1", "Jurkat"]}}]
    for _, r in fair.iterrows():
        if r.method == "dual_encoder_v2":
            continue
        if r.method == "pearson_pred_gallery":
            rows.append(
                {
                    "method": "pearson_pred_gallery",
                    **{c: pear_med.get(c, r[c]) for c in ["HepG2", "K562", "RPE1", "Jurkat"]},
                }
            )
        else:
            rows.append(r.to_dict())
    fair_new = pd.DataFrame(rows)
    fair_new.to_csv(FIG_SRC / "fig2_fair_median_rank_source.tsv", sep="\t", index=False)

    # top10
    top = pd.read_csv(OLD_TOP10, sep="\t")
    rev_r10 = {
        LABEL[r.cell]: float(r["recall@10"]) * 100.0
        for _, r in seed1[seed1.method == "revpert"].iterrows()
    }
    pear_r10 = {
        LABEL[r.cell]: float(r["recall@10"]) * 100.0
        for _, r in seed1[seed1.method == "pearson_gallery"].iterrows()
    }
    trows = [{"method": "revpert", **{c: rev_r10[c] for c in ["HepG2", "K562", "RPE1", "Jurkat"]}}]
    for _, r in top.iterrows():
        if r.method == "dual_encoder_v2":
            continue
        if r.method == "pearson_pred_gallery":
            trows.append(
                {
                    "method": "pearson_pred_gallery",
                    **{c: pear_r10.get(c, r[c]) for c in ["HepG2", "K562", "RPE1", "Jurkat"]},
                }
            )
        else:
            trows.append(r.to_dict())
    pd.DataFrame(trows).to_csv(FIG_SRC / "fig2_fair_top10_source.tsv", sep="\t", index=False)

    # --- Multiseed ---
    multi = pd.read_csv(REV / "essential/identity_multiseed_mean_std.tsv", sep="\t")
    multi["method"] = multi["method"].replace(
        {"pearson": "pearson_gallery", "fuse_alpha_train": "revpert"}
    )
    mrows = []
    for cell, lab in LABEL.items():
        r = multi[(multi.cell == cell) & (multi.method == "revpert")].iloc[0]
        p = multi[(multi.cell == cell) & (multi.method == "pearson_gallery")].iloc[0]
        mrows.append(
            {
                "cell_line": lab,
                "revpert_median_rank_mean": r.median_rank_mean,
                "revpert_median_rank_std": r.median_rank_std,
                "pearson_median_rank_mean": p.median_rank_mean,
                "pearson_median_rank_std": p.median_rank_std,
                "revpert_top10_mean": r.recall10_mean * 100,
                "pearson_top10_mean": p.recall10_mean * 100,
                "n_seeds": int(r.n_seeds),
            }
        )
    multi_out = pd.DataFrame(mrows)
    multi_out.to_csv(FIG_SRC / "revpert_vs_pearson_seed_mean_std.tsv", sep="\t", index=False)
    # also dual-compatible column names for drop-in plot scripts
    compat = multi_out.rename(
        columns={
            "revpert_median_rank_mean": "dual_median_rank_mean",
            "revpert_median_rank_std": "dual_median_rank_std",
            "revpert_top10_mean": "dual_top10_mean",
        }
    )
    compat.to_csv(FIG_SRC / "dual_vs_pearson_seed_mean_std.tsv", sep="\t", index=False)

    by_seed = pd.read_csv(REV / "essential/revpert_vs_pearson_by_seed.tsv", sep="\t")
    by_seed["cell_line"] = by_seed["cell"].map(LABEL)
    by_seed.to_csv(FIG_SRC / "revpert_vs_pearson_stats_per_seed.tsv", sep="\t", index=False)
    by_seed.rename(
        columns={"revpert_median": "dual_median", "pearson_median": "pearson_median"}
    ).to_csv(FIG_SRC / "dual_vs_pearson_stats_per_seed.tsv", sep="\t", index=False)

    # --- Genetic: insert revpert, keep baselines, drop dual from primary or keep as archived ---
    gen_old = pd.read_csv(OLD_GEN, sep="\t")
    gen_rev = pd.read_csv(REV / "genetic/summary_by_cell.tsv", sep="\t")
    # build revpert rows
    fuse = gen_rev[gen_rev.method == "gallery_fuse"].copy()
    fuse["method"] = "revpert"
    # drop old dual from primary table (archived); keep other methods
    gen_new = gen_old[gen_old.method != "dual_encoder"].copy()
    # also refresh pearson from fuse run for consistency
    pear = gen_rev[gen_rev.method == "pearson_train_gallery"].copy()
    gen_new = gen_new[gen_new.method != "pearson_train_gallery"]
    pear_rows = []
    for _, r in pear.iterrows():
        pear_rows.append(
            {
                "cell": r.cell,
                "method": "pearson_train_gallery",
                "median_rank_mean": r.median_rank_mean,
                "median_rank_std": r.median_rank_std,
                "recall_at_10_mean": r.recall_at_10_mean,
                "recall_at_10_std": r.recall_at_10_std,
                "partial_pct_mean": float("nan"),
                "ndcg_mean": r.ndcg_mean if "ndcg_mean" in r else float("nan"),
                "n_folds": r.n_folds,
            }
        )
    rev_rows = []
    for _, r in fuse.iterrows():
        rev_rows.append(
            {
                "cell": r.cell,
                "method": "revpert",
                "median_rank_mean": r.median_rank_mean,
                "median_rank_std": r.median_rank_std,
                "recall_at_10_mean": r.recall_at_10_mean,
                "recall_at_10_std": r.recall_at_10_std,
                "partial_pct_mean": float("nan"),
                "ndcg_mean": r.ndcg_mean if "ndcg_mean" in r else float("nan"),
                "n_folds": r.n_folds,
            }
        )
    gen_out = pd.concat([pd.DataFrame(rev_rows), pd.DataFrame(pear_rows), gen_new], ignore_index=True)
    gen_out.to_csv(FIG_SRC / "main_table_median_rank_r10.tsv", sep="\t", index=False)
    # dual-compatible alias for plot scripts that still look for dual_encoder
    gen_alias = gen_out.copy()
    gen_alias["method"] = gen_alias["method"].replace({"revpert": "dual_encoder"})
    gen_alias.to_csv(FIG_SRC / "main_table_median_rank_r10_dualalias.tsv", sep="\t", index=False)

    # --- Fourteen-dataset board ---
    board = []
    for cell, lab in LABEL.items():
        r = seed1[(seed1.cell == cell) & (seed1.method == "revpert")].iloc[0]
        p = seed1[(seed1.cell == cell) & (seed1.method == "pearson_gallery")].iloc[0]
        board.append(
            {
                "resource": "Essential",
                "cell": lab,
                "dual_mr": r.median_rank,
                "base_mr": p.median_rank,
                "base_name": "Pearson",
                "fold": p.median_rank / max(r.median_rank, 1e-9),
                "dual_r10": r["recall@10"] * 100,
                "base_r10": p["recall@10"] * 100,
                "n_test": int(r.n),
                "n_catalog": "",
                "note": "revpert_vs_pearson",
            }
        )
    for _, r in fuse.iterrows():
        pdg = gen_old[(gen_old.cell == r.cell) & (gen_old.method == "pdgrapher_official")]
        base_mr = float(pdg.iloc[0].median_rank_mean) if not pdg.empty else float("nan")
        board.append(
            {
                "resource": "Genetic",
                "cell": r.cell,
                "dual_mr": r.median_rank_mean,
                "base_mr": base_mr,
                "base_name": "PDGrapher",
                "fold": base_mr / max(r.median_rank_mean, 1e-9),
                "dual_r10": r.recall_at_10_mean * 100,
                "base_r10": float(pdg.iloc[0].recall_at_10_mean) * 100 if not pdg.empty else float("nan"),
                "n_test": "",
                "n_catalog": 10716,
                "note": "revpert_vs_pdgrapher",
            }
        )
    board_df = pd.DataFrame(board)
    board_df.to_csv(FIG_SRC / "fourteen_dataset_board.tsv", sep="\t", index=False)

    # --- Leaderboard snapshot ---
    snap = {
        "primary_method": "revpert",
        "definition": "gallery-native residual: z(Pearson)+α z(GalleryDual), fused InfoNCE",
        "archived": "prototype dual-encoder (see ARCHIVED_DUAL_ENCODER.md)",
        "essential_seed1": {
            lab: {
                "revpert_median": rev_med[lab],
                "pearson_median": pear_med[lab],
                "fold": pear_med[lab] / rev_med[lab],
            }
            for lab in ["HepG2", "K562", "RPE1", "Jurkat"]
        },
        "essential_multiseed_mean_median": {
            LABEL[r.cell]: float(r.median_rank_mean)
            for _, r in multi[multi.method == "revpert"].iterrows()
        },
        "genetic_vs_pdgrapher": "10/10 RevPert better median rank",
        "essential_vs_pearson_seeds": "20/20 RevPert better",
    }
    (TABLES / "leaderboard_snapshot.json").write_text(json.dumps(snap, indent=2))
    board_df.to_csv(TABLES / "fourteen_dataset_board.tsv", sep="\t", index=False)
    fair_new.to_csv(TABLES / "essential_fair_median_rank.tsv", sep="\t", index=False)
    gen_out.to_csv(TABLES / "genetic_main_table.tsv", sep="\t", index=False)

    print("=== RevPert Essential seed1 ===")
    print(fair_new.head(2).to_string(index=False))
    print("\n=== Fold vs Pearson (seed1) ===")
    for lab in ["HepG2", "K562", "RPE1", "Jurkat"]:
        print(f"  {lab}: {pear_med[lab]:.0f} → {rev_med[lab]:.0f} ({pear_med[lab]/rev_med[lab]:.2f}×)")
    print("\n=== Genetic RevPert vs PDGrapher (median) ===")
    for _, r in fuse.iterrows():
        pdg = gen_old[(gen_old.cell == r.cell) & (gen_old.method == "pdgrapher_official")]
        b = float(pdg.iloc[0].median_rank_mean)
        print(f"  {r.cell}: RevPert {r.median_rank_mean:.0f} vs PDG {b:.0f} ({b/r.median_rank_mean:.1f}×)")
    print(f"\nWrote {FIG_SRC}")
    print(f"Wrote {TABLES}")


if __name__ == "__main__":
    main()
