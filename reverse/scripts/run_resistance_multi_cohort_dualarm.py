#!/usr/bin/env python3
"""Build ΔY* for GSE121153 / GSE200098 / GSE186191 and dual-arm score on HepG2 gallery.

Acquisition scores: s = corr(KO, SR/LR − parental)
Arm+ post→pre (= activation heuristic for acquisition): rank by −s
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import resolve_cell_paths
from reverse.src.delta_y_star import align_delta_y_star, save_delta_y_star
from reverse.src.io_gallery import absolute_to_delta, load_ctrl_from_perturb_processed, load_prediction_dir
from reverse.src.score import score_gallery, top_k

SIG = _ROOT / "reverse/data/signatures"
OUT_ROOT = _ROOT / "reverse/results/resistance_multi_cohort_dualarm"

FOCUS = [
    "AURKA",
    "DUSP5",
    "LIPH",
    "MYOF",
    "RND3",
    "GRB10",
    "NCOA4",
    "FTH1",
    "TFRC",
    "PKM",
    "PTEN",
    "MCL1",
]


def _gallery_hepg2():
    paths = resolve_cell_paths("hepg2")
    genes, pred_abs = load_prediction_dir(paths["pred_dir"])
    ctrl = load_ctrl_from_perturb_processed(paths["dataset_h5ad"], genes)
    gal = absolute_to_delta(pred_abs, ctrl)
    return genes, gal


def _collapse_symbol_mean(df: pd.DataFrame) -> pd.Series:
    """df index=symbol, columns=samples → mean across samples then (optional later)."""
    return df.groupby(level=0).mean()


def build_gse200098() -> Path:
    raw = SIG / "raw_gse200098/GSE200098_normalised_gene_counts.txt.gz"
    df = pd.read_csv(raw, sep="\t")
    df = df.set_index("geneid")
    # normalised counts → log1p
    mat = np.log2(df.astype(float) + 1.0)
    parental = mat[["H1_Huh7", "H2_Huh7", "H3_Huh7"]].mean(axis=1)
    # combine SR1+SR2 as resistant pool (also write SR1/SR2 separately)
    sr1 = mat[["H4_SR1", "H5_SR1", "H6_SR1"]].mean(axis=1)
    sr2 = mat[["H7_SR2", "H8_SR2", "H9_SR2"]].mean(axis=1)
    sr = mat[[c for c in mat.columns if "SR" in c]].mean(axis=1)
    outs = {}
    for tag, delta in [
        ("gse200098_huh7_sr_pool", sr - parental),
        ("gse200098_huh7_sr1", sr1 - parental),
        ("gse200098_huh7_sr2", sr2 - parental),
    ]:
        s = delta.groupby(level=0).mean()  # unique symbols
        s.index = s.index.astype(str)
        path = SIG / f"{tag}_delta_y_star.tsv"
        pd.DataFrame({"gene": s.index, "value": s.values}).to_csv(path, sep="\t", index=False)
        (SIG / f"{tag}_delta_y_star.provenance.json").write_text(
            json.dumps(
                {
                    "source": "GSE200098",
                    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200098",
                    "definition": "mean(log2(norm+1) resistant) - mean(log2(norm+1) Huh7 parental)",
                    "tag": tag,
                },
                indent=2,
            )
        )
        outs[tag] = path
    return outs["gse200098_huh7_sr_pool"]


def build_gse186191() -> dict[str, Path]:
    outs = {}
    for line, fname, pcols, rcols in [
        (
            "huh7",
            "GSE186191_Huh7_P_vs_Huh7_LR.xlsx",
            ["Huh7_P1", "Huh7_P2", "Huh7_P3"],
            ["Huh7_LR1", "Huh7_LR2", "Huh7_LR3"],
        ),
        (
            "hep3b",
            "GSE186191_Hep3B_P_vs_Hep3B_LR.xlsx",
            ["Hep3B_P1", "Hep3B_P2", "Hep3B_P3"],
            ["Hep3B_LR1", "Hep3B_LR2", "Hep3B_LR3"],
        ),
    ]:
        x = pd.read_excel(SIG / "raw_gse186191" / fname)
        x.columns = [c.strip() for c in x.columns]
        # expression columns look like counts/TPM-ish; use log2(x+1) then LR-P
        for c in pcols + rcols:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        mat = x.set_index("Name")[pcols + rcols].astype(float)
        # duplicate symbols → mean
        mat = mat.groupby(level=0).mean()
        logm = np.log2(mat.clip(lower=0) + 1.0)
        delta = logm[rcols].mean(axis=1) - logm[pcols].mean(axis=1)
        tag = f"gse186191_{line}_lenvatinib"
        path = SIG / f"{tag}_delta_y_star.tsv"
        pd.DataFrame({"gene": delta.index.astype(str), "value": delta.values}).to_csv(path, sep="\t", index=False)
        (SIG / f"{tag}_delta_y_star.provenance.json").write_text(
            json.dumps(
                {
                    "source": "GSE186191",
                    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186191",
                    "line": line,
                    "drug": "lenvatinib",
                    "definition": "mean(log2(expr+1) LR) - mean(log2(expr+1) parental)",
                },
                indent=2,
            )
        )
        outs[tag] = path
    return outs


def _affy_to_symbol(probe_ids: list[str], cache: Path) -> dict[str, str]:
    if cache.exists():
        return json.loads(cache.read_text())
    mapping: dict[str, str] = {}
    url = "https://mygene.info/v3/query"
    chunk = 1000
    for i in range(0, len(probe_ids), chunk):
        batch = probe_ids[i : i + chunk]
        r = requests.post(
            url,
            json={
                "q": batch,
                "scopes": "reporter",
                "fields": "symbol",
                "species": "human",
            },
            timeout=180,
        )
        r.raise_for_status()
        for hit in r.json():
            q = hit.get("query")
            sym = hit.get("symbol")
            if q and sym and "notfound" not in hit:
                mapping[str(q)] = str(sym)
        time.sleep(0.15)
        print(f"  mapped {min(i+chunk, len(probe_ids))}/{len(probe_ids)}", flush=True)
    cache.write_text(json.dumps(mapping))
    return mapping


def build_gse121153() -> dict[str, Path]:
    path = SIG / "raw_gse121153/GSE121153_series_matrix.txt.gz"
    # parse matrix
    rows = []
    header = None
    with gzip.open(path, "rt") as fh:
        in_table = False
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue
            parts = line.rstrip("\n").split("\t")
            parts = [p.strip('"') for p in parts]
            if header is None:
                header = parts
                continue
            rows.append(parts)
    mat = pd.DataFrame(rows, columns=header).set_index("ID_REF")
    mat = mat.apply(pd.to_numeric, errors="coerce")

    # sample groups (paper Fig: resistant n=5 vs sensitive n=3 tissues/xenografts)
    parental_xeno = ["GSM3427012", "GSM3427013", "GSM3427014"]
    resistant_xeno = ["GSM3427017", "GSM3427018", "GSM3427019", "GSM3427020", "GSM3427021"]
    huh7_p = ["GSM3427015"]
    huh7_sr = ["GSM3427016"]

    probes = mat.index.astype(str).tolist()
    cache = SIG / "raw_gse121153/affy_probe_to_symbol.json"
    print("Mapping Affy probes → symbols (mygene)...", flush=True)
    p2s = _affy_to_symbol(probes, cache)

    def delta_for(res_cols, par_cols, tag: str) -> Path:
        sub = mat[res_cols + par_cols].copy()
        sym = pd.Series({p: p2s.get(p) for p in sub.index})
        sub = sub.assign(_sym=sym.values)
        sub = sub.dropna(subset=["_sym"])
        # already log-scale typical for series matrix; use as-is
        g = sub.groupby("_sym")
        res_m = g[res_cols].mean().mean(axis=1)
        par_m = g[par_cols].mean().mean(axis=1)
        delta = res_m - par_m
        outp = SIG / f"{tag}_delta_y_star.tsv"
        pd.DataFrame({"gene": delta.index.astype(str), "value": delta.values}).to_csv(outp, sep="\t", index=False)
        (SIG / f"{tag}_delta_y_star.provenance.json").write_text(
            json.dumps(
                {
                    "source": "GSE121153",
                    "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121153",
                    "platform": "GPL570",
                    "tag": tag,
                    "resistant": res_cols,
                    "parental": par_cols,
                    "n_probes_mapped": int(sym.notna().sum()),
                    "definition": "mean(resistant) - mean(parental) on series-matrix intensities, probes collapsed by symbol mean",
                },
                indent=2,
            )
        )
        return outp

    outs = {
        "gse121153_xenograft": delta_for(resistant_xeno, parental_xeno, "gse121153_xenograft_sorafenib"),
        "gse121153_huh7_cells": delta_for(huh7_sr, huh7_p, "gse121153_huh7_cell_sorafenib"),
    }
    return outs


def score_and_dualarm(tag: str, delta_path: Path, genes, gal) -> dict:
    out = OUT_ROOT / tag
    out.mkdir(parents=True, exist_ok=True)
    star_series = pd.read_csv(delta_path, sep="\t").set_index("gene")["value"]
    star = align_delta_y_star(star_series, genes)
    save_delta_y_star(out / "delta_y_star.aligned.tsv", pd.Series(star, index=genes))
    scored = score_gallery(gal, star, genes, metric="pearson")
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=50, covered_only=False).to_csv(out / "top50.tsv", sep="\t", index=False)

    s = scored.set_index("ko")["score"].astype(float)
    # acquisition: s = corr(KO, res-par)
    r_ko_acq = s.rank(ascending=False, method="average")
    r_act_acq = (-s).rank(ascending=False, method="average")  # activation heuristic / Arm+ post→pre
    dual = pd.DataFrame(
        {
            "ko": s.index,
            "score_res_minus_par": s.values,
            "rank_arm_plus_KO_phenocopy_acquisition": r_ko_acq.values,
            "rank_arm_minus_activation_heuristic_acquisition": r_act_acq.values,
            "rank_arm_plus_KO_phenocopy_post_to_pre": r_act_acq.values,
            "rank_arm_minus_activation_heuristic_post_to_pre": r_ko_acq.values,
        }
    )
    dual.to_csv(out / "dualarm_scores.tsv", sep="\t", index=False)
    dual.nsmallest(50, "rank_arm_minus_activation_heuristic_acquisition").to_csv(
        out / "top50_activation_arm_acquisition.tsv", sep="\t", index=False
    )
    dual.nsmallest(50, "rank_arm_plus_KO_phenocopy_post_to_pre").to_csv(
        out / "top50_arm_plus_post_to_pre.tsv", sep="\t", index=False
    )

    # signature values for focus genes
    star_map = star_series.astype(float).to_dict()
    rows = []
    for g in FOCUS:
        row = {"gene": g, "in_gallery": g in s.index, "delta_res_minus_par": star_map.get(g, np.nan)}
        if g in s.index:
            row.update(
                {
                    "rank_KO_acq": float(r_ko_acq.loc[g]),
                    "rank_activation_acq_or_ArmPlus_post_to_pre": float(r_act_acq.loc[g]),
                    "score": float(s.loc[g]),
                }
            )
        rows.append(row)
    focus = pd.DataFrame(rows)
    focus.to_csv(out / "focus_genes.tsv", sep="\t", index=False)

    aurka = focus.loc[focus.gene == "AURKA"].iloc[0].to_dict()
    return {
        "tag": tag,
        "delta_path": str(delta_path),
        "n_gallery": int(len(s)),
        "AURKA": aurka,
        "top10_activation_arm": dual.nsmallest(10, "rank_arm_minus_activation_heuristic_acquisition")["ko"].tolist(),
        "top10_KO_acq": dual.nsmallest(10, "rank_arm_plus_KO_phenocopy_acquisition")["ko"].tolist(),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Building signatures...", flush=True)
    paths = {}
    paths["gse200098_huh7_sr_pool"] = build_gse200098()
    for tag in ("gse200098_huh7_sr1", "gse200098_huh7_sr2"):
        paths[tag] = SIG / f"{tag}_delta_y_star.tsv"
    paths.update(build_gse186191())
    paths.update(build_gse121153())
    # also include existing GSE322742 signatures
    paths["gse322742_hepg2_sorafenib"] = SIG / "hepg2_sorafenib_delta_y_star.tsv"
    paths["gse322742_huh7_sorafenib"] = SIG / "huh7_sorafenib_delta_y_star.tsv"
    # dedupe gene labels in any pre-existing TSVs that may have dups
    for tag, p in list(paths.items()):
        df = pd.read_csv(p, sep="\t")
        if df["gene"].duplicated().any():
            df = df.groupby("gene", as_index=False)["value"].mean()
            df.to_csv(p, sep="\t", index=False)

    print("Loading HepG2 gallery...", flush=True)
    genes, gal = _gallery_hepg2()

    summaries = []
    for tag, p in paths.items():
        print(f"Scoring {tag} ...", flush=True)
        summaries.append(score_and_dualarm(tag, Path(p), genes, gal))

    # coverage table for paper's 6 genes
    six = ["AURKA", "DUSP5", "LIPH", "MYOF", "RND3", "GRB10"]
    gal_set = set(genes)
    # genes list may be feature genes not KO list — use scored kos from one run
    kos = set(pd.read_csv(OUT_ROOT / "gse322742_hepg2_sorafenib" / "reverse_scores.tsv", sep="\t")["ko"])
    cov = [{"gene": g, "in_hepg2_essential_gallery": g in kos} for g in six]
    # axis genes
    for g in ["NCOA4", "FTH1", "TFRC"]:
        cov.append({"gene": g, "in_hepg2_essential_gallery": g in kos, "role": "mechanism_axis"})
    pd.DataFrame(cov).to_csv(OUT_ROOT / "paper_gene_gallery_coverage.tsv", sep="\t", index=False)

    # comparison table
    comp_rows = []
    for s in summaries:
        a = s["AURKA"]
        comp_rows.append(
            {
                "cohort": s["tag"],
                "AURKA_in_gallery": a.get("in_gallery"),
                "AURKA_delta_res_minus_par": a.get("delta_res_minus_par"),
                "AURKA_rank_KO_acquisition": a.get("rank_KO_acq"),
                "AURKA_rank_activation_arm_or_post_to_pre_ArmPlus": a.get(
                    "rank_activation_acq_or_ArmPlus_post_to_pre"
                ),
                "top10_activation_arm": ",".join(s["top10_activation_arm"]),
            }
        )
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT_ROOT / "AURKA_cross_cohort_dualarm.tsv", sep="\t", index=False)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(comp.to_string(index=False))
    print(f"\nWrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
