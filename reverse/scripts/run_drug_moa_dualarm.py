#!/usr/bin/env python3
"""Literature-anchored drug MoA proving ground on HepG2 Essential gallery.

Primary HepG2-matched anchors (inhibitors → Arm A / KO phenocopy):
  - GSE158552 HepG2 JQ1 vs DMSO → BRD4
  - GSE158552 HepG2 OTX015 vs DMSO → BRD4 (same study replicate drug)
  - GSE145389 HepG2 palbociclib (PD-0332991) vs DMSO → CDK6
Cross-line note (kept, not a primary HepG2 match):
  - GSE159164 Huh7 shControl ± rapamycin → MTOR

DEG ranks are computed **within the HepG2 Essential catalog only** (same denominator as dual-arm).
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import resolve_cell_paths
from reverse.src.delta_y_star import align_delta_y_star, save_delta_y_star
from reverse.src.io_gallery import absolute_to_delta, load_ctrl_from_perturb_processed, load_prediction_dir
from reverse.src.score import score_gallery, top_k

SIG = _ROOT / "reverse/data/signatures"
OUT_ROOT = _ROOT / "reverse/results/drug_moa_hepg2_dualarm"
ENSEMBL_MAP = SIG / "raw_gse322742/ensembl_to_symbol.json"

# Pre-specified MoA anchors: expected arm is Arm A (KO phenocopy of inhibitor).
ANCHORS = [
    {
        "tag": "gse158552_hepg2_jq1",
        "drug": "JQ1",
        "line": "HepG2",
        "geo": "GSE158552",
        "target": "BRD4",
        "expected_arm": "A",
        "pathway_genes": ["BRD2", "BRD4", "BRD8"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158552",
        "primary": True,
    },
    {
        "tag": "gse158552_hepg2_otx015",
        "drug": "OTX015",
        "line": "HepG2",
        "geo": "GSE158552",
        "target": "BRD4",
        "expected_arm": "A",
        "pathway_genes": ["BRD2", "BRD4", "BRD8"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158552",
        "primary": True,
    },
    {
        "tag": "gse145389_hepg2_palbociclib",
        "drug": "palbociclib",
        "line": "HepG2",
        "geo": "GSE145389",
        "target": "CDK6",
        "expected_arm": "A",
        "pathway_genes": ["CDK6", "CDK1", "CDK2", "CDK7", "CDK9"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145389",
        "primary": True,
    },
    {
        "tag": "gse159164_huh7_rapamycin",
        "drug": "rapamycin",
        "line": "Huh7",
        "geo": "GSE159164",
        "target": "MTOR",
        "expected_arm": "A",
        "pathway_genes": ["MTOR"],
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159164",
        "primary": False,
    },
]


def _ensembl_map() -> dict[str, str]:
    raw = json.loads(ENSEMBL_MAP.read_text())
    # accept versioned and unversioned keys
    out = dict(raw)
    for k, v in list(raw.items()):
        out[k.split(".")[0]] = v
    return out


def _star_reads_per_gene(path: Path, ens_map: dict[str, str]) -> pd.Series:
    """STAR ReadsPerGene.out.tab → symbol-level unstranded counts."""
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("N_"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gid, cnt = parts[0], float(parts[1])
            base = gid.split(".")[0]
            sym = ens_map.get(gid) or ens_map.get(base)
            if not sym:
                continue
            rows.append((str(sym), cnt))
    if not rows:
        raise RuntimeError(f"No mappable genes in {path}")
    return pd.DataFrame(rows, columns=["gene", "count"]).groupby("gene")["count"].mean()


def build_gse158552_drug(drug: str, files: list[str], tag: str) -> Path:
    ens = _ensembl_map()
    ex = SIG / "raw_gse158552/extracted"
    ctrl = [
        _star_reads_per_gene(ex / "GSM4802741_HepG2_C1_24h.ReadsPerGene.out.tab.gz", ens),
        _star_reads_per_gene(ex / "GSM4802742_HepG2_C2_24h.ReadsPerGene.out.tab.gz", ens),
        _star_reads_per_gene(ex / "GSM4802743_HepG2_C3_24h.ReadsPerGene.out.tab.gz", ens),
    ]
    drug_s = [_star_reads_per_gene(ex / f, ens) for f in files]
    genes = sorted(set.intersection(*(set(s.index) for s in ctrl + drug_s)))
    c = np.mean([np.log2(s.reindex(genes).to_numpy(dtype=float) + 1.0) for s in ctrl], axis=0)
    d = np.mean([np.log2(s.reindex(genes).to_numpy(dtype=float) + 1.0) for s in drug_s], axis=0)
    delta = pd.Series(d - c, index=genes, dtype=float)
    path = SIG / f"{tag}_delta_y_star.tsv"
    pd.DataFrame({"gene": delta.index, "value": delta.values}).to_csv(path, sep="\t", index=False)
    (SIG / f"{tag}_delta_y_star.provenance.json").write_text(
        json.dumps(
            {
                "source": "GSE158552",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158552",
                "line": "HepG2",
                "drug": drug,
                "target": "BRD4",
                "expected_arm": "A",
                "definition": f"mean(log2(unstranded_count+1) {drug}) - mean(log2(unstranded_count+1) DMSO), Ensembl→symbol",
                "n_genes": int(len(delta)),
            },
            indent=2,
        )
    )
    return path


def build_gse158552_jq1() -> Path:
    return build_gse158552_drug(
        "JQ1",
        [
            "GSM4802744_HepG2_JQ1-1.ReadsPerGene.out.tab.gz",
            "GSM4802745_HepG2_JQ1-2.ReadsPerGene.out.tab.gz",
            "GSM4802746_HepG2_JQ1-3.ReadsPerGene.out.tab.gz",
        ],
        "gse158552_hepg2_jq1",
    )


def build_gse158552_otx015() -> Path:
    return build_gse158552_drug(
        "OTX015",
        [
            "GSM4802747_HepG2_OTX-1.ReadsPerGene.out.tab.gz",
            "GSM4802748_HepG2_OTX-2.ReadsPerGene.out.tab.gz",
            "GSM4802749_HepG2_OTX-3.ReadsPerGene.out.tab.gz",
        ],
        "gse158552_hepg2_otx015",
    )

def build_gse159164_rapa() -> Path:
    ex = SIG / "raw_gse159164/extracted"
    ctrl = pd.read_csv(ex / "GSM4820800_Huh7_shControl_RNA-seq.txt.gz", sep="\t").set_index("AccID")["FPKM"].astype(float)
    drug = pd.read_csv(ex / "GSM4820801_Huh7_shControl_Rapa_RNA-seq.txt.gz", sep="\t").set_index("AccID")["FPKM"].astype(float)
    ctrl = ctrl.groupby(level=0).mean()
    drug = drug.groupby(level=0).mean()
    genes = sorted(set(ctrl.index.astype(str)) & set(drug.index.astype(str)))
    # FPKM already continuous; use log2(FPKM+1) difference
    delta = np.log2(drug.reindex(genes).to_numpy(dtype=float) + 1.0) - np.log2(
        ctrl.reindex(genes).to_numpy(dtype=float) + 1.0
    )
    series = pd.Series(delta, index=genes, dtype=float)
    tag = "gse159164_huh7_rapamycin"
    path = SIG / f"{tag}_delta_y_star.tsv"
    pd.DataFrame({"gene": series.index, "value": series.values}).to_csv(path, sep="\t", index=False)
    (SIG / f"{tag}_delta_y_star.provenance.json").write_text(
        json.dumps(
            {
                "source": "GSE159164",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159164",
                "line": "Huh7",
                "drug": "rapamycin",
                "target": "MTOR",
                "expected_arm": "A",
                "note": "shControl ± rapamycin only (ARID1A arms unused)",
                "definition": "log2(FPKM+1)_Rapa - log2(FPKM+1)_vehicle on AccID symbols",
                "n_genes": int(len(series)),
            },
            indent=2,
        )
    )
    return path


def build_gse145389_palbo() -> Path:
    ex = SIG / "raw_gse145389/extracted"
    ctrl = pd.read_csv(ex / "GSM4317635_CTL.txt.gz", sep="\t").set_index("Gene symbol").iloc[:, 0].astype(float)
    drug = pd.read_csv(ex / "GSM4317636_PD.txt.gz", sep="\t").set_index("Gene symbol").iloc[:, 0].astype(float)
    ctrl = ctrl.groupby(level=0).mean()
    drug = drug.groupby(level=0).mean()
    genes = sorted(set(ctrl.index.astype(str)) & set(drug.index.astype(str)))
    delta = np.log2(drug.reindex(genes).to_numpy(dtype=float) + 1.0) - np.log2(
        ctrl.reindex(genes).to_numpy(dtype=float) + 1.0
    )
    series = pd.Series(delta, index=genes, dtype=float)
    tag = "gse145389_hepg2_palbociclib"
    path = SIG / f"{tag}_delta_y_star.tsv"
    pd.DataFrame({"gene": series.index, "value": series.values}).to_csv(path, sep="\t", index=False)
    (SIG / f"{tag}_delta_y_star.provenance.json").write_text(
        json.dumps(
            {
                "source": "GSE145389",
                "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145389",
                "line": "HepG2",
                "drug": "palbociclib (PD-0332991)",
                "target": "CDK6",
                "expected_arm": "A",
                "note": "n=1 per condition in deposited tables",
                "definition": "log2(count+1)_PD - log2(count+1)_CTL on gene symbols",
                "n_genes": int(len(series)),
            },
            indent=2,
        )
    )
    return path


def _deg_rank_catalog(star: pd.Series, gene: str, catalog: list[str]) -> tuple[float | None, int]:
    """Rank by |ΔY*| among catalog genes only (matched denominator to dual-arm)."""
    sub = star.reindex(catalog)
    mag = sub.abs()
    mag = mag[np.isfinite(mag)].sort_values(ascending=False)
    n = int(len(mag))
    if gene not in mag.index:
        return None, n
    return float(mag.index.get_loc(gene) + 1), n


def _deg_rank_genome(star: pd.Series, gene: str) -> tuple[float | None, int]:
    mag = star.abs()
    mag = mag[np.isfinite(mag)].sort_values(ascending=False)
    n = int(len(mag))
    if gene not in mag.index:
        return None, n
    return float(mag.index.get_loc(gene) + 1), n


def _gallery_hepg2():
    paths = resolve_cell_paths("hepg2")
    genes, pred_abs = load_prediction_dir(paths["pred_dir"])
    ctrl = load_ctrl_from_perturb_processed(paths["dataset_h5ad"], genes)
    gal = absolute_to_delta(pred_abs, ctrl)
    return genes, gal


def score_one(meta: dict, delta_path: Path, genes, gal) -> dict:
    out = OUT_ROOT / meta["tag"]
    out.mkdir(parents=True, exist_ok=True)
    star_series = pd.read_csv(delta_path, sep="\t").set_index("gene")["value"].astype(float)
    star_series = star_series.groupby(level=0).mean()
    star = align_delta_y_star(star_series, genes)
    save_delta_y_star(out / "delta_y_star.aligned.tsv", pd.Series(star, index=genes))
    scored = score_gallery(gal, star, genes, metric="pearson")
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=50, covered_only=False).to_csv(out / "top50.tsv", sep="\t", index=False)

    s = scored.set_index("ko")["score"].astype(float)
    catalog = s.index.astype(str).tolist()
    r_a = s.rank(ascending=False, method="average")  # Arm A: KO phenocopy
    r_b = (-s).rank(ascending=False, method="average")  # Arm B: activation
    dual = pd.DataFrame(
        {
            "ko": s.index,
            "score": s.values,
            "rank_armA_KO_phenocopy": r_a.values,
            "rank_armB_activation": r_b.values,
        }
    )
    dual.to_csv(out / "dualarm_scores.tsv", sep="\t", index=False)

    focus_genes = sorted(set(meta["pathway_genes"]) | {meta["target"]})
    rows = []
    for g in focus_genes:
        deg_cat, n_cat = _deg_rank_catalog(star_series, g, catalog)
        deg_gen, n_gen = _deg_rank_genome(star_series, g)
        row = {
            "gene": g,
            "in_gallery": g in s.index,
            "deg_rank": deg_cat,  # primary fair comparator
            "deg_rank_catalog": deg_cat,
            "n_deg_catalog": n_cat,
            "deg_rank_genome": deg_gen,
            "n_deg_genome": n_gen,
            "delta": float(star_series[g]) if g in star_series.index else np.nan,
        }
        if g in s.index:
            row.update(
                {
                    "rank_armA": float(r_a.loc[g]),
                    "rank_armB": float(r_b.loc[g]),
                    "score": float(s.loc[g]),
                    "preferred_rank": float(r_a.loc[g]),
                }
            )
        rows.append(row)
    focus = pd.DataFrame(rows)
    focus.to_csv(out / "focus_genes.tsv", sep="\t", index=False)

    tgt = meta["target"]
    tgt_row = focus.loc[focus.gene == tgt]
    summary = {
        **meta,
        "delta_path": str(delta_path),
        "n_gallery": int(len(s)),
        "n_signature_genes": int(star_series.shape[0]),
        "target_row": tgt_row.iloc[0].to_dict() if len(tgt_row) else None,
        "top10_armA": dual.nsmallest(10, "rank_armA_KO_phenocopy")["ko"].tolist(),
        "top10_armB": dual.nsmallest(10, "rank_armB_activation")["ko"].tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Building MoA ΔY* signatures...", flush=True)
    paths = {
        "gse158552_hepg2_jq1": build_gse158552_jq1(),
        "gse158552_hepg2_otx015": build_gse158552_otx015(),
        "gse159164_huh7_rapamycin": build_gse159164_rapa(),
        "gse145389_hepg2_palbociclib": build_gse145389_palbo(),
    }
    print("Loading HepG2 Essential gallery...", flush=True)
    genes, gal = _gallery_hepg2()
    summaries = []
    for meta in ANCHORS:
        print(f"Scoring {meta['tag']} ({meta['drug']} → {meta['target']})...", flush=True)
        summaries.append(score_one(meta, paths[meta["tag"]], genes, gal))

    rows = []
    for s in summaries:
        t = s.get("target_row") or {}
        rows.append(
            {
                "tag": s["tag"],
                "drug": s["drug"],
                "line": s["line"],
                "geo": s["geo"],
                "target": s["target"],
                "primary": s.get("primary", True),
                "expected_arm": s["expected_arm"],
                "rank_armA": t.get("rank_armA"),
                "rank_armB": t.get("rank_armB"),
                "deg_rank_catalog": t.get("deg_rank_catalog"),
                "n_deg_catalog": t.get("n_deg_catalog"),
                "deg_rank_genome": t.get("deg_rank_genome"),
                "n_deg_genome": t.get("n_deg_genome"),
                "in_gallery": t.get("in_gallery"),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_ROOT / "moa_anchor_summary.tsv", sep="\t", index=False)
    (OUT_ROOT / "summary.json").write_text(json.dumps({"anchors": summaries}, indent=2, default=str))
    (OUT_ROOT / "README.md").write_text(
        "# Drug MoA proving ground (HepG2 Essential)\n\n"
        "DEG ranks use the **catalog-matched** denominator (same genes as dual-arm).\n\n"
        "| Drug | GEO | Line | Target | Role |\n|------|-----|------|--------|------|\n"
        "| JQ1 | GSE158552 | HepG2 | BRD4 | primary |\n"
        "| OTX015 | GSE158552 | HepG2 | BRD4 | primary (same-study replicate) |\n"
        "| palbociclib | GSE145389 | HepG2 | CDK6 | primary |\n"
        "| rapamycin | GSE159164 | Huh7 | MTOR | cross-line note |\n"
    )
    print(tab.to_string(index=False))
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()