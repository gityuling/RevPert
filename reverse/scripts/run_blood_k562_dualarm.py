#!/usr/bin/env python3
"""Blood-lineage dual-arm pilot: K562 imatinib resistance (GSE120932) × K562 GWPS / Essential.

Gallery:
  - K562 GWPS observed bulk KO gallery (~9.9k) — primary (full knockout catalog)
  - K562 Essential L3 predicted gallery — secondary (Essential-matched Task-2 form)

Signatures (acquisition = resistant − parental):
  - ir_no_drug: K562-IR cultured without imatinib − parental K562
  - ir_with_drug: K562-IR maintained in imatinib − parental K562
  - spindle_ir: spindle-shaped K562-IR − parental K562

Claim boundary: lineage-matched usage extrapolation / signed calibration pilot,
not CML target discovery.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.cell_lines import resolve_cell_paths  # noqa: E402
from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv  # noqa: E402
from reverse.src.gwps_reverse import load_gwps_deltas  # noqa: E402
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_prediction_dir,
)
from reverse.src.score import score_gallery, top_k  # noqa: E402

RAW = _ROOT / "reverse/data/signatures/raw_gse120932"
SIG = _ROOT / "reverse/data/signatures"
OUT_ROOT = _ROOT / "reverse/results/blood_k562_imatinib_dualarm"
MATRIX = RAW / "GSE120932_series_matrix.txt.gz"
ANNOT_CACHE = RAW / "gpl10558_probe_to_symbol.json"

# Paper-associated / CML-relevant anchors to report if in gallery
FOCUS = [
    "IFITM3",
    "CD33",
    "CD36",
    "ABL1",
    "BCR",
    "MYC",
    "TP53",
    "NPM1",
    "DNMT3A",
    "RUNX1",
    "KIT",
    "KRAS",
    "NRAS",
    "WT1",
    "TET2",
    "ASXL1",
    "IDH1",
    "IDH2",
    "CEBPA",
    "STAT5A",
    "STAT5B",
    "LYN",
    "MTOR",
    "MCL1",
    "ITGB1",
    "BAX",
]


def _probe_to_symbol() -> dict[str, str]:
    if ANNOT_CACHE.exists():
        return json.loads(ANNOT_CACHE.read_text())
    ftp = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz"
    r = requests.get(ftp, timeout=180)
    r.raise_for_status()
    mapping: dict[str, str] = {}
    header: list[str] | None = None
    sym_idx: int | None = None
    for line in gzip.decompress(r.content).decode("utf-8", errors="replace").splitlines():
        if not line or line.startswith("#") or line.startswith("^") or line.startswith("!"):
            continue
        parts = line.split("\t")
        if header is None:
            if parts[0] != "ID":
                continue
            header = parts
            for i, h in enumerate(header):
                if h.strip().lower() == "gene symbol":
                    sym_idx = i
                    break
            if sym_idx is None:
                raise RuntimeError(f"No Gene symbol column in GPL10558 annot; cols={header[:10]}")
            continue
        if len(parts) <= sym_idx:
            continue
        pid, sym = parts[0], parts[sym_idx]
        if not sym or sym in {"", "---"}:
            continue
        sym0 = sym.split("///")[0].split("//")[0].strip()
        if sym0 and sym0 != "---":
            mapping[pid] = sym0
    if len(mapping) < 1000:
        raise RuntimeError(f"GPL10558 annot mapping too small ({len(mapping)}); check FTP")
    ANNOT_CACHE.write_text(json.dumps(mapping))
    print(f"Cached {len(mapping)} probe→symbol mappings (Gene symbol col)")
    return mapping
def _load_matrix() -> tuple[list[str], pd.DataFrame]:
    """Return sample titles and probe × sample expression (log2-ish as deposited)."""
    titles: list[str] = []
    gsms: list[str] = []
    table_lines: list[str] = []
    in_table = False
    with gzip.open(MATRIX, "rt") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                gsms = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                table_lines.append(line.rstrip("\n"))
    from io import StringIO

    df = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t")
    # first col = ID_REF
    idcol = df.columns[0]
    df = df.set_index(idcol)
    df.columns = gsms
    df = df.apply(pd.to_numeric, errors="coerce")
    return titles, df


def build_deltas() -> dict[str, Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    titles, mat = _load_matrix()
    # groups by title prefix
    groups = {
        "parental": [g for g, t in zip(mat.columns, titles) if t.startswith("K562,")],
        "spindle_ir": [g for g, t in zip(mat.columns, titles) if t.startswith("Spindle-shaped")],
        "ir_no_drug": [g for g, t in zip(mat.columns, titles) if t.startswith("K562-IR w/o")],
        "ir_with_drug": [
            g for g, t in zip(mat.columns, titles) if t.startswith("K562-IR w/") and "w/o" not in t
        ],
    }
    print({k: v for k, v in groups.items()})
    mapping = _probe_to_symbol()
    # probe → symbol mean
    sym = mat.copy()
    sym.index = [mapping.get(str(i), "") for i in sym.index]
    sym = sym[sym.index != ""]
    sym = sym.groupby(level=0).mean()

    out: dict[str, Path] = {}
    parental = sym[groups["parental"]].mean(axis=1)
    contrasts = {
        "ir_no_drug": groups["ir_no_drug"],
        "ir_with_drug": groups["ir_with_drug"],
        "spindle_ir": groups["spindle_ir"],
    }
    for tag, cols in contrasts.items():
        delta = sym[cols].mean(axis=1) - parental
        path = SIG / f"gse120932_k562_{tag}_delta_y_star.tsv"
        pd.DataFrame({"gene": delta.index.astype(str), "value": delta.values.astype(float)}).to_csv(
            path, sep="\t", index=False
        )
        prov = {
            "source": "GEO GSE120932",
            "platform": "GPL10558 Illumina HumanHT-12 V4.0",
            "pubmed": 32106243,
            "contrast": f"mean({tag}) - mean(parental K562)",
            "n_parental": len(groups["parental"]),
            "n_resistant": len(cols),
            "n_genes": int(delta.shape[0]),
            "claim_boundary": "K562 lineage-matched imatinib-resistance usage; not CML target discovery",
            "paper_markers_noted": ["IFITM3", "CD36", "CD33"],
        }
        path.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=2))
        out[tag] = path
        print(f"Wrote {path} n={len(delta)}")
    return out


def _gallery_essential():
    paths = resolve_cell_paths("k562")
    genes, pred = load_prediction_dir(paths["pred_dir"])
    ctrl = load_ctrl_from_perturb_processed(paths["dataset_h5ad"], genes)
    return genes, absolute_to_delta(pred, ctrl)


def _dualarm(scored: pd.DataFrame) -> pd.DataFrame:
    s = scored.set_index("ko")["score"].astype(float)
    # Arm A: KO phenocopy of acquisition (rank by s)
    # Arm B: activation / post→pre heuristic (rank by -s)
    dual = pd.DataFrame(
        {
            "ko": s.index,
            "score_res_minus_par": s.values,
            "rank_KO_phenocopy_acquisition": s.rank(ascending=False, method="average").values,
            "rank_activation_arm_or_post_to_pre": (-s).rank(ascending=False, method="average").values,
        }
    )
    return dual.sort_values("rank_KO_phenocopy_acquisition")


def score_one(tag: str, delta_path: Path, gallery_name: str, genes, gal) -> dict:
    out = OUT_ROOT / f"{gallery_name}__{tag}"
    out.mkdir(parents=True, exist_ok=True)
    star_raw = load_vector_tsv(delta_path)
    star = align_delta_y_star(star_raw, genes)
    cov = float(np.isfinite(star).mean())
    scored = score_gallery(gal, star, genes, metric="pearson")
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=50, covered_only=False).to_csv(out / "top50.tsv", sep="\t", index=False)
    dual = _dualarm(scored)
    dual.to_csv(out / "dualarm_scores.tsv", sep="\t", index=False)
    dual.nsmallest(50, "rank_KO_phenocopy_acquisition").to_csv(out / "top50_KO_acquisition.tsv", sep="\t", index=False)
    dual.nsmallest(50, "rank_activation_arm_or_post_to_pre").to_csv(
        out / "top50_activation_arm.tsv", sep="\t", index=False
    )

    # vs DEG
    abs_star = np.abs(star)
    deg_idx = np.argsort(-abs_star)[:50]
    deg_set = {genes[i] for i in deg_idx}
    top_a = set(dual.nsmallest(50, "rank_KO_phenocopy_acquisition")["ko"])
    top_b = set(dual.nsmallest(50, "rank_activation_arm_or_post_to_pre")["ko"])
    vs_deg = {
        "coverage_frac": float(cov),
        "overlap_KO_arm_vs_deg50": sorted(top_a & deg_set),
        "overlap_act_arm_vs_deg50": sorted(top_b & deg_set),
        "jaccard_KO": len(top_a & deg_set) / max(len(top_a | deg_set), 1),
        "jaccard_act": len(top_b & deg_set) / max(len(top_b | deg_set), 1),
    }
    (out / "vs_deg_overlap.json").write_text(json.dumps(vs_deg, indent=2))

    focus_rows = []
    s = scored.set_index("ko")["score"]
    dual_i = dual.set_index("ko")
    star_s = pd.Series(star, index=genes)
    for g in FOCUS:
        if g not in gal:
            focus_rows.append({"gene": g, "in_gallery": False})
            continue
        focus_rows.append(
            {
                "gene": g,
                "in_gallery": True,
                "delta": float(star_s[g]) if g in star_s.index else float("nan"),
                "score": float(s.loc[g]),
                "rank_KO_acq": float(dual_i.loc[g, "rank_KO_phenocopy_acquisition"]),
                "rank_activation": float(dual_i.loc[g, "rank_activation_arm_or_post_to_pre"]),
            }
        )
    focus = pd.DataFrame(focus_rows)
    focus.to_csv(out / "focus_genes.tsv", sep="\t", index=False)

    summary = {
        "tag": tag,
        "gallery": gallery_name,
        "n_gallery": len(gal),
        "delta_path": str(delta_path),
        "vs_deg": vs_deg,
        "focus": focus_rows,
        "top10_KO_acq": dual.nsmallest(10, "rank_KO_phenocopy_acquisition")["ko"].tolist(),
        "top10_activation": dual.nsmallest(10, "rank_activation_arm_or_post_to_pre")["ko"].tolist(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["tag", "gallery", "n_gallery", "top10_KO_acq", "top10_activation"]}, indent=2))
    return summary


def main():
    deltas = build_deltas()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Loading GWPS gallery...")
    genes_g, gal_g = load_gwps_deltas()
    print("Loading K562 Essential gallery...")
    genes_e, gal_e = _gallery_essential()

    summaries = []
    for tag, path in deltas.items():
        summaries.append(score_one(tag, path, "k562_gwps", genes_g, gal_g))
        summaries.append(score_one(tag, path, "k562_essential", genes_e, gal_e))

    # compact focus table for paper markers on primary contrast
    rows = []
    for s in summaries:
        for fr in s["focus"]:
            if fr.get("gene") in {"IFITM3", "CD33", "CD36", "ABL1", "BCR", "ITGB1", "MYC", "TP53"}:
                rows.append({"tag": s["tag"], "gallery": s["gallery"], **fr})
    pd.DataFrame(rows).to_csv(OUT_ROOT / "paper_marker_ranks.tsv", sep="\t", index=False)

    readme = """# Blood dual-arm pilot — K562 imatinib (GSE120932) × GWPS / Essential

## Why
HepG2 Essential Task-2 is liver-matched. For blood, the available **bulk genome-wide KO**
resource is Replogle **K562 GWPS** (~9.9k KOs). K562 Essential covers few classic CML/AML
drivers; GWPS is the natural blood catalog.

## Signature
GEO GSE120932 (PMID 32106243): parental K562 vs imatinib-resistant K562-IR.
Acquisition ΔY* = mean(resistant) − mean(parental).

## Claim boundary
Lineage-matched usage / signed calibration pilot — not CML target discovery.
"""
    (OUT_ROOT / "README.md").write_text(readme)
    (OUT_ROOT / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"Done → {OUT_ROOT}")


if __name__ == "__main__":
    main()
