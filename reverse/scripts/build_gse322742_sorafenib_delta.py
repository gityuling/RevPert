#!/usr/bin/env python3
"""Build HepG2/Huh7 sorafenib resistance ΔY* from GSE322742 FPKM CSVs.

ΔY* = mean(log2(FPKM+1) resistant) − mean(log2(FPKM+1) parental)
Ensembl → symbol via mygene.info batch query.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# GSM → (line, condition, rep)
SAMPLE_META = {
    "GSM9557553": ("HepG2", "WT", 1),
    "GSM9557554": ("HepG2", "WT", 2),
    "GSM9557555": ("HepG2", "SR", 1),
    "GSM9557556": ("HepG2", "SR", 2),
    "GSM9557557": ("Huh7", "WT", 1),
    "GSM9557558": ("Huh7", "WT", 2),
    "GSM9557559": ("Huh7", "SR", 1),
    "GSM9557560": ("Huh7", "SR", 2),
}


def _find_sample_files(raw_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(raw_dir.glob("GSM*_FPKM_values.csv.gz")):
        gsm = p.name.split("_")[0]
        if gsm in SAMPLE_META:
            out[gsm] = p
    missing = set(SAMPLE_META) - set(out)
    if missing:
        raise FileNotFoundError(f"Missing GSM files in {raw_dir}: {sorted(missing)}")
    return out


def _load_fpkm_matrix(files: dict[str, Path]) -> pd.DataFrame:
    series = []
    for gsm, path in files.items():
        df = pd.read_csv(path)
        # ensembl_gene_id, <sample>_FPKM
        id_col = df.columns[0]
        val_col = df.columns[1]
        s = df.set_index(id_col)[val_col].astype(float)
        s.name = gsm
        series.append(s)
    mat = pd.concat(series, axis=1).fillna(0.0)
    mat.index = mat.index.astype(str).str.split(".").str[0]
    return mat


def _ensembl_to_symbol(ensembl_ids: list[str], chunk: int = 1000) -> dict[str, str]:
    url = "https://mygene.info/v3/query"
    mapping: dict[str, str] = {}
    for i in range(0, len(ensembl_ids), chunk):
        batch = ensembl_ids[i : i + chunk]
        r = requests.post(
            url,
            json={
                "q": batch,
                "scopes": "ensembl.gene",
                "fields": "symbol",
                "species": "human",
            },
            timeout=120,
        )
        r.raise_for_status()
        for hit in r.json():
            q = hit.get("query")
            sym = hit.get("symbol")
            if q and sym and "notfound" not in hit:
                mapping[str(q)] = str(sym)
        time.sleep(0.2)
    return mapping


def _delta_for_line(log_mat: pd.DataFrame, line: str) -> pd.Series:
    wt = [g for g, (ln, cond, _) in SAMPLE_META.items() if ln == line and cond == "WT"]
    sr = [g for g, (ln, cond, _) in SAMPLE_META.items() if ln == line and cond == "SR"]
    return log_mat[sr].mean(axis=1) - log_mat[wt].mean(axis=1)


def _collapse_to_symbol(delta_ens: pd.Series, mapping: dict[str, str]) -> pd.Series:
    rows = []
    for eid, val in delta_ens.items():
        sym = mapping.get(str(eid))
        if not sym:
            continue
        rows.append((sym, float(val)))
    df = pd.DataFrame(rows, columns=["gene", "value"])
    # if multiple Ensembl → same symbol, take mean
    return df.groupby("gene", as_index=True)["value"].mean().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("reverse/data/signatures/raw_gse322742"),
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reverse/data/signatures"),
    )
    ap.add_argument("--cache_map", type=Path, default=None)
    args = ap.parse_args()

    files = _find_sample_files(args.raw_dir)
    mat = _load_fpkm_matrix(files)
    log_mat = np.log2(mat + 1.0)

    cache = args.cache_map or (args.raw_dir / "ensembl_to_symbol.json")
    if cache.is_file():
        mapping = json.loads(cache.read_text())
    else:
        mapping = _ensembl_to_symbol(list(log_mat.index.astype(str)))
        cache.write_text(json.dumps(mapping, indent=2, sort_keys=True))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source": "GSE322742",
        "url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE322742",
        "definition": "mean(log2(FPKM+1) SR) - mean(log2(FPKM+1) WT)",
        "samples": {k: {"line": v[0], "condition": v[1], "rep": v[2], "file": files[k].name} for k, v in SAMPLE_META.items()},
        "n_ensembl": int(log_mat.shape[0]),
        "n_mapped_symbols": len(mapping),
    }

    for line, tag in [("HepG2", "hepg2_sorafenib"), ("Huh7", "huh7_sorafenib")]:
        delta_ens = _delta_for_line(log_mat, line)
        delta_sym = _collapse_to_symbol(delta_ens, mapping)
        out_tsv = args.out_dir / f"{tag}_delta_y_star.tsv"
        delta_sym.rename("value").reset_index().rename(columns={"index": "gene"}).to_csv(
            out_tsv, sep="\t", index=False
        )
        # fix column name if reset_index used gene already
        pd.DataFrame({"gene": delta_sym.index.astype(str), "value": delta_sym.values}).to_csv(
            out_tsv, sep="\t", index=False
        )
        prov = dict(provenance)
        prov["line"] = line
        prov["n_genes_symbol"] = int(len(delta_sym))
        prov["out_tsv"] = str(out_tsv)
        (args.out_dir / f"{tag}_delta_y_star.provenance.json").write_text(json.dumps(prov, indent=2))
        print(f"{line}: wrote {out_tsv} ({len(delta_sym)} genes)")


if __name__ == "__main__":
    main()
