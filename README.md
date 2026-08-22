# RevPert: predicting candidate drivers of transcriptomic state transitions via gallery-native reverse perturbation

**RevPert** predicts candidate drivers from a fixed knockout catalog for an observed transcriptomic
contrast (`delta_y_star = YB - YA`). It combines signed Pearson connectivity
with a learned residual on a predicted gallery and returns a directional
shortlist, not a causal target list.

This repository contains analysis code, processed query signatures, ranking
tables, and scripts to retrain or rescore RevPert.

## Install

```bash
conda env create -f environment.yml
conda activate revpert
# or: pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## License (software vs data)

- **MIT** (`LICENSE`) covers **this repository's code, docs, and ranking tables we computed**.
- Third-party datasets (Replogle Perturb-seq, NCBI GEO/SRA, TCGA/UCSC Xena, PDGrapher) keep **their original terms**.
- This repo does **not** re-license, mirror, or claim ownership of those datasets.
- Download them from the providers below and cite the original papers / accessions.
- RevPert outputs are **candidate-driver predictions** for follow-up, not wet-validated causal or therapeutic claims, and not a held-out identity claim on screen-external signatures.

## Data downloads

### 1. Perturb-seq galleries (required for training / identity)

Large files — **not** shipped here. After preparing a GEARS-compatible layout:

```bash
export REVPERT_BENCH_ROOT=/path/to/.../benchmark
# optional: export REVPERT_REPO_ROOT=$PWD
```

Expected layout:

```text
$REVPERT_BENCH_ROOT/
  data/gears_pert_data/
    replogle_hepg2_essential/
    replogle_k562_essential/
    replogle_rpe1_essential/
    replogle_jurkat_essential/
    replogle_k562_gwps/all_pseudobulk_deltas.h5ad   # CML
  working_dir/results/
    progressive_stack_fulltest/   # linear predicted galleries
    seed_*_replogle_*_split
```

| Resource | Links |
|----------|-------|
| Replogle et al. 2022 (paper) | https://doi.org/10.1016/j.cell.2022.05.013 |
| Processed AnnData (Figshare+) | https://doi.org/10.25452/figshare.plus.20029387 |
| Figshare landing page | https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387 |
| Raw SRA BioProject | https://www.ncbi.nlm.nih.gov/bioproject/PRJNA831566 |
| SRA/GEO file manifest | https://doi.org/10.25452/figshare.plus.20022944 |

Figshare+ provides K562 GWPS, K562 Essential, and RPE1 Essential processed products.
HepG2 / Jurkat Essential and linear predicted galleries should be placed under the
paths above using the same GEARS-style packaging as your forward benchmark; cite
Replogle et al. and the packaging source you use.

### 2. Screen-external signatures (processed vectors shipped here)

Processed `delta_y_star` files: `reverse/data/signatures/*.tsv`  
Provenance (accession + contrast): matching `*.provenance.json`

| Accession | Role | Download |
|-----------|------|----------|
| GSE322742 | HepG2/Huh7 sorafenib SR − parental | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE322742 |
| GSE143233 | Patient SR HCC − normal | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143233 |
| GSE120932 | K562-IR − parental (×3 contrasts) | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120932 |
| GSE121153 | Huh7 / xenograft sorafenib | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121153 |
| GSE14520 | HCC tumor − non-tumor | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14520 |
| GSE145389 | HepG2 palbociclib | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145389 |
| GSE158552 | HepG2 JQ1 / OTX015 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE158552 |
| GSE159164 | Huh7 rapamycin | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159164 |
| GSE186191 | Huh7 / Hep3B lenvatinib | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186191 |
| GSE193094 | HL-60 chemoresistant − parental | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193094 |
| GSE200098 | Huh7 sorafenib resistant | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200098 |

GEO supplementary FTP pattern (example for GSE322742):

```text
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE322nnn/GSE322742/suppl/
```

Rebuild helper for GSE322742: `reverse/scripts/build_gse322742_sorafenib_delta.py`

**TCGA-LIHC (usage check):**

| Asset | URL |
|-------|-----|
| Expression HiSeqV2 | https://tcga.xenahubs.net/download/TCGA.LIHC.sampleMap/HiSeqV2.gz |
| Clinical matrix | https://tcga.xenahubs.net/download/TCGA.LIHC.sampleMap/LIHC_clinicalMatrix |
| Xena browser | https://xenabrowser.net/ |

### 3. PDGrapher genetic comparison (optional)

Obtain official PDGrapher genetic models / folds from the PDGrapher authors' release
(cite Gonzalez et al.). Weights are not redistributed here. Entry point:
`reverse/scripts/run_pdgrapher_gallery_fuse.py`.

### Minimum citations for data you use

1. Replogle et al., Cell 2022 — https://doi.org/10.1016/j.cell.2022.05.013  
2. Each GEO accession you analyze  
3. TCGA / UCSC Xena if using LIHC  
4. PDGrapher if reporting genetic comparisons  
5. This RevPert software (`CITATION.cff`)

## Quick start

```bash
bash examples/run_hepg2_seed1.sh
python reverse/scripts/run_revpert_gwps_cml.py --epochs 30 --device cuda
python reverse/scripts/build_revpert_stats_and_deposit.py
```

Ranking tables: `frozen/tables/`.

## Repository layout

| Path | Role |
|------|------|
| `reverse/src/` | Core library |
| `reverse/scripts/` | Training, Essential matrix, GWPS CML, signature builders |
| `reverse/data/signatures/` | Processed `delta_y_star` vectors + provenance JSON |
| `reverse/data/gene_sets/` | Focus / enrichment gene lists |
| `frozen/tables/` | Manuscript ranking tables |
| `docs/` | Reproducibility notes |
| `examples/` | Shell entry points |

## Claim boundaries

- Candidate-driver **predictions**, not wet-validated causality.
- Primary identity metrics: median rank and Recall@10.
- Screen-external HCC/CML results are a signed-geometry check on pre-specified anchors, not held-out recovery.

## Citation

See `CITATION.cff`. A preprint DOI will be added when posted.
Also cite upstream data providers listed above.
