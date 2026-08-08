# Task-1 protocol — held-out KO identity recovery

**Goal:** Rank catalog genetic interventions for a held-out query expression
shift under a shared split and catalog (RevPert primary method).

## 1. Task definition

| Item | Definition |
|------|------------|
| **Query** | Observed pseudo-bulk knockout response \(\Delta Y(g^\star)\) for a held-out test KO \(g^\star\) |
| **Catalog** | Candidate KOs with forward-predicted or train-fold \(\widehat{\Delta Y}(g)\) |
| **Output** | Ranking of catalog KOs; success = recovering the true \(g^\star\) |
| **Main-table metrics** | Median rank, Recall@10 (\%) |
| **SI / diagnostics** | Mean rank, Top1 / Top100, MRR |

Lower median rank is better. Oracle upper bound (observed profiles in the gallery) is **SI-only**.

## 2. Data and split

- **Atlas:** Replogle Essential Perturb-seq (K562, RPE1, HepG2, Jurkat).
- **Split:** GEARS simulation seed-1 `train` / `val` / `test` under the companion benchmark tree.
- **Forward gallery:** Linear L3 progressive-stack predictions (primary fair gallery).
- **Queries:** control-subtracted pseudo-bulk \(\Delta Y\) aligned to the gallery gene list.

Only KOs present in **both** the observed \(\Delta Y\) table and the gallery enter ranking.

## 3. Leakage boundary (hard rules)

1. RevPert InfoNCE uses **train KOs only**.
2. \(\Delta Y\) PCA is **fit on train profiles only**, then applied to val/test.
3. Held-out observed responses are **never** used as gallery vectors.
4. `pearson_obs_gallery_oracle` puts observed profiles (including test) in the gallery → **upper bound only**.

## 4. Methods on the fair leaderboard

| Method | Role |
|--------|------|
| `pearson_pred_gallery` | Pearson(query obs \(\Delta Y\), L3 predicted \(\widehat{\Delta Y}\)) |
| `cmap_lite_pred_gallery` | Up/down set enrichment vs predicted gallery |
| `gem_lite_topDEG_corr` | Corr on query top-\|\(\Delta\)\| subspace |
| `ridge_delta_to_P` | Ridge \(\Delta Y \to P\) on train; nearest catalog \(P\) |
| **`revpert`** | \(z(\mathrm{Pearson})+\alpha\,z(\mathrm{learned\ similarity})\); fused InfoNCE |

Forward-gallery swaps under Pearson matching (TxPert, GEARS, scGPT, UniPert-ridge) are sensitivity rows, not a second primary claim.

## 5. Statistics (minimum)

- Per line: point estimates for all fair methods.
- **Primary contrast:** RevPert vs Pearson on paired per-KO ranks (Wilcoxon + bootstrap CI on median-rank difference).

## 6. Reproducibility

```bash
bash examples/run_hepg2_seed1.sh
bash examples/run_essential_matrix.sh
```

Set `REVPERT_BENCH_ROOT` to the companion Perturb-seq / L3 gallery tree (see root README).
