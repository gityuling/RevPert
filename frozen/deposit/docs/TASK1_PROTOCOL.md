# Task-1 protocol — held-out KO identity recovery

**Benchmark name:** Reverse Perturbation Task-1  
**Goal:** Formalize *signature → genetic intervention* as a reproducible retrieval task and compare fair baselines under a shared split and catalog.

## 1. Task definition

| Item | Definition |
|------|------------|
| **Query** | Observed pseudo-bulk knockout response \(\Delta Y(g^\star)\) for a held-out test KO \(g^\star\) |
| **Catalog** | Candidate KOs with atlas perturbation coordinates \(P(g)\) and/or forward-predicted \(\widehat{\Delta Y}(g)\) |
| **Output** | Ranking of catalog KOs; success = recovering the true \(g^\star\) |
| **Main-table metrics** | Median rank, Top-10 (\%) |
| **SI / diagnostics** | Mean rank, Top1 / Top50 / Top100, MRR \(=\mathrm{mean}(1/\mathrm{rank})\) |

Lower median rank is better. Oracle upper bound (observed profiles in the gallery) is **SI-only** — excluded from the main leaderboard.

## 2. Data and split

- **Atlas:** Replogle Essential Perturb-seq (K562, RPE1, HepG2, Jurkat).
- **Split:** GEARS simulation **seed-1** `train` / `val` / `test` JSON under  
  `benchmark/working_dir/results/seed_1_replogle_{cell}_essential_split`.
- **Forward gallery:** Linear L3 progressive-stack predictions  
  `.../progressive_stack_fulltest/replogle_{cell}_essential__prog_L3_* / all_predictions.json`.
- **Perturbation coordinates \(P\):** L3 reference PCA TSV  
  `.../reference_pca/replogle_{cell}_essential__prog_3ref_*_30d.tsv`.
- **Queries:** `all_pseudobulk_deltas.h5ad` aligned to gallery gene list.

Only KOs present in **both** observed \(\Delta Y\) and \(P\) enter train/val/test tables.

## 3. Leakage boundary (hard rules)

1. Dual-encoder InfoNCE uses **train KOs only**.
2. \(\Delta Y\) PCA for the dual encoder is **fit on train only**, then applied to val/test.
3. Retrieval prototypes use atlas \(P\) (and optional ESM gene prior), **not** test observed \(\Delta Y\) as gallery prototypes.
4. `pearson_obs_gallery_oracle` puts observed profiles (including test) in the gallery → **upper bound only**.
5. Forward L3 \(\widehat{\Delta Y}\) may use cross-line measured coordinates (same as the deployable forward paper); reverse methods must not peek at the query’s own held-out identity beyond the shared catalog membership.

## 4. Methods on the fair leaderboard

| Method | Role |
|--------|------|
| `pearson_pred_gallery` | CMap-style: Pearson(query obs \(\Delta Y\), L3 predicted \(\widehat{\Delta Y}\)) |
| `cmap_lite_pred_gallery` | Up/down set enrichment vs predicted gallery |
| `gem_lite_topDEG_corr` | GEM-inspired proxy: corr on query top-\|\(\Delta\)\| subspace (not full GEM) |
| `ridge_delta_to_P` | Ridge \(\Delta Y \to P\) on train; nearest catalog \(P\) |
| `esm_mean_topDEG_to_KO` | Average ESM of top-\|\(\Delta\)\| genes → match KO ESM (weak prior) |
| **`dual_encoder_v2`** | Trainable reverse retrieval: \(f(\mathrm{PCA}\Delta Y)\) ↔ \(h(P\|\mathrm{ESM})\), InfoNCE + hard negatives |

**Not apples-to-apples (appendix / notes only):** full CRISPR-GEM phenotype MLP; GEARS/scGPT alone as forward models (usable as alternate gallery builders).

## 5. Cell lines (Phase 1)

Run the **same** protocol independently on:

- HepG2, K562, RPE1, Jurkat (each with its L3 gallery and seed-1 split).

Aggregate table: `reverse/results/fair_compare_all/summary.tsv`.

## 6. Statistics (minimum)

- Per line: point estimates for all fair methods.
- **Primary contrast:** `dual_encoder_v2` vs `pearson_pred_gallery` on paired per-KO ranks:
  - Wilcoxon signed-rank (or sign test if needed)
  - Bootstrap 95% CI on median-rank difference (resampling queries)

## 7. Phase 2 (out of Task-1 scope, same repo)

Disease/control bulk \(\Delta Y^\star\) → gene prioritization, vs DEG, nulls, DepMap/driver enrichment. See `PHASE2_DISEASE.md`.

## 8. Reproducibility

```bash
bash reverse/scripts/run_task1_four_lines.sh
```

Requires `gears_env2` (PyTorch) and the forward L3 artifacts under `linear_perturbation_prediction-Paper-main/benchmark/`.
