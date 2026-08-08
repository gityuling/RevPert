# Methods registry — ReversePerturb-Bench

Fair inclusion rule: same Task-1 split, same catalog membership policy, same **main-table**
metrics (median rank, Top-10). MRR may appear in SI / training selection only.
Otherwise → Related work / protocol note only.

**NCS freeze:** do not expand the fair table with additional foundation-model galleries
(Geneformer, STATE, …). TxPert/scGPT/GEARS remain gallery *builders*, not native reverse methods.
See `manuscript/NCS_MINIMAL_PACKAGE.md`.

| ID | Method | Role | Fair on Task-1? | Implementation | Status |
|----|--------|------|-----------------|----------------|--------|
| B1 | pearson_pred_gallery | CMap-style matcher | Yes | `compare_fair_baselines.py` | done |
| B2 | cmap_lite_pred_gallery | Up/down enrichment | Yes | same | done |
| B3 | gem_lite_topDEG_corr | GEM-inspired proxy | Partial (not full GEM) | same | done |
| B4 | ridge_delta_to_P | Linear invert | Yes | same | done |
| B5 | esm_mean_topDEG_to_KO | Weak prior | Yes | same | done |
| B6 | dual_encoder_v2 | Learned reverse retrieval | Yes | `train_reverse_retrieval.py` | done |
| B7 | pearson_obs_gallery_oracle | Upper bound | No (leak) | same | SI only |
| G1 | linear_L3 gallery | Forward gallery | Yes (builder) | L3 + gallery compare | done |
| G2 | GEARS gallery | Forward gallery | Yes (builder) | `run_txpert_gallery_recovery.py` | four-line done |
| G3 | scGPT gallery | Forward gallery | Yes (builder; missing KO → worst rank) | `run_txpert_gallery_recovery.py` | **four-line done** |
| G4 | TxPert-GAT gallery | Forward gallery | Yes (builder) | matched `txpert_gat_seed1` | **four-line done** |
| G5 | TxPert x-cell LOO gallery | Forward gallery | Yes (builder) | matched `txpert_gat_xcell_loo_seed1` | **four-line done** |
| E1 | CRISPR-GEM full | Phenotype-shift MLP | Likely no | external | deferred |
| E2 | GenePerturbR reverse | Bulk genetic DB match | Task-2 maybe | external | deferred |
| E3 | PDGrapher | Combinatorial targets | No (different task) | — | related work only |
| E4 | CMap/CLUE genetic | External genetic reagents | Task-2 optional | — | deferred |

**Note:** TxPert has no native reverse task; G4/G5 use TxPert only as \(\widehat{\Delta Y}\) gallery builders.

**Catalog policy:** All gallery builders (including scGPT) are scored on the same shared KO catalog and test queries. If a forward model lacks a predicted profile for a catalog KO, that entry receives worst rank (coverage failure is part of the score).

**Multi-seed:** Task-1 dual + Pearson fair compare completed for seeds 1–5 (`results/fair_compare_multiseed/`).

Last updated: 2026-07-22 (NCS package: main metrics median rank + Top-10; oracle SI-only).
