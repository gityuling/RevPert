# Methods registry — RevPert identity recovery

Fair inclusion rule: same held-out split, same catalog membership policy, same
**main-table** metrics (median rank, Recall@10 / Top-10). MRR may appear in SI /
training selection only.

| ID | Method | Role | Fair on identity? | Implementation | Status |
|----|--------|------|-----------------|----------------|--------|
| B1 | pearson_pred_gallery | CMap-style matcher | Yes | `compare_fair_baselines.py` | done |
| B2 | cmap_lite_pred_gallery | Up/down enrichment | Yes | same | done |
| B3 | gem_lite_topDEG_corr | GEM-inspired proxy | Partial (not full GEM) | same | done |
| B4 | ridge_delta_to_P | Linear invert | Yes | same | done |
| B5 | esm_mean_topDEG_to_KO | Weak prior | Yes | same | done |
| B6 | **revpert** | Gallery-native residual reverse scorer | Yes | `run_gallery_dual_g2.py` | done |
| B7 | pearson_obs_gallery_oracle | Upper bound | No (leak) | same | SI only |
| G1 | linear_L3 gallery | Forward gallery | Yes (builder) | L3 + gallery compare | done |
| G2 | GEARS gallery | Forward gallery | Yes (builder) | gallery recovery scripts | done |
| G3 | scGPT gallery | Forward gallery | Yes (builder; missing KO → worst rank) | gallery recovery scripts | done |
| G4 | TxPert-GAT gallery | Forward gallery | Yes (builder) | gallery recovery scripts | done |
| G5 | TxPert x-cell LOO gallery | Forward gallery | Yes (builder) | gallery recovery scripts | done |
| G6 | UniPert-ridge gallery | Forward gallery | Yes (builder) | UniPert probe scripts | done |

**Note:** TxPert / GEARS / scGPT / UniPert enter as predicted-gallery builders under Pearson matching, not as dedicated reverse models.
