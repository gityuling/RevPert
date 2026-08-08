# Anonymous RevPert reproducibility deposit

Double-blind peer-review package for **RevPert** (gallery-native residual reverse
perturbation scoring). Author names and institutional paths are withheld.

## Contents

| Path | Description |
|------|-------------|
| `tables/essential_wilcoxon_bootstrap.tsv` | Seed-1 paired Wilcoxon + bootstrap CI (RevPert vs Pearson) |
| `tables/pdgrapher_fairness_*.tsv` | Protocol audit + per-line fold vs official PDGrapher |
| `tables/fourteen_dataset_board.tsv` | Main 14-dataset board (Essential + genetic) |
| `essential_ranks/` | Per-query ranks for Essential seed 1 |
| `genetic/` | Fold metrics and cell-line summaries |
| `scripts/` | Table / stats builders (no private credentials) |
| `docs/` | Method registry and Task-1 protocol |

## Recompute Essential Wilcoxon (requires full analysis environment)

```bash
# From the full analysis repository (not this deposit alone):
PYTHONPATH=. python reverse/scripts/build_revpert_stats_and_deposit.py
```

This deposit ships the **frozen numeric outputs** used in the manuscript SI.
Model checkpoints are not deposited here (stated in Data availability).

## Claim boundaries

- Essential primary baseline: Pearson on linear L3 predicted gallery.
- Genetic primary baseline: official PDGrapher genetic model under the same
  median-rank / Recall@10 interface.
- Resources are **not** pooled into one leaderboard.
- PDGrapher comparison is fair on splits/metrics/catalog restriction; model
  classes differ (residual gallery scorer vs GNN discovery).

## Checksums

See `SHA256SUMS` in this directory.
