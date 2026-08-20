# Ranking tables

Numeric exports used in the manuscript: Essential ranks, LINCS / PDGrapher
summaries, and Wilcoxon tables.

## Contents

| Path | Description |
|------|-------------|
| `tables/essential_wilcoxon_bootstrap.tsv` | Seed-1 paired Wilcoxon + bootstrap CI (RevPert vs Pearson) |
| `tables/pdgrapher_fairness_*.tsv` | Protocol audit + per-line fold vs official PDGrapher |
| `tables/fourteen_dataset_board.tsv` | Main 14-dataset board (Essential + genetic) |
| `essential_ranks/` | Per-query ranks for Essential seed 1 |
| `genetic/` | Fold metrics and cell-line summaries |
| `scripts/` | Table / stats builders (no private credentials) |
| `docs/` | Method registry and identity-recovery protocol |

## Recompute Essential Wilcoxon (requires full analysis environment)

```bash
PYTHONPATH=. python reverse/scripts/build_revpert_stats_and_deposit.py
```

Model checkpoints are not deposited here.

## Claim boundaries

- Essential primary baseline: Pearson on linear L3 predicted gallery.
- Genetic primary baseline: official PDGrapher genetic model under the same
  median-rank / Recall@10 interface.
- Resources are **not** pooled into one leaderboard.
- PDGrapher comparison is fair on splits/metrics/catalog restriction; model
  classes differ (residual gallery scorer vs GNN discovery).

## Checksums

See `SHA256SUMS` in this directory.
