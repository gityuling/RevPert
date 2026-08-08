#!/usr/bin/env bash
# Freeze RevPert deliverables with SHA-256 for Data availability.
# Run from repo root: bash reverse/scripts/freeze_benchmark_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REV="${ROOT}/reverse"
OUT="${ROOT}/frozen/SHA256SUMS"
MANIFEST_DIR="${ROOT}/frozen"
mkdir -p "${MANIFEST_DIR}"

{
  echo "# RevPert frozen assets — generated $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "# Paths relative to repository root"
  echo
} > "${OUT}"

hash_if_exists() {
  local rel="$1"
  local f="${ROOT}/${rel}"
  if [[ -f "$f" ]]; then
    (cd "${ROOT}" && sha256sum "${rel}") >> "${OUT}"
  else
    echo "# MISSING: ${rel}" >> "${OUT}"
  fi
}

# Protocol / deposit docs
hash_if_exists "frozen/deposit/docs/METHODS_REGISTRY.md"
hash_if_exists "frozen/deposit/docs/TASK1_PROTOCOL.md"
hash_if_exists "frozen/deposit/SHA256SUMS"

# Manuscript numeric tables
for rel in \
  frozen/tables/essential_fair_median_rank.tsv \
  frozen/tables/essential_wilcoxon_bootstrap.tsv \
  frozen/tables/fourteen_dataset_board.tsv \
  frozen/tables/genetic_main_table.tsv \
  frozen/tables/leaderboard_snapshot.json \
  frozen/tables/pdgrapher_fairness_perline.tsv \
  frozen/tables/pdgrapher_fairness_protocol.tsv \
  frozen/deposit/tables/fourteen_dataset_board.tsv \
  frozen/deposit/essential_ranks/all_seed1_ranks.tsv \
  frozen/deposit/genetic/summary_by_cell.tsv
do
  hash_if_exists "$rel"
done

# Processed signatures used in the paper
hash_if_exists "reverse/data/signatures/hepg2_sorafenib_delta_y_star.tsv"
hash_if_exists "reverse/data/signatures/gse143233_resistant_minus_normal_delta_y_star.tsv"
hash_if_exists "reverse/data/signatures/gse120932_k562_ir_with_drug_delta_y_star.tsv"

echo "Wrote ${OUT}"
wc -l "${OUT}"
