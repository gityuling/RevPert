#!/usr/bin/env bash
# Freeze Task-1 (and optional Task-2) deliverables with SHA-256 for Data availability.
# Run from repo root: bash reverse/scripts/freeze_benchmark_assets.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REV="${ROOT}/reverse"
OUT="${REV}/benchmark/SHA256SUMS"
MANIFEST_DIR="${REV}/benchmark/frozen_lists"
mkdir -p "${MANIFEST_DIR}"

{
  echo "# ReversePerturb-Bench frozen assets — generated $(date -u +%Y-%m-%dT%H:%MZ)"
  echo "# Paths relative to reverse/"
  echo
} > "${OUT}"

hash_if_exists() {
  local rel="$1"
  local f="${REV}/${rel}"
  if [[ -f "$f" ]]; then
    (cd "${REV}" && sha256sum "${rel}") >> "${OUT}"
  else
    echo "# MISSING: ${rel}" >> "${OUT}"
  fi
}

# Protocol / registry
hash_if_exists "benchmark/METHODS_REGISTRY.md"
hash_if_exists "dual_encoder_v2/manuscript/docs/TASK1_PROTOCOL.md"
hash_if_exists "manuscript/NCS_MINIMAL_PACKAGE.md"

# Fair / gallery / multiseed summaries
# (Dual-encoder fair_compare_* lives under dual_encoder_v2/; RevPert tables under results/revpert/)
for rel in \
  dual_encoder_v2/results/fair_compare_all/summary_median_rank.tsv \
  dual_encoder_v2/results/fair_compare_all/summary_top10.tsv \
  dual_encoder_v2/results/fair_compare_all/dual_vs_pearson_stats.tsv \
  dual_encoder_v2/results/fair_compare_multiseed/summary_seed_mean_std.tsv \
  results/gallery_compare_all/summary_median_rank.tsv \
  results/tables_task1/leaderboard_snapshot.json \
  results/tables_task1/table_fair_seed1_median_rank.tsv \
  results/tables_task1/table_fair_seed1_top10.tsv \
  results/tables_task1/table_gallery_seed1_median_rank.tsv \
  results/tables_task1/table_multiseed_mean_std.tsv \
  results/tables_task1/table_dual_vs_pearson_stats.tsv
do
  hash_if_exists "$rel"
done

# Optional Task-2 (real signature when present)
hash_if_exists "data/signatures/lihc_delta_y_star.tsv"
hash_if_exists "results/disease_lihc/reverse_scores.tsv"
hash_if_exists "results/disease_lihc/vs_deg_overlap.json"
hash_if_exists "results/disease_lihc/gene_set_enrichment.tsv"
hash_if_exists "data/signatures/gse14520_delta_y_star.tsv"
hash_if_exists "results/disease_lihc_gse14520/reverse_scores.tsv"
hash_if_exists "results/disease_lihc_cohort_compare/tcga_vs_gse14520_compare.json"
hash_if_exists "results/disease_lihc_cohort_compare/tcga_patient_split_compare.json"

# Dual-encoder checkpoints (archived under dual_encoder_v2/)
shopt -s nullglob
CKPTS=(
  "${REV}"/dual_encoder_v2/results/retrieval_hepg2_v2/best.pt
  "${REV}"/dual_encoder_v2/results/retrieval_k562_v2/best.pt
  "${REV}"/dual_encoder_v2/results/retrieval_rpe1_v2/best.pt
  "${REV}"/dual_encoder_v2/results/retrieval_jurkat_v2/best.pt
  "${REV}"/dual_encoder_v2/results/retrieval_*_seed1_v2/best.pt
)
n_ckpt=0
for f in "${CKPTS[@]}"; do
  [[ -f "$f" ]] || continue
  rel="${f#${REV}/}"
  (cd "${REV}" && sha256sum "${rel}") >> "${OUT}"
  n_ckpt=$((n_ckpt + 1))
done
if ((n_ckpt == 0)); then
  echo "# NOTE: no dual-encoder best.pt under dual_encoder_v2/results/retrieval_*_v2" >> "${OUT}"
fi

# Copy a short freeze note
cat > "${MANIFEST_DIR}/README.md" <<'EOF'
# Frozen lists

`../SHA256SUMS` is the deposit checklist for Data availability.
Re-run `bash reverse/scripts/freeze_benchmark_assets.sh` after regenerating tables or Task-2 outputs.
Do not edit SHA256SUMS by hand.
EOF

echo "Wrote ${OUT}"
wc -l "${OUT}"
