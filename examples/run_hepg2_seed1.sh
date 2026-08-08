#!/usr/bin/env bash
# Minimal RevPert identity + HCC dual-arm run (requires Perturb-seq galleries).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# Optional: export REVPERT_BENCH_ROOT=/path/to/.../benchmark
python reverse/scripts/run_gallery_dual_g2.py \
  --cell_line hepg2 \
  --seed 1 \
  --epochs 50 \
  --skip_gwps 1 \
  --device cuda \
  --out_dir reverse/results/revpert/essential/hepg2/seed1
