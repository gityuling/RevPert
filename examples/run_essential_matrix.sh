#!/usr/bin/env bash
# Four Essential lines × seeds 1–5 (long GPU job).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python reverse/scripts/run_revpert_essential_matrix.py --device cuda
