#!/usr/bin/env bash
# Run TxPert/GEARS/scGPT/linear gallery reverse recovery on four Essential lines (seed-1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
CELLS="${CELLS:-hepg2 k562 rpe1 jurkat}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"

for cell in $CELLS; do
  echo "======== gallery_compare $cell seed$SEED ========"
  "$PY" -m reverse.src.run_txpert_gallery_recovery --cell_line "$cell" --seed "$SEED" --device "$DEVICE"
done

echo "======== aggregate ========"
"$PY" -m reverse.src.summarize_gallery_compare
echo done
