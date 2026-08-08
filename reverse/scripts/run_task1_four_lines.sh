#!/usr/bin/env bash
# Run Essential Task-1 on four lines: train RevPert → fair compare → aggregate.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
EPOCHS="${EPOCHS:-50}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CELLS="${CELLS:-hepg2 k562 rpe1 jurkat}"

echo "[task1] root=$ROOT python=$PY epochs=$EPOCHS cells=$CELLS"

for cell in $CELLS; do
  out="reverse/results/retrieval_${cell}_v2"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out/best.pt" && -f "$out/test_summary.tsv" ]]; then
    echo "[task1] skip train $cell (found $out/best.pt)"
  else
    echo "[task1] train $cell → $out"
    "$PY" -m reverse.src.train_reverse_retrieval \
      --cell_line "$cell" --force_cell_outdir \
      --epochs "$EPOCHS" --device "$DEVICE"
  fi

  fair="reverse/results/fair_compare_${cell}"
  if [[ "$SKIP_EXISTING" == "1" && -f "$fair/fair_compare_summary.tsv" ]]; then
    echo "[task1] skip fair $cell (found summary)"
  else
    echo "[task1] fair compare $cell"
    "$PY" -m reverse.src.compare_fair_baselines \
      --cell_line "$cell" --device "$DEVICE"
  fi
done

echo "[task1] aggregate + stats"
"$PY" -m reverse.src.summarize_fair_all

echo "[task1] figures"
"$PY" -m reverse.src.plot_task1_figures

echo "[task1] done → reverse/results/fair_compare_all/ reverse/results/figures_task1/"
