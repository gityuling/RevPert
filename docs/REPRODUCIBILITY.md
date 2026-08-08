# Reproducibility

## Software

- Python 3.10, PyTorch 2.x with CUDA recommended for training
- See `environment.yml` / `requirements.txt`

## Determinism

Training scripts accept `--seed` for data splits. GPU reductions may still vary
slightly across drivers; report median rank to the nearest integer as in the paper.

## Minimal CPU smoke test (no training)

```bash
python -c "from reverse.src.delta_y_star import load_vector_tsv; \
from pathlib import Path; \
v=load_vector_tsv(Path('reverse/data/signatures/hepg2_sorafenib_delta_y_star.tsv')); \
print(len(v))"
```

## Full Essential matrix

```bash
bash examples/run_essential_matrix.sh
```

Wall-clock depends on GPU; four lines × five seeds is a multi-hour job.

## Genetic PDGrapher comparison

Requires the PDGrapher genetic resources configured under the companion benchmark
tree. Entry point: `reverse/scripts/run_pdgrapher_gallery_fuse.py`.

## What is not required to verify table numbers

`frozen/tables/` already contains the SI numeric exports. Retraining is needed
only to regenerate checkpoints or to audit end-to-end training.
