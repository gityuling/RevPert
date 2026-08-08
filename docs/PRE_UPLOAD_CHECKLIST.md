# Pre-upload checklist

Run before `git push` to a public GitHub repository.

## Must pass

- [ ] No absolute personal paths (`/home/...`, `C:\Users\...`)
- [ ] No API keys, tokens, passwords, `.env` files
- [ ] No Chinese (or other non-English) in README / docs / user-facing strings
- [ ] No dual-encoder checkpoints or `dual_encoder_v2` tree
- [ ] No manuscript author metadata you are not ready to reveal (double-blind)
- [ ] No huge binaries: `*.h5ad`, `*.pt`, raw GEO tarballs (gitignored)
- [ ] `LICENSE` present; third-party data licenses acknowledged in README
- [ ] `REVPERT_BENCH_ROOT` documented for external galleries
- [ ] Smoke test: `python -c "import reverse.src.cell_lines"` from repo root

## Recommended

- [ ] Add remote with a neutral repo name (e.g. `RevPert`)
- [ ] Tag a release that matches the frozen table checksums
- [ ] Optional Zenodo DOI for the code snapshot cited in Data/Code availability
- [ ] Strip local IDE / Cursor / agent config (`.cursor/`, `.vscode/` private notes)
- [ ] Confirm `git status` has no unintended `results/` or notebook outputs

## Automated scan

```bash
bash docs/run_preflight.sh
```
