# Upload to GitHub

```bash
# Option A — from the assembled tree
cd RevPert
bash docs/run_preflight.sh
git init
git add .
git commit -m "Initial public release of RevPert analysis code"
git branch -M main
git remote add origin git@github.com:<ORG_OR_USER>/RevPert.git
git push -u origin main

# Option B — unpack the tarball into a fresh clone
# tar -xzf RevPert_github_release.tar.gz
```

Do **not** push the parent `linearbaseline` monorepo: it contains private paths,
archived dual-encoder trees, large raw results, and manuscript working files.
