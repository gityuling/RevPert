#!/usr/bin/env python3
"""CLI: reverse-perturbation gene ranking from an existing linear prediction gallery.

Examples
--------
# Oracle smoke test (HepG2 L3 predictions; ΔY* = observed KO response):
python -m reverse.src.run_reverse oracle \\
  --pred_dir .../replogle_hepg2_essential__prog_L3_k562_rpe1_jurkat \\
  --dataset_h5ad .../replogle_hepg2_essential/perturb_processed.h5ad \\
  --pseudobulk_deltas .../replogle_hepg2_essential/all_pseudobulk_deltas.h5ad \\
  --true_ko AAAS \\
  --out_dir reverse/results/oracle_hepg2_AAAS

# Score a disease–control ΔY* TSV against the same gallery:
python -m reverse.src.run_reverse score \\
  --pred_dir .../L3... \\
  --dataset_h5ad .../perturb_processed.h5ad \\
  --delta_y_star path/to/delta_y_star.tsv \\
  --out_dir reverse/results/lihc_run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow `python reverse/src/run_reverse.py` and `python -m reverse.src.run_reverse`
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.benchmarks import (  # noqa: E402
    oracle_recovery_rank,
    random_null_max_score,
    sign_flip_scores,
)
from reverse.src.delta_y_star import (  # noqa: E402
    align_delta_y_star,
    delta_y_star_from_gallery_ko,
    delta_y_star_from_two_profiles,
    load_vector_tsv,
    save_delta_y_star,
)
from reverse.src.io_gallery import (  # noqa: E402
    absolute_to_delta,
    load_ctrl_from_perturb_processed,
    load_coverage_from_pca_tsv,
    load_observed_deltas,
    load_prediction_dir,
)
from reverse.src.score import score_gallery, top_k  # noqa: E402

BENCH = _ROOT / "linear_perturbation_prediction-Paper-main" / "benchmark"
DEFAULT_HEPG2_PRED = (
    BENCH
    / "working_dir/results/progressive_stack_fulltest"
    / "replogle_hepg2_essential__prog_L3_k562_rpe1_jurkat"
)
DEFAULT_HEPG2_H5AD = BENCH / "data/gears_pert_data/replogle_hepg2_essential/perturb_processed.h5ad"
DEFAULT_HEPG2_DELTA = (
    BENCH / "data/gears_pert_data/replogle_hepg2_essential/all_pseudobulk_deltas.h5ad"
)


def _build_gallery(pred_dir: Path, dataset_h5ad: Path):
    genes, pred_abs = load_prediction_dir(pred_dir)
    ctrl = load_ctrl_from_perturb_processed(dataset_h5ad, genes)
    gallery = absolute_to_delta(pred_abs, ctrl)
    return genes, gallery


def _maybe_coverage(pca_tsv: str | None) -> set[str] | None:
    if not pca_tsv:
        return None
    return load_coverage_from_pca_tsv(Path(pca_tsv))


def cmd_score(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genes, gallery = _build_gallery(Path(args.pred_dir), Path(args.dataset_h5ad))
    covered = _maybe_coverage(args.coverage_pca_tsv)

    if args.delta_y_star:
        star = load_vector_tsv(Path(args.delta_y_star))
    elif args.disease_tsv and args.control_tsv:
        disease = load_vector_tsv(Path(args.disease_tsv))
        control = load_vector_tsv(Path(args.control_tsv))
        star = delta_y_star_from_two_profiles(disease, control, mode=args.delta_mode)
        save_delta_y_star(out / "delta_y_star.tsv", star)
    else:
        raise SystemExit("Provide --delta_y_star or both --disease_tsv and --control_tsv")

    star_vec = align_delta_y_star(star, genes)
    scored = score_gallery(
        gallery,
        star_vec,
        genes,
        metric=args.metric,
        covered=covered,
        primary_only_covered=args.primary_only_covered,
    )
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=args.top_k, covered_only=bool(covered)).to_csv(
        out / f"top{args.top_k}.tsv", sep="\t", index=False
    )

    flip = sign_flip_scores(gallery, star_vec, genes, metric=args.metric, covered=covered)
    flip.to_csv(out / "reverse_scores_signflip.tsv", sep="\t", index=False)

    if args.n_random > 0:
        null = random_null_max_score(
            gallery,
            star_vec,
            genes,
            n_rand=args.n_random,
            seed=args.seed,
            metric=args.metric,
            covered=covered,
        )
        null.to_csv(out / "random_null_max_score.tsv", sep="\t", index=False)
        summary = {
            "n_gallery": len(scored),
            "n_covered": int(scored["covered"].sum()) if covered is not None else len(scored),
            "max_score": float(scored["score"].max()),
            "random_max_score_mean": float(null["max_score"].mean()),
            "random_max_score_p95": float(null["max_score"].quantile(0.95)),
            "metric": args.metric,
        }
    else:
        summary = {
            "n_gallery": len(scored),
            "max_score": float(scored["score"].max()),
            "metric": args.metric,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out / 'reverse_scores.tsv'}")


def cmd_oracle(args: argparse.Namespace) -> None:
    """ΔY* = observed (or predicted) response of --true_ko; report recovery rank."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genes, gallery_pred = _build_gallery(Path(args.pred_dir), Path(args.dataset_h5ad))
    covered = _maybe_coverage(args.coverage_pca_tsv)

    if args.star_source == "observed":
        _, obs = load_observed_deltas(Path(args.pseudobulk_deltas), genes)
        if args.true_ko not in obs:
            raise SystemExit(f"true_ko {args.true_ko} not in observed deltas")
        star = delta_y_star_from_gallery_ko(obs, genes, args.true_ko)
        gallery_for_star_note = "observed_pseudobulk"
    else:
        star = delta_y_star_from_gallery_ko(gallery_pred, genes, args.true_ko)
        gallery_for_star_note = "predicted_gallery"

    save_delta_y_star(out / "delta_y_star.tsv", star)
    star_vec = align_delta_y_star(star, genes)

    scored = score_gallery(
        gallery_pred,
        star_vec,
        genes,
        metric=args.metric,
        covered=covered,
        primary_only_covered=False,
    )
    scored.to_csv(out / "reverse_scores.tsv", sep="\t", index=False)
    top_k(scored, k=args.top_k, covered_only=False).to_csv(
        out / f"top{args.top_k}.tsv", sep="\t", index=False
    )

    recovery = oracle_recovery_rank(scored, args.true_ko)
    recovery["star_source"] = gallery_for_star_note
    recovery["pred_dir"] = str(args.pred_dir)
    (out / "oracle_recovery.json").write_text(json.dumps(recovery, indent=2))
    print(json.dumps(recovery, indent=2))
    print(f"Wrote {out}")


def cmd_recovery(args: argparse.Namespace) -> None:
    """Multi-KO recovery: observed (and optional predicted) gallery baselines."""
    from reverse.src.recovery import run_multi_ko_recovery, summarize_recovery

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    genes, gallery_pred = _build_gallery(Path(args.pred_dir), Path(args.dataset_h5ad))
    _, gallery_obs = load_observed_deltas(Path(args.pseudobulk_deltas), genes)
    common = sorted(set(gallery_pred) & set(gallery_obs))
    rng = np.random.default_rng(args.seed)
    if args.n_ko <= 0 or args.n_ko >= len(common):
        query_kos = common
    else:
        query_kos = sorted(rng.choice(common, size=args.n_ko, replace=False).tolist())

    # A: obs ΔY* → search predicted gallery (realistic reverse with model)
    df_op = run_multi_ko_recovery(query_kos, gallery_obs, gallery_pred, genes)
    df_op.to_csv(out / "recovery_obs_query_pred_gallery.tsv", sep="\t", index=False)
    sum_op = summarize_recovery(df_op)
    sum_op["setting"] = "obs_star__pred_gallery"

    # B: obs ΔY* → search observed gallery (upper bound / CMap-style lookup)
    df_oo = run_multi_ko_recovery(query_kos, gallery_obs, gallery_obs, genes)
    df_oo.to_csv(out / "recovery_obs_query_obs_gallery.tsv", sep="\t", index=False)
    sum_oo = summarize_recovery(df_oo)
    sum_oo["setting"] = "obs_star__obs_gallery"

    # C: pred ΔY* → search predicted gallery (self-consistency)
    df_pp = run_multi_ko_recovery(query_kos, gallery_pred, gallery_pred, genes)
    df_pp.to_csv(out / "recovery_pred_query_pred_gallery.tsv", sep="\t", index=False)
    sum_pp = summarize_recovery(df_pp)
    sum_pp["setting"] = "pred_star__pred_gallery"

    summary = {
        "n_ko_requested": args.n_ko,
        "n_ko_used": len(query_kos),
        "seed": args.seed,
        "pred_dir": str(args.pred_dir),
        "settings": [sum_op, sum_oo, sum_pp],
    }
    (out / "recovery_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary["settings"]).to_csv(out / "recovery_summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


def cmd_from_config(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(cfg.get("output_dir", args.out_dir))
    ns = argparse.Namespace(
        pred_dir=cfg["gallery"]["pred_dir"],
        dataset_h5ad=cfg["gallery"]["dataset_h5ad"],
        coverage_pca_tsv=cfg["gallery"].get("coverage_pca_tsv"),
        delta_y_star=cfg.get("delta_y_star_tsv"),
        disease_tsv=cfg.get("disease_tsv"),
        control_tsv=cfg.get("control_tsv"),
        delta_mode=cfg.get("delta_mode", "control_minus_disease"),
        metric=cfg.get("scoring", {}).get("metric", "pearson"),
        primary_only_covered=cfg.get("scoring", {}).get("primary_only_covered", True),
        top_k=cfg.get("scoring", {}).get("top_k", 50),
        n_random=cfg.get("benchmarks", {}).get("n_random", 20),
        seed=cfg.get("seed", 0),
        out_dir=str(out),
    )
    cmd_score(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reverse perturbation ranking")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pred_dir", type=str, default=str(DEFAULT_HEPG2_PRED))
        sp.add_argument("--dataset_h5ad", type=str, default=str(DEFAULT_HEPG2_H5AD))
        sp.add_argument("--coverage_pca_tsv", type=str, default=None)
        sp.add_argument("--metric", choices=["pearson", "spearman", "cosine"], default="pearson")
        sp.add_argument("--top_k", type=int, default=50)
        sp.add_argument("--out_dir", type=str, required=True)

    sp = sub.add_parser("score", help="Score gallery vs ΔY* TSV or disease/control profiles")
    add_common(sp)
    sp.add_argument("--delta_y_star", type=str, default=None)
    sp.add_argument("--disease_tsv", type=str, default=None)
    sp.add_argument("--control_tsv", type=str, default=None)
    sp.add_argument(
        "--delta_mode",
        choices=["control_minus_disease", "disease_minus_control"],
        default="control_minus_disease",
    )
    sp.add_argument("--primary_only_covered", action="store_true", default=True)
    sp.add_argument("--n_random", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_score)

    so = sub.add_parser("oracle", help="Smoke test: recover a KO from its own ΔY*")
    add_common(so)
    so.add_argument("--true_ko", type=str, required=True)
    so.add_argument(
        "--star_source",
        choices=["observed", "predicted"],
        default="observed",
        help="Build ΔY* from observed pseudobulk or from predicted gallery vector",
    )
    so.add_argument("--pseudobulk_deltas", type=str, default=str(DEFAULT_HEPG2_DELTA))
    so.set_defaults(func=cmd_oracle)

    sr = sub.add_parser("recovery", help="Multi-KO recovery curves (obs/pred galleries)")
    add_common(sr)
    sr.add_argument("--pseudobulk_deltas", type=str, default=str(DEFAULT_HEPG2_DELTA))
    sr.add_argument("--n_ko", type=int, default=200, help="0 = all overlapping KOs")
    sr.add_argument("--seed", type=int, default=0)
    sr.set_defaults(func=cmd_recovery)

    st = sub.add_parser(
        "train_retrieval",
        help="Train dual-encoder reverse retrieval (needs PyTorch env)",
    )
    st.add_argument("--out_dir", type=str, default=str(_ROOT / "reverse/results/retrieval_hepg2"))
    st.add_argument("--epochs", type=int, default=30)
    st.add_argument("--device", type=str, default="cuda")

    def _run_train(a: argparse.Namespace) -> None:
        from reverse.src.train_reverse_retrieval import build_parser as tp, train as tr

        ns = tp().parse_args(
            ["--out_dir", a.out_dir, "--epochs", str(a.epochs), "--device", a.device]
        )
        tr(ns)

    st.set_defaults(func=_run_train)

    sc = sub.add_parser("from_config", help="Run score from YAML config")
    sc.add_argument("--config", type=str, required=True)
    sc.add_argument("--out_dir", type=str, default="reverse/results/from_config")
    sc.set_defaults(func=cmd_from_config)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
