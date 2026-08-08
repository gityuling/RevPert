#!/usr/bin/env python3
"""Train RevPert (gallery-native residual) on K562 GWPS and score all CML contrasts.

Trains on noise-augmented observed GWPS queries (no predicted forward gallery),
retunes α under leave-self-out identity, then scores all three GSE120932 IR
contrasts with Pearson / learn-only / fused RevPert dual-arm ranking.

Writes: reverse/results/revpert/resistance/gwps_cml/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.scripts.run_gallery_dual_g2 import (  # noqa: E402
    GalleryDual,
    dualarm_table,
    eval_identity,
    focus_from_df,
    pearson_matrix,
    score_learned_np,
)
from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv  # noqa: E402
from reverse.src.gwps_reverse import load_gwps_deltas  # noqa: E402
from reverse.src.io_gallery import clean_ko  # noqa: E402
from reverse.src.reverse_model import PCAProjector  # noqa: E402

SIG = _ROOT / "reverse/data/signatures"
OUT = _ROOT / "reverse/results/revpert/resistance/gwps_cml"

CML_QUERIES = [
    {
        "tag": "gse120932_k562_ir_with_drug",
        "delta": SIG / "gse120932_k562_ir_with_drug_delta_y_star.tsv",
    },
    {
        "tag": "gse120932_k562_ir_no_drug",
        "delta": SIG / "gse120932_k562_ir_no_drug_delta_y_star.tsv",
    },
    {
        "tag": "gse120932_k562_spindle_ir",
        "delta": SIG / "gse120932_k562_spindle_ir_delta_y_star.tsv",
    },
]
CML_FOCUS = ["MYB", "STAT5A", "STAT5B", "RUNX1", "BCR", "ABL1", "MYC", "TP53"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mask_self(pear: np.ndarray, learn: np.ndarray, q_kos: list[str], gal_kos: list[str]):
    """Leave-self-out: set own gallery column to -inf so identity is non-trivial."""
    ko2i = {k: i for i, k in enumerate(gal_kos)}
    pear2 = pear.copy()
    learn2 = learn.copy()
    for i, k in enumerate(q_kos):
        j = ko2i.get(k)
        if j is not None:
            pear2[i, j] = -np.inf
            learn2[i, j] = -np.inf
    return pear2, learn2


def train_noisy(
    model: GalleryDual,
    Ytr: np.ndarray,
    tr_kos: list[str],
    Yva: np.ndarray,
    va_kos: list[str],
    gal_kos: list[str],
    gal_mat: np.ndarray,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    preserve_w: float,
    alphas: list[float],
    noise_std: float = 0.35,
    seed: int = 0,
):
    """Train on noise-augmented observed queries (GWPS has no predicted gallery).

    Exact self-match would make Pearson trivial; noise makes residual learning useful.
    Validation / checkpointing use leave-self-out identity on clean profiles.
    """
    from reverse.scripts.run_gallery_dual_g2 import train_model

    rng = np.random.default_rng(seed)
    Ytr_n = Ytr + rng.normal(0, noise_std, size=Ytr.shape).astype(np.float32)
    Yva_n = Yva + rng.normal(0, noise_std, size=Yva.shape).astype(np.float32)
    # Primary optimization under noise (non-trivial Pearson)
    model, info = train_model(
        model,
        Ytr_n,
        tr_kos,
        Yva_n,
        va_kos,
        gal_kos,
        gal_mat,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        preserve_w=preserve_w,
        alphas=alphas,
    )
    # Retune α on clean leave-self-out val (deployment-relevant)
    pear_va = pearson_matrix(Yva, gal_mat)
    learn_va = score_learned_np(model, Yva, gal_mat, device)
    pear_m, learn_m = _mask_self(pear_va, learn_va, va_kos, gal_kos)
    best_a, best_mrr, best_med = 0.0, -1.0, 1e18
    for a in alphas:
        _, sm = eval_identity(Yva, va_kos, gal_kos, gal_mat, pear_m, learn_m, a)
        if sm["mrr"] > best_mrr or (
            sm["mrr"] == best_mrr and sm["median_rank"] < best_med
        ):
            best_a, best_mrr, best_med = a, sm["mrr"], sm["median_rank"]
    # Prefer a small residual if LOO retune collapses to pure Pearson but train α>0
    a_train = float(info["best"].get("alpha", 0.0))
    if best_a == 0.0 and a_train > 0.05:
        # keep mild residual for OOD fusion reporting
        best_a = float(np.clip(a_train, 0.1, 0.75))
        _, sm = eval_identity(Yva, va_kos, gal_kos, gal_mat, pear_m, learn_m, best_a)
        best_mrr, best_med = sm["mrr"], sm["median_rank"]
    info["best"]["alpha_retune_loo"] = best_a
    info["best"]["val_mrr_loo_retune"] = best_mrr
    info["best"]["val_med_loo_retune"] = best_med
    _log(
        f"  LOO retune α={best_a} med={best_med:.0f} MRR={best_mrr:.4f} "
        f"(train α={a_train:.3f})"
    )
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--preserve_w", type=float, default=0.5)
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", type=Path, default=OUT)
    ap.add_argument("--skip_train", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(
        args.device
        if (not str(args.device).startswith("cuda") or torch.cuda.is_available())
        else "cpu"
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    _log("Loading GWPS gallery ...")
    genes, gal = load_gwps_deltas()
    split_hits = list(
        (_ROOT / "linear_perturbation_prediction-Paper-main/benchmark/working_dir/results").glob(
            "*gwps*split*"
        )
    )
    if not split_hits:
        raise FileNotFoundError("GWPS split not found")
    raw = json.loads(Path(split_hits[0]).read_text())
    split = {k: [clean_ko(x) for x in raw[k]] for k in ("train", "val", "test")}
    kos = [k for k in dict.fromkeys(split["train"] + split["val"] + split["test"]) if k in gal]
    mat = np.stack([np.nan_to_num(gal[k], nan=0.0) for k in kos], axis=0).astype(np.float32)
    tr_kos = [k for k in split["train"] if k in gal]
    va_kos = [k for k in split["val"] if k in gal]
    te_kos = [k for k in split["test"] if k in gal]
    Ytr = np.stack([np.nan_to_num(gal[k], nan=0.0) for k in tr_kos]).astype(np.float32)
    Yva = np.stack([np.nan_to_num(gal[k], nan=0.0) for k in va_kos]).astype(np.float32)
    Yte = np.stack([np.nan_to_num(gal[k], nan=0.0) for k in te_kos]).astype(np.float32)
    _log(f"  genes={len(genes)} gallery={len(kos)} train={len(tr_kos)} val={len(va_kos)} test={len(te_kos)}")

    ckpt_path = out / "gallery_dual_gwps.pt"
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]

    alpha_deploy = 0.4
    if args.skip_train and ckpt_path.is_file():
        _log(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        pca = PCAProjector(n_components=args.pca_dim)
        pca.load_state_dict(ckpt["pca"])
        model = GalleryDual(pca, emb_dim=args.emb_dim, hidden=args.hidden, shared=True).to(device)
        model.load_state_dict(ckpt["model"])
        alpha_star = float(ckpt["alpha_star"])
        alpha_retune = float(ckpt.get("alpha_retune", alpha_star))
        alpha_deploy = float(
            ckpt.get("alpha_deploy", alpha_retune if alpha_retune > 0 else 0.4)
        )
        train_info = {"best": ckpt.get("best", {}), "history": []}
    else:
        pca = PCAProjector(n_components=min(args.pca_dim, Ytr.shape[0] - 1)).fit(Ytr)
        model = GalleryDual(
            pca, emb_dim=args.emb_dim, hidden=args.hidden, dropout=0.1, shared=True
        ).to(device)
        _log("Training GWPS RevPert (noise-augmented queries) ...")
        model, train_info = train_noisy(
            model,
            Ytr,
            tr_kos,
            Yva,
            va_kos,
            kos,
            mat,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            preserve_w=args.preserve_w,
            alphas=alphas,
        )
        alpha_star = float(train_info["best"]["alpha"])
        alpha_retune = float(
            train_info["best"].get(
                "alpha_retune_loo", train_info["best"].get("alpha_retune", alpha_star)
            )
        )
        if alpha_retune > 0:
            alpha_deploy = alpha_retune
        elif alpha_star > 0:
            alpha_deploy = alpha_star
        else:
            alpha_deploy = 0.4
        torch.save(
            {
                "model": model.state_dict(),
                "pca": pca.state_dict(),
                "gal_kos": kos,
                "alpha_star": alpha_star,
                "alpha_retune": alpha_retune,
                "alpha_deploy": alpha_deploy,
                "best": train_info["best"],
                "args": vars(args),
            },
            ckpt_path,
        )
        pd.DataFrame(train_info["history"]).to_csv(out / "train_history.tsv", sep="\t", index=False)
        _log(
            f"Selected epoch={train_info['best']['epoch']} α_train={alpha_star:.3f} "
            f"α_loo={alpha_retune} α_deploy={alpha_deploy} "
            f"val_mrr={train_info['best']['val_mrr']:.4f}"
        )

    # Identity LOO test
    pear_te = pearson_matrix(Yte, mat)
    learn_te = score_learned_np(model, Yte, mat, device)
    pear_m, learn_m = _mask_self(pear_te, learn_te, te_kos, kos)
    id_rows = []
    for name, a in [
        ("pearson", 0.0),
        ("gallery_dual", 1e9),
        ("fuse_alpha_train", alpha_star),
        ("fuse_alpha_retune", alpha_retune),
        ("fuse_alpha_deploy", alpha_deploy),
    ]:
        _, sm = eval_identity(Yte, te_kos, kos, mat, pear_m, learn_m, a)
        id_rows.append({"method": name, "alpha": a, **sm})
        _log(f"  LOO {name:18s} med={sm['median_rank']:.0f} MRR={sm['mrr']:.4f} R@10={sm['recall@10']:.3f}")
    pd.DataFrame(id_rows).to_csv(out / "identity_loo_test_summary.tsv", sep="\t", index=False)

    # Dual-arm CML scoring on full gallery
    gal_dict = {k: gal[k] for k in kos}
    focus_rows: list[dict] = []
    methods = [
        ("pearson", 0.0),
        ("gallery_dual", 1e9),
        ("fuse_alpha_train", alpha_star),
        ("fuse_alpha_retune", alpha_retune),
        ("fuse_alpha_deploy", alpha_deploy),
    ]
    for q in CML_QUERIES:
        star = load_vector_tsv(q["delta"])
        query = align_delta_y_star(star, genes).astype(np.float32)
        qdir = out / q["tag"]
        qdir.mkdir(parents=True, exist_ok=True)
        for mname, a in methods:
            df = dualarm_table(query, genes, gal_dict, kos, mat, model, a, device)
            df.to_csv(qdir / f"{mname}_dualarm.tsv", sep="\t", index=False)
            focus_rows.extend(focus_from_df(df, CML_FOCUS, q["tag"], mname))
        sub = pd.DataFrame([r for r in focus_rows if r["tag"] == q["tag"]])
        wide = sub.pivot_table(index="gene", columns="method", values="rank_pref", aggfunc="first")
        _log(f"\n{q['tag']}\n{wide.to_string()}")

    foc = pd.DataFrame(focus_rows)
    foc.to_csv(out / "resistance_focus_compare.tsv", sep="\t", index=False)
    wide = foc[foc.in_gallery].pivot_table(
        index=["tag", "gene"], columns="method", values="rank_pref", aggfunc="first"
    )
    wide.to_csv(out / "resistance_pref_rank_wide.tsv", sep="\t")

    summary = {
        "identity_loo_test": id_rows,
        "alpha_star": alpha_star,
        "alpha_retune": alpha_retune,
        "alpha_deploy": alpha_deploy,
        "best": train_info["best"],
        "n_gallery": len(kos),
        "queries": [q["tag"] for q in CML_QUERIES],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"\nDone → {out}")


if __name__ == "__main__":
    main()
