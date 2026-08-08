#!/usr/bin/env python3
"""CML / GSE120932 signed nulls on K562 GWPS (Pearson dual-arm).

Mirrors HCC proving-ground checks:
  - Arm Top-50 vs |ΔY*| Top-50 Jaccard
  - Top-50 overlap after ΔY* sign-flip
  - Empirical preferred-arm rank p-values under gene-axis permutations of ΔY*

Writes: reverse/results/revpert/resistance/gwps_cml/nulls/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from reverse.src.delta_y_star import align_delta_y_star, load_vector_tsv  # noqa: E402
from reverse.src.gwps_reverse import load_gwps_deltas  # noqa: E402

SIG = _ROOT / "reverse/data/signatures"
OUT = _ROOT / "reverse/results/revpert/resistance/gwps_cml/nulls"

QUERIES = [
    ("gse120932_k562_ir_with_drug", "IR + drug", SIG / "gse120932_k562_ir_with_drug_delta_y_star.tsv"),
    ("gse120932_k562_ir_no_drug", "IR off drug", SIG / "gse120932_k562_ir_no_drug_delta_y_star.tsv"),
    ("gse120932_k562_spindle_ir", "spindle IR", SIG / "gse120932_k562_spindle_ir_delta_y_star.tsv"),
]
ANCHORS = ["MYB", "STAT5A", "STAT5B", "RUNX1", "BCR", "ABL1"]
TOP_K = 50
N_PERM = 500
SEED = 0


def _gallery_z_on_mask(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-zscore gallery on finite query genes; return (n_gal, n_finite)."""
    G = np.nan_to_num(mat[:, mask], nan=0.0)
    gmu = G.mean(axis=1, keepdims=True)
    gsd = G.std(axis=1, keepdims=True)
    gsd = np.where(gsd < 1e-12, 1.0, gsd)
    return (G - gmu) / gsd


def _query_z(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    qv = q[mask].astype(np.float64)
    qv = qv - qv.mean()
    qsd = qv.std()
    if qsd < 1e-12:
        return np.zeros_like(qv)
    return qv / qsd


def _pearson_scores_fast(G_z: np.ndarray, q_z: np.ndarray) -> np.ndarray:
    """Pearson via (n_gal, n_finite) @ (n_finite,); both already z-scored."""
    n = max(q_z.shape[0], 1)
    return (G_z @ q_z) / n


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    """1-based ranks; higher score → better (rank 1)."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _top_set(ranks: np.ndarray, kos: list[str], k: int = TOP_K) -> set[str]:
    idx = np.argsort(ranks)[:k]
    return {kos[i] for i in idx}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading GWPS gallery ...", flush=True)
    genes, gal = load_gwps_deltas()
    kos = sorted(gal)
    mat = np.stack([gal[k] for k in kos], axis=0).astype(np.float64)
    # Precompute nothing heavy; per-query mask differs slightly but gene axis shared
    print(f"  genes={len(genes)} gallery={len(kos)}", flush=True)

    summary_rows: list[dict] = []
    anchor_rows: list[dict] = []
    rng = np.random.default_rng(SEED)

    for tag, label, path in QUERIES:
        print(f"Scoring {tag} ...", flush=True)
        star_raw = load_vector_tsv(path)
        star = align_delta_y_star(star_raw, genes).astype(np.float64)
        mask = np.isfinite(star)
        if int(mask.sum()) < 10:
            raise RuntimeError(f"{tag}: too few finite genes")
        G_z = _gallery_z_on_mask(mat, mask)
        q_z = _query_z(star, mask)
        scores = _pearson_scores_fast(G_z, q_z)
        rank_a = _ranks_desc(scores)
        rank_b = _ranks_desc(-scores)
        pref = np.minimum(rank_a, rank_b)
        pref_arm = np.where(rank_a <= rank_b, "A", "B")

        # DEG magnitude Top-50 on overlapping finite genes
        abs_star = np.abs(star)
        abs_star_f = np.where(mask, abs_star, -np.inf)
        deg_idx = np.argsort(-abs_star_f)[:TOP_K]
        deg_set = {genes[i] for i in deg_idx if mask[i]}

        top_a = _top_set(rank_a, kos)
        top_b = _top_set(rank_b, kos)
        jac_a = _jaccard(top_a, deg_set)
        jac_b = _jaccard(top_b, deg_set)

        # Sign-flip null (exact negation of query on same mask)
        scores_flip = _pearson_scores_fast(G_z, -q_z)
        rank_a_f = _ranks_desc(scores_flip)
        rank_b_f = _ranks_desc(-scores_flip)
        top_a_f = _top_set(rank_a_f, kos)
        top_b_f = _top_set(rank_b_f, kos)
        flip_ov_a = len(top_a & top_a_f)
        flip_ov_b = len(top_b & top_b_f)
        cross_a_bf = len(top_a & top_b_f)

        summary_rows.append(
            {
                "tag": tag,
                "label": label,
                "n_gallery": len(kos),
                "jaccard_armA_vs_deg50": jac_a,
                "jaccard_armB_vs_deg50": jac_b,
                "overlap_armA_vs_deg50": len(top_a & deg_set),
                "overlap_armB_vs_deg50": len(top_b & deg_set),
                "signflip_armA_top50_overlap": flip_ov_a,
                "signflip_armB_top50_overlap": flip_ov_b,
                "signflip_armA_vs_flipped_armB_overlap": cross_a_bf,
            }
        )

        # Observed preferred ranks for anchors
        ko2i = {k: i for i, k in enumerate(kos)}
        obs = {}
        for g in ANCHORS:
            if g not in ko2i:
                continue
            i = ko2i[g]
            obs[g] = {
                "rank_armA": int(rank_a[i]),
                "rank_armB": int(rank_b[i]),
                "rank_pref": int(pref[i]),
                "pref_arm": str(pref_arm[i]),
                "score": float(scores[i]),
            }

        # Permutation null on preferred-arm rank (permute finite query values)
        base = star[mask].copy()
        null_pref = {g: np.empty(N_PERM, dtype=np.int32) for g in obs}
        for r in range(N_PERM):
            fake_vals = rng.permutation(base)
            fake_z = fake_vals - fake_vals.mean()
            fsd = fake_z.std()
            fake_z = np.zeros_like(fake_vals) if fsd < 1e-12 else fake_z / fsd
            s = _pearson_scores_fast(G_z, fake_z)
            ra = _ranks_desc(s)
            rb = _ranks_desc(-s)
            pr = np.minimum(ra, rb)
            for g in obs:
                null_pref[g][r] = int(pr[ko2i[g]])

        # Long-form samples for SI figure
        sample_rows = []
        for g, info in obs.items():
            arr = null_pref[g]
            for rep, v in enumerate(arr):
                sample_rows.append(
                    {"tag": tag, "label": label, "anchor": g, "rep": int(rep), "rank_pref_null": int(v)}
                )
            p = float(np.mean(arr <= info["rank_pref"]))
            p_geomean = float(np.exp(np.mean(np.log(arr.astype(float)))))
            row = {
                "tag": tag,
                "label": label,
                "anchor": g,
                **info,
                "n_perm": N_PERM,
                "null_median_pref": float(np.median(arr)),
                "null_mean_log_pref": p_geomean,
                "emp_p_pref": p,
            }
            anchor_rows.append(row)
            print(
                f"  {g}: pref={info['rank_pref']} ({info['pref_arm']}) "
                f"null_med={np.median(arr):.0f} emp_p={p:.4f}",
                flush=True,
            )
        pd.DataFrame(sample_rows).to_csv(OUT / f"null_pref_samples__{tag}.tsv", sep="\t", index=False)

        print(
            f"  Jaccard A/B vs DEG: {jac_a:.4f}/{jac_b:.4f}; "
            f"sign-flip Top50 ov A/B: {flip_ov_a}/{flip_ov_b}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    anchors = pd.DataFrame(anchor_rows)
    summary.to_csv(OUT / "cml_top50_nulls.tsv", sep="\t", index=False)
    anchors.to_csv(OUT / "cml_anchor_perm_nulls.tsv", sep="\t", index=False)

    # Compact SI-ready rows (like tab:task2_null)
    si_rows = []
    for _, r in summary.iterrows():
        si_rows.append(
            {
                "signature": f"GSE120932 {r['label']}",
                "check": "ArmA Top-50 vs |ΔY*| Top-50 Jaccard",
                "value": float(r["jaccard_armA_vs_deg50"]),
                "detail": f"overlap={int(r['overlap_armA_vs_deg50'])}/50",
            }
        )
        si_rows.append(
            {
                "signature": f"GSE120932 {r['label']}",
                "check": "ArmB Top-50 vs |ΔY*| Top-50 Jaccard",
                "value": float(r["jaccard_armB_vs_deg50"]),
                "detail": f"overlap={int(r['overlap_armB_vs_deg50'])}/50",
            }
        )
        si_rows.append(
            {
                "signature": f"GSE120932 {r['label']}",
                "check": "Top-50 ArmA overlap after ΔY* sign-flip",
                "value": float(r["signflip_armA_top50_overlap"]),
                "detail": f"overlap={int(r['signflip_armA_top50_overlap'])}/50",
            }
        )
    # Key anchors on on-drug contrast
    on = anchors[anchors["tag"] == "gse120932_k562_ir_with_drug"]
    for _, r in on.iterrows():
        if r["anchor"] in {"MYB", "STAT5A", "RUNX1", "STAT5B", "BCR"}:
            si_rows.append(
                {
                    "signature": "GSE120932 IR + drug",
                    "check": f"{r['anchor']} preferred-arm rank / emp. p (n={N_PERM})",
                    "value": float(r["rank_pref"]),
                    "detail": f"arm {r['pref_arm']}; emp_p={r['emp_p_pref']:.4f}; null_med={r['null_median_pref']:.0f}",
                }
            )
    si = pd.DataFrame(si_rows)
    si.to_csv(OUT / "table_cml_nulls_si.tsv", sep="\t", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "n_perm": N_PERM,
                "seed": SEED,
                "top_k": TOP_K,
                "n_gallery": len(kos),
                "queries": [t for t, _, _ in QUERIES],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
