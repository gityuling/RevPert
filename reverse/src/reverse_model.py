"""Dual-encoder reverse retrieval with PCA-ΔY, gene priors, hard negatives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaEncoder(nn.Module):
    """Encode PCA-compressed (or raw) ΔY."""

    def __init__(self, in_dim: int, emb_dim: int = 128, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(y), dim=-1)


class PertEncoder(nn.Module):
    """Encode [P || gene_prior] → embedding."""

    def __init__(self, p_dim: int, gene_dim: int, emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.gene_dim = gene_dim
        in_dim = p_dim + gene_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, p: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        if self.gene_dim > 0:
            if g is None:
                raise ValueError("gene prior required")
            x = torch.cat([p, g], dim=-1)
        else:
            x = p
        return F.normalize(self.net(x), dim=-1)


class ReverseDualEncoder(nn.Module):
    def __init__(
        self,
        delta_in_dim: int,
        p_dim: int,
        gene_dim: int = 0,
        emb_dim: int = 128,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.f = DeltaEncoder(delta_in_dim, emb_dim=emb_dim, hidden=hidden, dropout=dropout)
        self.h = PertEncoder(p_dim, gene_dim=gene_dim, emb_dim=emb_dim, hidden=hidden)
        self.gene_dim = gene_dim
        self.logit_scale = nn.Parameter(torch.tensor(2.3))

    def encode_delta(self, y: torch.Tensor) -> torch.Tensor:
        return self.f(y)

    def encode_pert(self, p: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        return self.h(p, g)

    def forward(
        self, y: torch.Tensor, p: torch.Tensor, g: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zy = self.encode_delta(y)
        zp = self.encode_pert(p, g)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return zy, zp, scale


def info_nce_loss(zy: torch.Tensor, zp: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logits = scale * (zy @ zp.T)
    labels = torch.arange(zy.shape[0], device=zy.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def hard_negative_info_nce(
    zy: torch.Tensor,
    zp: torch.Tensor,
    scale: torch.Tensor,
    hard_sim: torch.Tensor,
    hard_weight: float = 0.5,
) -> torch.Tensor:
    """InfoNCE with harder negatives (high prototype similarity) up-weighted in logits."""
    b = zy.shape[0]
    eye = torch.eye(b, device=zy.device, dtype=torch.bool)
    h = hard_sim.detach().clamp(min=0.0).masked_fill(eye, 0.0)
    clean = scale * (zy @ zp.T)
    logits = clean + hard_weight * h * scale
    # exact diagonal from clean similarities
    logits = logits.clone()
    logits[eye] = clean[eye]
    labels = torch.arange(b, device=zy.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def augment_delta(y: torch.Tensor, gene_drop: float = 0.3, noise_std: float = 0.05) -> torch.Tensor:
    out = y.clone()
    if gene_drop > 0:
        mask = torch.rand_like(out) > gene_drop
        out = out * mask
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


class PCAProjector:
    """Train-only PCA for ΔY rows (n_samples × n_genes)."""

    def __init__(self, n_components: int = 256):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # (k, n_genes)

    def fit(self, Y: "np.ndarray") -> "PCAProjector":
        import numpy as np
        from sklearn.decomposition import PCA

        X = np.nan_to_num(Y, nan=0.0)
        k = min(self.n_components, X.shape[0] - 1, X.shape[1])
        pca = PCA(n_components=k, random_state=0)
        pca.fit(X)
        self.mean_ = pca.mean_.astype(np.float32)
        self.components_ = pca.components_.astype(np.float32)
        self.n_components = k
        return self

    def transform(self, Y: "np.ndarray") -> "np.ndarray":
        import numpy as np

        X = np.nan_to_num(Y, nan=0.0)
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean_, "components": self.components_, "n_components": self.n_components}

    def load_state_dict(self, d: dict) -> None:
        self.mean_ = d["mean"]
        self.components_ = d["components"]
        self.n_components = int(d["n_components"])


# late import annotation
import numpy as np  # noqa: E402
