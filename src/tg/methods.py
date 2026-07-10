"""Parameter interpolation & extrapolation methods.

Every method takes a list of past checkpoint *parameter vectors* (oldest ->
newest) and returns an estimated future parameter vector. None of them may look
at future data. Hyperparameter (alpha) selection lives in engine.py and only
ever uses past/current validation data (the paper's core methodological rule).

Reference: Madaan, Chopra, Cho. "Temporal Generalization: A Reality Check."
ICLR 2026, arXiv:2509.23487, Section 3.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def recent(vecs: List[torch.Tensor]) -> torch.Tensor:
    """Deploy the most recent model theta_t. The baseline everyone must beat."""
    return vecs[-1].clone()


def average(vecs: List[torch.Tensor]) -> torch.Tensor:
    """Uniform convex combination of all past checkpoints (model averaging)."""
    return torch.stack(vecs, dim=0).mean(dim=0)


def ema(vecs: List[torch.Tensor], decay: float = 0.9) -> torch.Tensor:
    """Exponentially-weighted average, favouring recent checkpoints."""
    out = vecs[0].clone()
    for v in vecs[1:]:
        out = decay * out + (1.0 - decay) * v
    # renormalise emphasis toward the newest while staying in the convex hull
    return decay * out + (1.0 - decay) * vecs[-1]


def downscale(vecs: List[torch.Tensor], alpha: float) -> torch.Tensor:
    """theta_tilde = alpha * theta_t, alpha in [0, 1] (Eq. 2).

    Shrinks the magnitude of the most recent model while preserving direction.
    Motivation: parameter norms grow over training ("overconfidence"); shrinking
    can improve generalisation to an unpredictable future.
    """
    return alpha * vecs[-1]


def taylor_extrapolate(vecs: List[torch.Tensor], alpha: float, dt: float = 1.0) -> torch.Tensor:
    """theta_tilde = theta_t + alpha * (theta_t - theta_{t-dt}) / dt (Eq. 4).

    First-order forward extrapolation along the most recent parameter-change
    direction. Needs at least two checkpoints.
    """
    if len(vecs) < 2:
        return vecs[-1].clone()
    direction = (vecs[-1] - vecs[-2]) / dt
    return vecs[-1] + alpha * direction


# ---------------------------------------------------------------------------
# Extension E1: per-layer, norm-aware downscaling.
# ---------------------------------------------------------------------------
def layer_norm_growth(
    vecs: List[torch.Tensor], slices: Dict[str, Tuple[int, int]]
) -> Dict[str, float]:
    """Relative L2-norm growth of each layer across the checkpoint trajectory.

    growth = (||layer_T|| - ||layer_1||) / (||layer_1|| + eps)

    The paper argues global downscaling helps because norms grow over time. But
    growth is not uniform across layers -- this measures it per layer so we can
    shrink each layer in proportion to how much it grew.
    """
    eps = 1e-8
    growth: Dict[str, float] = {}
    for name, (s, e) in slices.items():
        n0 = vecs[0][s:e].norm().item()
        nT = vecs[-1][s:e].norm().item()
        growth[name] = (nT - n0) / (n0 + eps)
    return growth


def per_layer_downscale(
    vecs: List[torch.Tensor],
    slices: Dict[str, Tuple[int, int]],
    alpha_base: float,
    growth: Dict[str, float],
) -> torch.Tensor:
    """Downscale each layer by its own factor, tuned by a single scalar alpha_base.

    Layers that grew more (relative to the mean growth) get shrunk more:
        alpha_layer = alpha_base ** (growth_layer / mean_growth)
    A single 1-D search over alpha_base keeps past-only tuning cheap, while the
    per-layer exponent lets high-growth layers be shrunk harder. When all layers
    grow equally this reduces to global downscaling by alpha_base.
    """
    out = vecs[-1].clone()
    values = [g for g in growth.values() if g > 0]
    mean_growth = (sum(values) / len(values)) if values else 1.0
    mean_growth = max(mean_growth, 1e-6)
    for name, (s, e) in slices.items():
        g = max(growth.get(name, 0.0), 0.0)
        exponent = g / mean_growth
        alpha_layer = alpha_base ** exponent
        out[s:e] = alpha_layer * out[s:e]
    return out


# ---------------------------------------------------------------------------
# ORIGINAL METHOD: Anisotropic Trajectory Downscaling (ATD).
# ---------------------------------------------------------------------------
def recency_subspace(vecs: List[torch.Tensor], k: int = 2, m: int = 3) -> torch.Tensor:
    """Top-k orthonormal directions the model most recently moved along.

    We stack the last `m` parameter-update deltas (theta_i - theta_{i-1}) and take
    the top-k right singular vectors. This is the low-dimensional subspace in
    which the model has been adapting to the most recent timesteps -- the part
    most likely to be over-fit to 'now'. Returns a (k_eff, D) matrix of
    orthonormal rows (empty tensor if there is not enough history).
    """
    if len(vecs) < 2:
        return torch.empty(0)
    m_eff = min(m, len(vecs) - 1)
    deltas = torch.stack([vecs[-i] - vecs[-i - 1] for i in range(1, m_eff + 1)], dim=0)
    # SVD of an (m_eff x D) matrix is cheap because m_eff is small.
    _, _, Vh = torch.linalg.svd(deltas, full_matrices=False)
    k_eff = min(k, Vh.shape[0])
    return Vh[:k_eff]  # (k_eff, D), orthonormal rows


def anisotropic_trajectory_downscale(
    vecs: List[torch.Tensor], beta: float, k: int = 2, m: int = 3
) -> torch.Tensor:
    """Shrink theta_t ONLY along the recency subspace by factor beta in [0, 1].

        theta_tilde = theta_t - (1 - beta) * P_recency(theta_t)

    where P_recency projects onto the top-k directions of recent parameter
    change. Intuition: the model's recent, present-specific adaptation lives in
    this subspace; damping it (beta < 1) reduces over-confidence about the
    present while preserving the stable bulk of the model in the orthogonal
    complement. Special cases: beta = 1 recovers `recent`; if the subspace were
    the whole space it would recover global `downscale`. Unlike the paper's
    methods this is NOT a convex combination of checkpoints -- it is a directed
    subspace rescaling, so it lies outside their interpolation/extrapolation
    family.
    """
    theta = vecs[-1]
    dirs = recency_subspace(vecs, k=k, m=m)
    if dirs.numel() == 0:
        return theta.clone()
    coeffs = dirs @ theta          # (k_eff,) projection coefficients
    proj = coeffs @ dirs           # (D,) component of theta inside the subspace
    return theta - (1.0 - beta) * proj
