"""Post-hoc analysis of the parameter trajectory.

  * per-layer L2-norm growth (why downscaling might work, and the substrate for
    the per-layer extension);
  * global norm vs time;
  * PCA of the parameter trajectory (are consecutive checkpoints close, i.e. did
    continual learning produce a smooth path?);
  * basin-barrier diagnostic -- the loss along the straight line between two
    consecutive checkpoints. A high barrier means the two solutions sit in
    disconnected loss basins, so interpolating/extrapolating between them must
    cross high-loss regions. This is the mechanism the paper *asserts* explains
    why interpolation fails; here we *measure* it, and it lets us test whether
    the barrier differs between modalities (e.g. text vs vision) -- which would
    explain why downscaling helps one and not the other.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from . import methods as M
from .utils import Checkpoint, set_param_vector


def global_norm_trajectory(vectors: List[torch.Tensor]) -> List[float]:
    return [v.norm().item() for v in vectors]


def layer_norm_trajectories(
    vectors: List[torch.Tensor], slices: Dict[str, Tuple[int, int]]
) -> Dict[str, List[float]]:
    traj: Dict[str, List[float]] = {name: [] for name in slices}
    for v in vectors:
        for name, (s, e) in slices.items():
            traj[name].append(v[s:e].norm().item())
    return traj


def layer_growth_summary(vectors, slices) -> Dict[str, float]:
    return M.layer_norm_growth(vectors, slices)


def loss_along_path(
    model_fn: Callable,
    ckpt_a: Checkpoint,
    ckpt_b: Checkpoint,
    ds,
    device,
    eval_loss_fn: Callable,
    n_points: int = 11,
):
    """Cross-entropy loss along the linear path (1-s)*A + s*B in weight space.

    Both parameters AND buffers (e.g. BatchNorm stats) are interpolated, so this
    is a genuine straight line through weight space. Returns (s_grid, losses,
    barrier), where the barrier is the maximum height of the loss curve above the
    straight line connecting the two endpoint losses (0 = perfectly connected;
    large = disconnected basins).
    """
    ss = np.linspace(0.0, 1.0, n_points)
    losses: List[float] = []
    model = model_fn()
    for s in ss:
        vec = (1 - s) * ckpt_a.vec + s * ckpt_b.vec
        set_param_vector(model, vec)
        with torch.no_grad():
            for name, buf in model.named_buffers():
                if name in ckpt_a.buffers and name in ckpt_b.buffers:
                    interp = (1 - s) * ckpt_a.buffers[name] + s * ckpt_b.buffers[name]
                    buf.copy_(interp.to(buf.device, buf.dtype))
        losses.append(eval_loss_fn(model, ds, device))
    losses_arr = np.array(losses)
    endpoint_line = (1 - ss) * losses_arr[0] + ss * losses_arr[-1]
    barrier = float(np.max(losses_arr - endpoint_line))
    return ss, losses, barrier


def trajectory_barriers(
    model_fn: Callable, checkpoints: List[Checkpoint], stream, device, eval_loss_fn,
    n_points: int = 11,
):
    """Barrier between each consecutive pair (theta_t, theta_{t+1}), evaluated on
    the later timestep's validation data. Returns (rows, example_path).

    rows: [{t, barrier, loss_start, loss_end}], example_path: (s, losses) for the
    first pair, for a representative loss-along-path figure.
    """
    rows = []
    example = None
    for t in range(len(checkpoints) - 1):
        ds = stream[t + 1]["val"]
        ss, losses, barrier = loss_along_path(
            model_fn, checkpoints[t], checkpoints[t + 1], ds, device,
            eval_loss_fn, n_points,
        )
        rows.append({"t": t, "barrier": barrier,
                     "loss_start": losses[0], "loss_end": losses[-1]})
        if example is None:
            example = (ss, losses)
    return rows, example


def pca_trajectory(vectors: List[torch.Tensor], k: int = 2) -> np.ndarray:
    """Project the checkpoint vectors to k dims via PCA (numpy SVD)."""
    X = torch.stack(vectors, dim=0).numpy()
    X = X - X.mean(axis=0, keepdims=True)
    # economy SVD; rows of Vt are principal directions
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (U[:, :k] * S[:k])


def alpha_norm_correlation(rows, vectors, method: str = "downscale"):
    """Test the paper's overconfidence hypothesis quantitatively.

    The paper argues downscaling helps because parameter norm grows over time.
    If true, the norm at time t should predict how aggressively we downscale
    (a smaller optimal alpha when the norm is larger). We pair each tuned alpha
    with the parameter norm at its timestep and return the (norm, alpha) points
    plus the Pearson correlation. A negative correlation supports the hypothesis.
    """
    import pandas as pd

    norms = [float(v.norm()) for v in vectors]
    df = pd.DataFrame([r for r in rows if r["method"] == method and r["alpha"] is not None])
    if df.empty:
        return [], [], float("nan")
    xs = [norms[int(t)] for t in df["t"]]
    ys = list(df["alpha"])
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return xs, ys, float("nan")
    r = float(np.corrcoef(xs, ys)[0, 1])
    return xs, ys, r
