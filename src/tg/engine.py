"""Training + evaluation engine.

Implements:
  * sequential fine-tuning (continual learning): model at time t is initialised
    from theta_{t-1} (paper Eq. 5), producing a checkpoint trajectory;
  * delta-forward-transfer evaluation of every method;
  * PAST-ONLY hyperparameter (alpha) tuning that simulates a one-step-back
    deployment (paper Eq. 6) -- it is structurally impossible for it to see
    future data, which is the whole methodological point of the paper.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import methods as M
from .data import loader
from .utils import Checkpoint, layer_slices, set_seed


# --------------------------------------------------------------------------- #
# Basic train / eval primitives
# --------------------------------------------------------------------------- #
def train_one_step(model, ds, device, epochs=3, lr=1e-3, batch_size=128) -> None:
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = loader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate_acc(model, ds, device, batch_size=256) -> float:
    model.to(device).eval()
    correct = total = 0
    for xb, yb in loader(ds, batch_size=batch_size):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_loss(model, ds, device, batch_size=256) -> float:
    """Mean cross-entropy loss. Used by the basin-barrier diagnostic."""
    model.to(device).eval()
    total_loss = total = 0.0
    for xb, yb in loader(ds, batch_size=batch_size):
        xb, yb = xb.to(device), yb.to(device)
        loss = F.cross_entropy(model(xb), yb, reduction="sum")
        total_loss += loss.item()
        total += yb.numel()
    return total_loss / max(total, 1)


# --------------------------------------------------------------------------- #
# Stage 1: build the checkpoint trajectory via sequential fine-tuning
# --------------------------------------------------------------------------- #
def build_trajectory(model_fn: Callable, stream, device, train_cfg: dict):
    """Return (checkpoints, param_vectors, slices).

    Continual learning: keep one model, re-initialise each step from the
    previous solution, fine-tune on the current timestep only.
    """
    model = model_fn()
    slices = layer_slices(model)
    checkpoints: List[Checkpoint] = []
    vectors: List[torch.Tensor] = []
    for step in stream:
        train_one_step(
            model,
            step["train"],
            device,
            epochs=train_cfg.get("epochs", 3),
            lr=train_cfg.get("lr", 1e-3),
            batch_size=train_cfg.get("batch_size", 128),
        )
        ckpt = Checkpoint(model)
        checkpoints.append(ckpt)
        vectors.append(ckpt.vec.clone())
    return checkpoints, vectors, slices


# --------------------------------------------------------------------------- #
# Method registry -- each maps past vectors (+ optional alpha) -> future vector
# --------------------------------------------------------------------------- #
ALPHA_GRIDS = {
    "downscale": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    "taylor": [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
    "per_layer_downscale": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    "aniso_downscale": [0.0, 0.3, 0.5, 0.7, 0.85, 1.0],  # ATD: shrink recency subspace
}
ATD_K = 2  # number of recency-subspace directions
ATD_M = 3  # number of recent deltas used to estimate the subspace


def build_estimate(name: str, vecs: List[torch.Tensor], slices, alpha: float | None):
    if name == "recent":
        return M.recent(vecs)
    if name == "average":
        return M.average(vecs)
    if name == "ema":
        return M.ema(vecs, decay=0.9)
    if name == "downscale":
        return M.downscale(vecs, alpha)
    if name == "taylor":
        return M.taylor_extrapolate(vecs, alpha)
    if name == "per_layer_downscale":
        growth = M.layer_norm_growth(vecs, slices)
        return M.per_layer_downscale(vecs, slices, alpha, growth)
    if name == "aniso_downscale":
        return M.anisotropic_trajectory_downscale(vecs, alpha, k=ATD_K, m=ATD_M)
    raise ValueError(f"unknown method {name}")


def tune_alpha_past_only(
    name, vecs_up_to_t, slices, model_fn, checkpoints, stream, t, delta, device
) -> float:
    """Select alpha using ONLY data available at time t (paper Eq. 6).

    We emulate the decision we would have made one horizon ago: build an
    estimate from checkpoints up to (t-delta) and score it on D_t's validation
    split -- data that was 'future' from t-delta's perspective but is 'current'
    now. The grid value with the best current-val accuracy is chosen and then
    applied to checkpoints up to t to predict t+delta. No future data touched.
    """
    grid = ALPHA_GRIDS.get(name)
    if grid is None:
        return None  # method has no alpha
    if t - delta < 0:
        return grid[-1]  # not enough history to simulate; default to alpha=1-ish

    past_vecs = vecs_up_to_t[: (t - delta) + 1]
    val_ds = stream[t]["val"]
    model = model_fn()
    best_alpha, best_acc = grid[-1], -1.0
    for a in grid:
        est = build_estimate(name, past_vecs, slices, a)
        checkpoints[t - delta].apply_to(model, est)
        acc = evaluate_acc(model, val_ds, device)
        if acc > best_acc:
            best_acc, best_alpha = acc, a
    return best_alpha


# --------------------------------------------------------------------------- #
# Stage 2: delta-forward-transfer evaluation for all methods
# --------------------------------------------------------------------------- #
def forward_transfer(
    model_fn, stream, checkpoints, vectors, slices, method_names, device, deltas
):
    """Return rows: {method, t, delta, alpha, acc}.

    For each evaluation time t and horizon delta, estimate theta_tilde_{t+delta}
    from checkpoints up to t and evaluate on the (held-out) test/val data of the
    genuinely future timestep t+delta.
    """
    T = len(stream)
    rows = []
    model = model_fn()
    for name in method_names:
        for t in range(T):
            vecs_up_to_t = vectors[: t + 1]
            for delta in deltas:
                if t + delta >= T:
                    continue
                alpha = tune_alpha_past_only(
                    name, vecs_up_to_t, slices, model_fn, checkpoints,
                    stream, t, delta, device,
                )
                est = build_estimate(name, vecs_up_to_t, slices, alpha)
                # buffers come from the most recent real checkpoint (theta_t)
                checkpoints[t].apply_to(model, est)
                acc = evaluate_acc(model, stream[t + delta]["val"], device)
                rows.append(
                    {"method": name, "t": t, "delta": delta,
                     "alpha": (None if alpha is None else float(alpha)),
                     "acc": acc}
                )
    return rows


def run_experiment(model_fn, stream, device, train_cfg, method_names, deltas, seed=0):
    set_seed(seed)
    checkpoints, vectors, slices = build_trajectory(model_fn, stream, device, train_cfg)
    rows = forward_transfer(
        model_fn, stream, checkpoints, vectors, slices, method_names, device, deltas
    )
    return rows, vectors, slices, checkpoints
