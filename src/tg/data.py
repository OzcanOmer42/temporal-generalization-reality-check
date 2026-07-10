"""Temporal datasets.

Two sources:

1. `make_synthetic_stream` -- a self-contained stream of 32x32x1 image
   classification tasks whose decision boundary drifts smoothly with time.
   Requires no download, so the smoke test and CI can run anywhere. The drift
   is real, so the temporal-generalization methods genuinely differ on it.

2. `load_yearbook_stream` -- the real Wilds-Time Yearbook benchmark (portraits
   by year, binary gender prediction). Used on your Mac for the actual
   reproduction. Falls back with a clear message if the `wilds` package or data
   is unavailable.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def _make_timestep(
    t: int, T: int, n: int, img: int, num_classes: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate one timestep. A class-dependent Gaussian blob whose centre
    rotates around the image as t advances -> smooth temporal distribution shift.
    """
    rng = np.random.default_rng(seed + t)
    X = rng.normal(0.0, 0.15, size=(n, 1, img, img)).astype(np.float32)
    y = rng.integers(0, num_classes, size=(n,)).astype(np.int64)

    yy, xx = np.mgrid[0:img, 0:img].astype(np.float32)
    phase = 2.0 * math.pi * t / T  # drift over time
    for c in range(num_classes):
        # each class gets a blob; blob centre drifts with time + class offset
        angle = phase + 2.0 * math.pi * c / num_classes
        cx = img / 2 + (img / 3) * math.cos(angle)
        cy = img / 2 + (img / 3) * math.sin(angle)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (img / 6) ** 2))
        idx = y == c
        X[idx, 0] += blob[None] * 1.0
    return torch.from_numpy(X), torch.from_numpy(y)


def make_synthetic_stream(
    T: int = 8,
    n_per_step: int = 512,
    img: int = 32,
    num_classes: int = 2,
    val_frac: float = 0.3,
    seed: int = 0,
) -> List[dict]:
    """Return a list of length T; each entry has train/val TensorDatasets."""
    stream = []
    for t in range(T):
        X, y = _make_timestep(t, T, n_per_step, img, num_classes, seed)
        n_val = int(len(X) * val_frac)
        perm = torch.randperm(len(X))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        stream.append(
            {
                "t": t,
                "train": TensorDataset(X[tr_idx], y[tr_idx]),
                "val": TensorDataset(X[val_idx], y[val_idx]),
            }
        )
    return stream


def load_yearbook_stream(
    root: str = "~/wild-time-data", val_frac: float = 0.2, seed: int = 0,
    max_steps: int | None = None,
) -> List[dict]:
    """Load the Wild-Time **Yearbook** benchmark as a per-year stream.

    Yearbook is part of the *Wild-Time* benchmark, served by the lightweight
    `wild-time-data` package (NOT the `wilds` package). Install with
    `pip install wild-time-data`; data downloads on first use (~37 MB).

    Each timestep = one year. We split each year's train partition into
    train/val (val is used only for past-only alpha tuning + forward-transfer
    evaluation, matching the synthetic stream). `max_steps` truncates to the
    first N years for a quick first run (Yearbook spans many years).
    """
    try:
        from wild_time_data import load_dataset, available_time_steps  # type: ignore
    except Exception as exc:  # pragma: no cover - only on user machine
        raise RuntimeError(
            "The 'wild-time-data' package is required for the Yearbook track. "
            "Install it with `pip install wild-time-data`, then re-run. "
            "For a no-download demo use --dataset synthetic."
        ) from exc

    import os
    root = os.path.expanduser(root)
    years = available_time_steps("yearbook")
    if max_steps is not None:
        years = years[:max_steps]

    stream = []
    for yi, year in enumerate(years):
        full = load_dataset(
            dataset_name="yearbook", time_step=year, split="train",
            data_dir=root, in_memory=True,
        )
        n = len(full)
        n_val = max(1, int(n * val_frac))
        n_tr = max(1, n - n_val)
        # if rounding overshoots, clamp
        n_val = n - n_tr
        gen = torch.Generator().manual_seed(seed * 1000 + yi)
        tr, val = random_split(full, [n_tr, n_val], generator=gen)
        stream.append({"t": yi, "year": int(year), "train": tr, "val": val})
    return stream


def loader(ds, batch_size: int = 128, shuffle: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
