"""Figure generation. All figures regenerate from the results CSV + trajectory."""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_forward_transfer(df: pd.DataFrame, out_path: str) -> None:
    """Mean accuracy vs delta, one line per method. Error bars = std across seeds
    (and timesteps) when multiple seeds are present."""
    grp = df.groupby(["method", "delta"])["acc"]
    agg = grp.agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(7, 5))
    for method, g in agg.groupby("method"):
        g = g.sort_values("delta")
        yerr = g["std"].fillna(0.0)
        plt.errorbar(g["delta"], g["mean"], yerr=yerr, marker="o",
                     capsize=3, label=method)
    plt.xlabel("forward horizon delta (timesteps into the future)")
    plt.ylabel("mean accuracy on future data")
    plt.title("delta-forward transfer by method (error bars: std)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_alpha_norm(xs, ys, r, out_path: str) -> None:
    """Scatter of optimal downscaling alpha vs parameter norm at that timestep."""
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys)
    plt.xlabel("parameter L2 norm at timestep t")
    plt.ylabel("optimal downscaling alpha (tuned past-only)")
    title = "Does higher norm -> stronger downscaling?"
    if r == r:  # not NaN
        title += f"   (Pearson r = {r:.2f})"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_norm_trajectory(norms: List[float], out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(norms) + 1), norms, marker="o")
    plt.xlabel("timestep t")
    plt.ylabel("global parameter L2 norm")
    plt.title("Parameter norm grows over time (motivates downscaling)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_layer_growth(growth: Dict[str, float], out_path: str) -> None:
    items = sorted(growth.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(8, max(3, 0.4 * len(names))))
    plt.barh(names, vals)
    plt.gca().invert_yaxis()
    plt.xlabel("relative L2-norm growth over the trajectory")
    plt.title("Norm growth is NOT uniform across layers (extension motivation)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loss_path(example, out_path: str) -> None:
    """Loss along the straight line between two consecutive checkpoints."""
    ss, losses = example
    plt.figure(figsize=(7, 4))
    plt.plot(ss, losses, "-o", label="loss along path")
    plt.plot([0, 1], [losses[0], losses[-1]], "--", color="gray",
             label="linear endpoint reference")
    plt.xlabel("interpolation s:  theta_t  ->  theta_{t+1}")
    plt.ylabel("cross-entropy loss")
    plt.title("Basin barrier between consecutive checkpoints")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_barriers(rows, out_path: str) -> None:
    """Barrier height for each consecutive checkpoint pair over time."""
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    plt.bar(df["t"], df["barrier"])
    plt.xlabel("consecutive pair (theta_t -> theta_{t+1})")
    plt.ylabel("loss barrier height")
    plt.title("Are consecutive checkpoints in connected basins?")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pca(points, out_path: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(points[:, 0], points[:, 1], "-o")
    for i, (x, y) in enumerate(points):
        plt.annotate(str(i + 1), (x, y))
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Parameter trajectory (PCA) -- is the path smooth?")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
