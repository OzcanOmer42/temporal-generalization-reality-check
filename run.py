#!/usr/bin/env python3
"""Entry point: reproduce Temporal Generalization (scaled) + original method (ATD).

Methods compared: recent / average / ema / downscale / taylor /
per_layer_downscale / aniso_downscale (ATD, the original contribution).

Examples
--------
# fast synthetic sanity check (<1 min, runs anywhere, no download):
python run.py --smoke-test

# full synthetic run with 3 seeds + error bars:
python run.py --dataset synthetic --T 8 --epochs 3 --seeds 0 1 2

# real Yearbook reproduction on your Mac (downloads Wilds-Time on first run):
python run.py --dataset yearbook --epochs 5 --device mps --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import pandas as pd

from src.tg.analysis import (
    alpha_norm_correlation,
    global_norm_trajectory,
    layer_growth_summary,
    pca_trajectory,
    trajectory_barriers,
)
from src.tg.data import load_yearbook_stream, make_synthetic_stream
from src.tg.engine import evaluate_loss, run_experiment
from src.tg.model import build_model
from src.tg.plots import (
    plot_alpha_norm,
    plot_barriers,
    plot_forward_transfer,
    plot_layer_growth,
    plot_loss_path,
    plot_norm_trajectory,
    plot_pca,
)
from src.tg.utils import get_device

METHODS = [
    "recent", "average", "ema", "downscale", "taylor",
    "per_layer_downscale", "aniso_downscale",
]


def build_stream(args, seed):
    if args.dataset == "synthetic":
        stream = make_synthetic_stream(T=args.T, n_per_step=args.n_per_step, seed=seed)
        cfg = {"in_channels": 1, "num_classes": 2,
               "width": 16 if args.smoke_test else 32}
    else:
        stream = load_yearbook_stream(seed=seed, max_steps=args.max_steps)
        cfg = {"in_channels": 1, "num_classes": 2, "width": 32}
    return stream, cfg


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "yearbook"], default="synthetic")
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny run to verify the pipeline")
    ap.add_argument("--T", type=int, default=8, help="number of timesteps (synthetic)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Yearbook: use only the first N years (quick first run)")
    ap.add_argument("--n-per-step", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="one run per seed; results averaged with std error bars")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.smoke_test:
        args.dataset = "synthetic"
        args.T = 4
        args.n_per_step = 128
        args.epochs = 1
        args.seeds = [0]

    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    print(f"[tg] device={device} dataset={args.dataset} "
          f"epochs={args.epochs} seeds={args.seeds}")

    t0 = time.time()
    train_cfg = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size}

    all_rows = []
    ref_vectors = ref_slices = ref_ckpts = ref_stream = ref_model_fn = None
    ref_rows = None
    for si, seed in enumerate(args.seeds):
        stream, model_cfg = build_stream(args, seed)
        model_fn = (lambda c=model_cfg: build_model(c))
        deltas = [1, 2, 3] if not args.smoke_test else [1]
        deltas = [d for d in deltas if d < len(stream)]
        rows, vectors, slices, checkpoints = run_experiment(
            model_fn, stream, device, train_cfg, METHODS, deltas, seed=seed
        )
        for r in rows:
            r["seed"] = seed
        all_rows.extend(rows)
        if si == 0:  # keep first-seed trajectory for the single-run diagnostics
            ref_vectors, ref_slices, ref_ckpts = vectors, slices, checkpoints
            ref_stream, ref_model_fn, ref_rows = stream, model_fn, rows
            print(f"  timesteps in stream: {len(stream)}")
        print(f"  seed {seed} done ({si + 1}/{len(args.seeds)})")

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.out, "forward_transfer.csv")
    df.to_csv(csv_path, index=False)

    # --- summary: mean +/- std future accuracy per method (across seeds, t, delta)
    g = df.groupby("method")["acc"]
    summary = g.agg(["mean", "std"]).sort_values("mean", ascending=False).round(4)
    summary.to_csv(os.path.join(args.out, "summary_by_method.csv"))
    recent_acc = float(summary.loc["recent", "mean"])

    # --- trajectory analysis (first seed) + figures ---
    norms = global_norm_trajectory(ref_vectors)
    growth = layer_growth_summary(ref_vectors, ref_slices)
    with open(os.path.join(args.out, "layer_growth.json"), "w") as f:
        json.dump(growth, f, indent=2)

    plot_forward_transfer(df, os.path.join(args.out, "fig_forward_transfer.png"))
    plot_norm_trajectory(norms, os.path.join(args.out, "fig_norm_trajectory.png"))
    plot_layer_growth(growth, os.path.join(args.out, "fig_layer_growth.png"))
    if len(ref_vectors) >= 2:
        plot_pca(pca_trajectory(ref_vectors), os.path.join(args.out, "fig_pca.png"))

    # --- alpha-vs-norm correlation (tests the overconfidence hypothesis) ---
    xs, ys, r_corr = alpha_norm_correlation(ref_rows, ref_vectors, method="downscale")
    if xs:
        plot_alpha_norm(xs, ys, r_corr, os.path.join(args.out, "fig_alpha_norm.png"))

    # --- basin-barrier diagnostic ---
    barrier_rows, example = trajectory_barriers(
        ref_model_fn, ref_ckpts, ref_stream, device, evaluate_loss
    )
    bdf = pd.DataFrame(barrier_rows)
    bdf.to_csv(os.path.join(args.out, "basin_barriers.csv"), index=False)
    if example is not None:
        plot_loss_path(example, os.path.join(args.out, "fig_loss_path.png"))
        plot_barriers(barrier_rows, os.path.join(args.out, "fig_barriers.png"))
    mean_barrier = float(bdf["barrier"].mean()) if len(bdf) else float("nan")

    # --- run manifest (reproducibility) ---
    manifest = {
        "dataset": args.dataset, "T": args.T, "epochs": args.epochs,
        "seeds": args.seeds, "device": str(device), "git": git_hash(),
        "python": platform.python_version(),
        "wall_clock_sec": round(time.time() - t0, 1),
        "methods": METHODS,
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
    except Exception:
        pass
    with open(os.path.join(args.out, "run_meta.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # --- console report ---
    print("\n=== Mean future-accuracy by method (mean +/- std, higher is better) ===")
    for name, row in summary.iterrows():
        m, s = row["mean"], (0.0 if pd.isna(row["std"]) else row["std"])
        flag = ""
        if name != "recent":
            flag = "  <-- beats Recent" if m > recent_acc + 1e-4 else "  (<= Recent)"
        star = " *ATD (original)" if name == "aniso_downscale" else ""
        print(f"  {name:22s} {m:.4f} +/- {s:.4f}{flag}{star}")
    winners = [n for n in summary.index
               if n != "recent" and summary.loc[n, "mean"] > recent_acc + 1e-4]
    print(f"\nCentral claim: methods reliably beating Recent: "
          f"{winners if winners else 'NONE (matches the paper)'}")

    print("\n=== Diagnostics ===")
    print(f"  optimal-alpha vs norm correlation (downscale): r = {r_corr:.3f}  "
          f"(negative supports the overconfidence hypothesis)")
    print(f"  mean basin barrier (consecutive checkpoints):  {mean_barrier:.4f}  "
          f"(low = connected basins)")
    print(f"\nWrote results + figures + run_meta.json to {args.out}/  "
          f"({manifest['wall_clock_sec']}s)")


if __name__ == "__main__":
    main()
