# Temporal Generalization: A Reality Check — scaled reproduction + an original method (ATD)

A small, honest reproduction of **Madaan, Chopra, Cho — *Temporal Generalization: A Reality Check* (ICLR 2026, [arXiv:2509.23487](https://arxiv.org/abs/2509.23487); official code [divyam3897/TG](https://github.com/divyam3897/TG))**, scaled to run on a laptop, plus one curiosity-driven extension: **per-layer, norm-aware downscaling**.

The paper's central claim: given only past model checkpoints and *no access to the future*, none of the popular "predict the future model" methods — model averaging/merging, EMA, or Taylor-series parameter extrapolation — consistently beats the trivial baseline of deploying the **most recent** model. The only method that reliably avoids degradation is **downscaling** (shrinking recent parameters toward zero), and even that helps for language modeling but not for vision. This repo reproduces that pattern and probes the "why."

## Headline result (Yearbook, 3 seeds)

![forward transfer results](assets/forward_transfer.png)

Reproduced cleanly: **no method reliably beats simply deploying the most recent model.** `taylor` edges ahead by 1.5 points, but that gap sits *inside* a 15-point standard deviation — the "apparent improvement that isn't" the paper warns about.

Two findings I care about most, because they test the paper's *explanations* for why these methods fail — and on Yearbook neither holds:

- **The loss basins are connected** (barrier ≈ 0.003). The paper blames disconnected basins; here they're connected and the methods still fail.
- **The overconfidence story isn't supported** (α-vs-norm correlation r = +0.15, wrong sign for the "shrink because norms grow" hypothesis).

So the methods fail even though neither intuitive explanation applies — which strengthens the paper's deepest point: without assumptions about how data evolves, the future is arbitrary. My original method **ATD** (shrink only along the recent-change subspace) also does not beat the baseline — an honest, informative negative result. Full write-up in `ONE_PAGER.md` / `FINDINGS.md`.


## What this repo does

- Builds a **checkpoint trajectory** by sequential fine-tuning (continual learning): the model at time *t* is initialized from θ_{t−1} and fine-tuned on the current timestep only (paper Eq. 5).
- Implements every method — `recent`, `average`, `ema`, `downscale`, `taylor` (extrapolation), the extension `per_layer_downscale`, and an **original method `aniso_downscale` (ATD)** — as pure operations on past parameter vectors (`src/tg/methods.py`).
- Runs across **multiple seeds** and reports **mean ± std** with error bars (the paper is a skepticism paper — single-run numbers would undercut it).
- Evaluates **δ-forward transfer**: for each time *t* and horizon δ, estimate θ̃_{t+δ} from past checkpoints and score it on genuinely future data.
- Tunes the α hyperparameters using **past/current data only**, by simulating a one-step-back deployment (paper Eq. 6). This is enforced in code — the tuner is structurally unable to see future data. *This is the whole point of the paper.*
- Produces the key figures: forward-transfer curves, parameter-norm-vs-time, per-layer norm growth, and a PCA of the parameter trajectory.
- Runs a **basin-barrier diagnostic** (`src/tg/analysis.py`): the loss along the straight line between consecutive checkpoints. This *measures* the mechanism the paper only asserts — whether checkpoints sit in connected or disconnected loss basins.

## Quick start

```bash
pip install -r requirements.txt

# 1) Sanity check the whole pipeline (<1 min, no download, runs anywhere):
python run.py --smoke-test

# 2) Full synthetic run, 3 seeds with error bars (self-contained benchmark):
python run.py --dataset synthetic --T 8 --epochs 3 --seeds 0 1 2

# 3) Real Yearbook reproduction on your Mac (downloads Wilds-Time on first run):
pip install wild-time-data
python run.py --dataset yearbook --epochs 5 --device mps --seeds 0 1 2 --max-steps 12
```

Outputs land in `results/`: `forward_transfer.csv` (per seed/t/δ), `summary_by_method.csv` (mean ± std), `basin_barriers.csv`, `layer_growth.json`, `run_meta.json` (config/versions/git/timing), and figures — forward-transfer with error bars, norm trajectory, per-layer growth, PCA, α-vs-norm scatter, and the two basin-barrier plots. The console prints a per-method mean ± std table, the "does any method reliably beat Recent?" verdict, the α-vs-norm correlation, and the mean basin barrier.

Unit tests (method math + convex-hull / direction-preservation guarantees):

```bash
pip install pytest && pytest -q
```

## The extension: per-layer, norm-aware downscaling

The paper downscales the whole model by a single scalar α and argues it works because parameter L2-norm grows over time ("overconfidence"). But norm growth is **not uniform across layers** — some layers barely move while others blow up. Hypothesis: a **per-layer** downscaling factor, still tuned only on past data, should shrink high-growth layers harder and recover gains the global scalar misses — especially on the vision task where global downscaling underperforms.

Implementation (`methods.per_layer_downscale`): measure each layer's relative norm growth over the trajectory, then set `alpha_layer = alpha_base ** (growth_layer / mean_growth)`, tuning a single scalar `alpha_base` on past data. When growth is uniform this reduces exactly to global downscaling (see `tests/test_methods.py`).

`results/fig_layer_growth.png` shows the non-uniform growth that motivates it; `summary_by_method.csv` shows whether it helps. **Report the result honestly — a negative result (per-layer also fails to beat Recent) is a valid, on-brand finding for this paper.**

## Original method: Anisotropic Trajectory Downscaling (ATD)

The paper's downscaling shrinks *all* parameters toward zero (global); the per-layer extension shrinks each layer. Both assume the model's over-confidence in the present is spread evenly. **ATD makes a sharper hypothesis:** the model's recent adaptation to the current timestep lives in a low-dimensional subspace — the directions θ actually moved over the last few checkpoints — so we should shrink θ_t *only along that "recency subspace,"* leaving the stable bulk untouched.

Mechanically (`methods.anisotropic_trajectory_downscale`): take the last *m* parameter-update deltas, extract their top-*k* principal directions by SVD (the recency subspace), project θ_t onto it, and damp only that component by β ∈ [0,1] (tuned past-only):

```
theta_tilde = theta_t - (1 - beta) * P_recency(theta_t)
```

β = 1 recovers `recent`; if the subspace were the whole space it would recover global `downscale`. Crucially, ATD is **not** a convex combination of checkpoints — it is a directed subspace rescaling — so it lies *outside* the paper's interpolation/extrapolation family. The hypothesis worth testing: ATD may help on vision (where global downscaling failed) by damping only the present-specific directions. A negative result is still informative — it would say the vision failure isn't about a low-dimensional recency direction at all. Unit tests in `tests/test_methods.py` pin down both limiting cases.

## The diagnostic: are consecutive checkpoints in connected basins?

The paper argues interpolation fails because independently-drifting checkpoints land in disconnected loss basins (non-identifiability). Rather than take that on faith, this repo *measures* it: `trajectory_barriers` interpolates linearly between θ_t and θ_{t+1} (parameters **and** buffers) and reports the **loss barrier** — how high the loss climbs above the straight line joining the two endpoint losses. A near-zero barrier means the basins are connected (interpolation is meaningful); a large barrier means they are not (interpolation must cross high-loss regions, so no amount of averaging will help).

Outputs: `results/basin_barriers.csv`, `fig_loss_path.png` (one representative path), `fig_barriers.png` (barrier per timestep). The console prints the mean barrier. **The payoff question to explore:** does the barrier differ between modalities — low for text, high for vision? If so, it *explains* why downscaling helps text but not vision, going a step beyond the paper. (You'll need one text run to make that comparison; the language track is scaffolded for Colab.)

## Repository layout

```
run.py                     # entrypoint (--smoke-test / --dataset / --device)
src/tg/
  utils.py                 # seeding, device, flat param-vector <-> model
  model.py                 # SmallCNN for 32x32x1 images
  data.py                  # synthetic temporal stream + Wilds-Time Yearbook loader
  methods.py               # recent / average / ema / downscale / taylor / per_layer
  engine.py                # sequential training, past-only alpha tuning, forward transfer
  analysis.py              # norm trajectories, per-layer growth, PCA
  plots.py                 # figure generation
tests/test_methods.py      # unit tests for the method math
tools/verify_logic_numpy.py# torch-free logic check of the whole pipeline
results/                   # generated csv + figures
```

## Reproduction status & honesty notes

- **Scope.** This is a *scaled* reproduction aimed at the paper's qualitative findings, not bit-exact numbers. It uses a small CNN and a subset of the paper's tasks (Yearbook vision + a self-contained synthetic stream; the T5/NewsRoom language track is scaffolded for Colab).
- **Verified here.** All modules byte-compile; the algorithm runs end-to-end via a torch-free NumPy mirror (`tools/verify_logic_numpy.py`) which confirms the checkpoint-trajectory build, the past-only α tuning, and the forward-transfer loop. On that smooth synthetic stream, Taylor extrapolation already lands ≤ Recent, echoing the paper.
- **Run on your hardware.** The PyTorch experiments (`run.py`) are intended to run on an Apple-Silicon Mac (MPS) / Colab; fill in the real numbers there before sharing.
- **What to claim.** "I reproduced the qualitative finding that no interpolation/extrapolation method reliably beats the recent-model baseline, and I tested a per-layer variant of the one method that does help." Nothing more, nothing less.

## Findings note

See `ONE_PAGER.md` for a short narrative write-up and `FINDINGS.md` for the detailed results note to fill in after your runs — this is the document to send Professor Cho alongside the repo link.

## Citation

```
@inproceedings{madaan2026temporal,
  title={Temporal Generalization: A Reality Check},
  author={Madaan, Divyam and Chopra, Sumit and Cho, Kyunghyun},
  booktitle={ICLR},
  year={2026}
}
```

MIT License.
