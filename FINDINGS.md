# Findings — reproducing "Temporal Generalization: A Reality Check" + my own extension

*My notes on what I reproduced, what I added, and what I found. — Omer Ozcan*

## What I did
I reproduced the core of Madaan, Chopra & Cho, *Temporal Generalization: A Reality Check* (ICLR 2026, arXiv:2509.23487) on the Wild-Time **Yearbook** benchmark (portraits by year, binary gender prediction) with a small CNN and sequential fine-tuning. I evaluated delta-forward transfer for delta in {1,2,3} over 3 seeds, and — importantly — I tuned every hyperparameter using only past/current data (the paper's Eq. 6), enforced in code so the tuner cannot see the future.

On top of the reproduction I added three things: an **original method** (ATD, below), a **basin-barrier diagnostic** that measures whether consecutive checkpoints are geometrically connected, and an **alpha-vs-norm test** that quantitatively checks the paper's "overconfidence" explanation.

## Did the central claim hold? Yes.
Mean +/- std future accuracy over 3 seeds (higher is better):

| method | mean +/- std | vs. recent |
|---|---|---|
| taylor (extrapolation) | 0.846 +/- 0.156 | +0.015 (within noise) |
| **recent (baseline)** | **0.831 +/- 0.157** | — |
| per-layer downscale | 0.829 +/- 0.159 | <= recent |
| **aniso_downscale / ATD (mine)** | 0.806 +/- 0.165 | <= recent |
| downscale | 0.794 +/- 0.170 | <= recent |
| ema | 0.785 +/- 0.166 | <= recent |
| average | 0.654 +/- 0.176 | <= recent |

No method reliably beats `recent`. My script's naive flag marked `taylor` as a "winner," but its 1.5-point edge sits inside a 15-point standard deviation — it is not statistically distinguishable from the baseline. That apparent-but-fake improvement is exactly the trap the paper warns about, and I think it is a nice illustration of why the error bars matter.

## My original method: ATD (Anisotropic Trajectory Downscaling)
The paper's downscaling shrinks *all* parameters toward zero; I hypothesized that the model's recent, present-specific adaptation lives in a low-dimensional subspace — the directions theta moved over the last few checkpoints — so I shrink theta *only* along that "recency subspace" (top-k SVD of recent deltas), tuned past-only. It is outside the paper's method family (a directed subspace rescaling, not a convex combination of checkpoints).

Result: ATD did **not** beat `recent` on Yearbook (0.806 vs 0.831). An honest negative result — but an informative one: it says the vision failure is not a low-dimensional "recent-direction overfit" that a targeted shrink can undo.

## The two things I found most interesting
Both of the paper's intuitive explanations for *why* these methods fail turned out not to apply on Yearbook:

1. **The basins are connected.** The loss barrier between consecutive checkpoints is about 0.003 (essentially flat). The paper attributes interpolation's failure to checkpoints living in disconnected loss basins — but here they are connected, and the methods still fail. So on this dataset, "disconnected basins" is not the reason.
2. **The overconfidence story is not supported.** The alpha-vs-norm correlation is r = +0.15. The "shrink because parameter norms grow" hypothesis predicts a *negative* correlation; here it is mildly positive.

Taken together, the methods fail even though neither the geometric explanation (disconnected basins) nor the magnitude explanation (norm overconfidence) holds. That points squarely at the paper's deepest claim: without assumptions about how the data-generating process evolves, the future can be arbitrarily different, and no trick on past weights recovers it.

## Honest caveats
This is a scaled reproduction on a single vision benchmark with a small model — I am matching the paper's qualitative findings, not its exact numbers. The natural next step is the language track (T5 on a monthly news corpus), where I would expect downscaling to behave differently and could compare basin barriers across modalities.

## Reproduce
```bash
pip install -r requirements.txt wild-time-data
python run.py --dataset yearbook --epochs 5 --device mps --seeds 0 1 2
```
Code + figures: [GitHub link].
