# Can you build a future-proof model from past checkpoints? A reproduction of "Temporal Generalization: A Reality Check"

*Omer Ozcan — a short write-up of a reproduction + extension I did of Madaan, Chopra & Cho (ICLR 2026, arXiv:2509.23487).*

## The problem the paper tackles
Models trained on today's data degrade as the world drifts. Retraining constantly is expensive, so it is tempting to ask: given a history of past model checkpoints, can I compute a model that will generalize to the *future* without ever seeing future data? The paper groups the candidate tricks into interpolation (average or blend past checkpoints, or shrink the latest one) and extrapolation (project the parameter trajectory forward with a Taylor step), and evaluates them under a strict rule that a lot of prior work quietly breaks: no peeking at the future, including when tuning hyperparameters.

## What I did
I reproduced the core result on the Wild-Time Yearbook benchmark (portraits by year, predict gender) with a small CNN and sequential fine-tuning, across 3 seeds with error bars, tuning every knob on past data only — and I enforced that constraint in code so the tuner literally cannot index a future timestep. Then I went a step further and added an original method plus two diagnostics that measure the explanations the paper gives.

## What I found
The central claim reproduced cleanly: no method reliably beats the trivial baseline of deploying the most recent model. One method (Taylor extrapolation) looked like a winner by a hair, but its 1.5-point edge sits inside a 15-point standard deviation — a perfect example of the "apparent improvement that isn't" the paper cautions against.

The part I found most interesting: I measured the two mechanisms the paper offers for *why* these methods fail, and on Yearbook neither one holds. The loss barriers between consecutive checkpoints are essentially zero, so the "disconnected loss basins" explanation doesn't apply here — the basins are connected and the methods still fail. And the correlation the "overconfidence / norm growth" story predicts came out with the wrong sign. So the methods fail even though the intuitive reasons don't apply, which to me strengthens the paper's deepest point: without assumptions about how the data evolves, the future is arbitrary and no weight-space trick recovers it.

## My original method (ATD)
I proposed Anisotropic Trajectory Downscaling: instead of shrinking all parameters toward zero, shrink the model only along the low-dimensional subspace it most recently moved in (top-k directions of recent parameter change), tuned past-only. It's outside the paper's method family. On Yearbook it did not beat the baseline — an honest negative result, but a useful one: it suggests the vision failure isn't a targeted "recent-direction" overfit.

## Honest scope
Scaled reproduction, single vision benchmark, small model — qualitative match, not exact numbers. Obvious next step is a language track to compare behavior (and basin barriers) across modalities.

Code, exact reproduction commands, and all figures: [GitHub link].
