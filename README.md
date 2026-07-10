# Temporal Generalization: A Reality Check — a reproduction, and an idea of my own

This is my reproduction of Madaan, Chopra & Cho, *Temporal Generalization: A Reality Check* ([ICLR 2026](https://arxiv.org/abs/2509.23487), official code [divyam3897/TG](https://github.com/divyam3897/TG)), scaled down to run on a laptop. I started it because the paper's result surprised me and I wanted to see it fail for myself. Along the way I added a method of my own and two diagnostics that measure *why* the failure happens, rather than taking the explanation on faith.

The question the paper asks is a tempting one: models rot as the world drifts, and retraining is expensive, so — given a history of past checkpoints and no access to the future — can you compute a model that will hold up on data you haven't seen yet? It tries the obvious tricks (average past checkpoints, shrink the latest one, extrapolate the trajectory forward) under one strict rule that a lot of prior work quietly breaks: you're not allowed to peek at the future, not even to tune a hyperparameter. Under that rule, none of the tricks reliably beat just deploying the most recent model.

## What I found

![forward transfer results](assets/forward_transfer.png)

The headline reproduced cleanly: on Yearbook, across three seeds, nothing beat the most-recent-model baseline. One method (Taylor extrapolation) edged ahead by about a point and a half, but that gap sits comfortably inside a fifteen-point standard deviation, so it's noise — which is exactly the kind of mirage the paper is warning people about.

The part I didn't expect: the two explanations the paper gives for *why* these methods fail don't actually hold on Yearbook.

- The loss barrier between consecutive checkpoints is essentially flat (≈ 0.003), so the checkpoints sit in connected regions of the loss surface, not the disconnected "basins" the paper points to — yet the methods still fail. (`results/fig_barriers.png`)
- The correlation the "parameters grow overconfident, so shrinking helps" story predicts should be negative came out mildly positive (r ≈ +0.15). (`results/fig_alpha_norm.png`)

So the methods fail even though neither intuitive reason applies here. To me that actually makes the paper's deeper point land harder: without some assumption about how the data changes over time, the future can be arbitrary, and no amount of clever averaging of past weights recovers it.

## The idea I tried (ATD)

The paper's downscaling shrinks the whole model toward zero, on the theory that it's grown overconfident about the present. My hunch was that the overconfidence isn't spread evenly — it should live in the few directions the model has *most recently* moved in, as it adapted to the latest data. So I tried shrinking the model only along that "recency subspace" (the top few principal directions of the recent parameter updates, found with an SVD) and leaving the rest of the model alone. I call it Anisotropic Trajectory Downscaling.

It's a bit different from everything in the paper: it isn't a blend of checkpoints, it's a targeted rescaling of one subspace. On Yearbook it didn't beat the baseline either — but that's a useful thing to know, because it says the vision failure isn't a tidy low-dimensional "recent-direction" problem you can undo with a targeted shrink. The math and the two limiting cases (β = 1 is just the recent model; shrinking every direction is ordinary downscaling) are pinned down by unit tests in `tests/test_methods.py`.

## Running it

```bash
pip install -r requirements.txt

# quick check that everything works (under a minute, no download):
python run.py --smoke-test

# the self-contained synthetic benchmark, 3 seeds:
python run.py --dataset synthetic --T 8 --epochs 3 --seeds 0 1 2

# the real Yearbook reproduction (downloads the Wild-Time data the first time):
pip install wild-time-data
python run.py --dataset yearbook --epochs 5 --device mps --seeds 0 1 2
```

Each run drops its numbers and plots into `results/` — a per-method mean ± std table, the forward-transfer chart with error bars, the basin-barrier plots, the α-vs-norm scatter, and a `run_meta.json` recording the exact config, versions, and timing. There's also a small torch-free script (`tools/verify_logic_numpy.py`) that re-checks the whole pipeline's logic without needing a GPU, and unit tests you can run with `pytest -q`.

## A note on honesty

This is a scaled reproduction, not a carbon copy of the paper. It uses a small CNN and a couple of the paper's tasks (Yearbook, plus a synthetic drifting-data stream I use for quick checks), so I'm matching the paper's qualitative story rather than its exact numbers. The obvious next step, and the one I'd most like to do, is a language track — the paper's most interesting contrast is that shrinking helps text but not vision, and comparing basin barriers across the two would be a real test of the explanation.

Everything I concluded is written up plainly in `ONE_PAGER.md` (also as a PDF) and in more detail in `FINDINGS.md`.

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
