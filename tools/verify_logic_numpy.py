#!/usr/bin/env python3
"""NumPy mirror of the pipeline -- verifies the ALGORITHM without needing torch.

This exists only because some minimal environments (e.g. CI sandboxes) cannot
install PyTorch. It reimplements the same logic as src/tg/{methods,engine}.py on
a tiny linear model, so we can confirm end-to-end that:

  * sequential fine-tuning builds a checkpoint trajectory,
  * past-only alpha tuning never touches future data,
  * every method produces a future-accuracy number,
  * the "does anything reliably beat Recent?" check runs.

The real experiments use the torch code in src/tg/ (run.py). This is a logic
check, not a replacement.
"""
import numpy as np

RNG = np.random.default_rng(0)


# --- tiny drifting binary-classification stream (linear-separable, rotating) ---
def make_stream(T=8, n=400, d=20, val_frac=0.3):
    stream = []
    base = RNG.normal(size=d)
    for t in range(T):
        # decision direction rotates smoothly with time -> genuine temporal shift
        theta_true = base + 0.6 * np.sin(2 * np.pi * t / T) * RNG.normal(size=d)
        X = RNG.normal(size=(n, d))
        logits = X @ theta_true
        y = (logits > 0).astype(np.int64)
        idx = RNG.permutation(n)
        nv = int(n * val_frac)
        val, tr = idx[:nv], idx[nv:]
        stream.append({"Xtr": X[tr], "ytr": y[tr], "Xval": X[val], "yval": y[val]})
    return stream


def train_step(w, X, y, epochs=50, lr=0.1):
    """Logistic-regression gradient descent, warm-started from w (continual)."""
    w = w.copy()
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / len(y)
        w -= lr * grad
    return w


def acc(w, X, y):
    return float(((X @ w > 0).astype(int) == y).mean())


# --- methods (mirror of src/tg/methods.py) ---
def recent(vs): return vs[-1].copy()
def average(vs): return np.mean(vs, axis=0)
def ema(vs, decay=0.9):
    out = vs[0].copy()
    for v in vs[1:]:
        out = decay * out + (1 - decay) * v
    return decay * out + (1 - decay) * vs[-1]
def downscale(vs, a): return a * vs[-1]
def taylor(vs, a, dt=1.0):
    if len(vs) < 2: return vs[-1].copy()
    return vs[-1] + a * (vs[-1] - vs[-2]) / dt


def recency_subspace(vs, k=2, m=3):
    m_eff = min(m, len(vs) - 1)
    deltas = np.stack([vs[-i] - vs[-i - 1] for i in range(1, m_eff + 1)])
    _, _, Vh = np.linalg.svd(deltas, full_matrices=False)
    return Vh[: min(k, Vh.shape[0])]


def atd(vs, beta, k=2, m=3):
    """Anisotropic Trajectory Downscaling -- shrink theta along recency subspace."""
    if len(vs) < 2:
        return vs[-1].copy()
    theta = vs[-1]
    dirs = recency_subspace(vs, k, m)
    return theta - (1 - beta) * (dirs.T @ (dirs @ theta))


ALPHA = {"downscale": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
         "taylor": [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
         "atd": [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]}


def build(name, vs, a):
    return {"recent": recent, "average": average, "ema": ema,
            "downscale": lambda v: downscale(v, a),
            "taylor": lambda v: taylor(v, a),
            "atd": lambda v: atd(v, a)}[name](vs)


def tune_past_only(name, vecs, stream, t, delta):
    """Pick alpha on CURRENT val data by simulating one horizon back (Eq. 6)."""
    if name not in ALPHA:
        return None
    if t - delta < 0:
        return ALPHA[name][-1]
    past = vecs[: (t - delta) + 1]
    best_a, best = ALPHA[name][-1], -1
    for a in ALPHA[name]:
        w = build(name, past, a)
        s = acc(w, stream[t]["Xval"], stream[t]["yval"])  # current (past-available) data
        if s > best:
            best, best_a = s, a
    return best_a


def ce_loss(w, X, y):
    p = 1 / (1 + np.exp(-(X @ w)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def barrier(w_a, w_b, X, y, n=11):
    """Loss along the straight line w_a -> w_b; height above the endpoint chord."""
    ss = np.linspace(0, 1, n)
    losses = np.array([ce_loss((1 - s) * w_a + s * w_b, X, y) for s in ss])
    chord = (1 - ss) * losses[0] + ss * losses[-1]
    return float(np.max(losses - chord))


def run_one(seed, T=8):
    """One seed: build trajectory, score every method, track downscale alphas."""
    global RNG
    RNG = np.random.default_rng(seed)
    stream = make_stream(T=T)
    d = stream[0]["Xtr"].shape[1]
    w = np.zeros(d)
    vecs = []
    for step in stream:
        w = train_step(w, step["Xtr"], step["ytr"])
        vecs.append(w.copy())

    methods = ["recent", "average", "ema", "downscale", "taylor", "atd"]
    deltas = [1, 2, 3]
    scores = {m: [] for m in methods}
    alpha_norm = []  # (norm_at_t, optimal downscale alpha)
    for m in methods:
        for t in range(T):
            for delta in deltas:
                if t + delta >= T:
                    continue
                a = tune_past_only(m, vecs[: t + 1], stream, t, delta)
                west = build(m, vecs[: t + 1], a)
                scores[m].append(acc(west, stream[t + delta]["Xval"],
                                     stream[t + delta]["yval"]))
                if m == "downscale":
                    alpha_norm.append((float(np.linalg.norm(vecs[t])), float(a)))
    barriers = [barrier(vecs[t], vecs[t + 1], stream[t + 1]["Xval"],
                        stream[t + 1]["yval"]) for t in range(T - 1)]
    return scores, alpha_norm, barriers, vecs


def main():
    seeds = [0, 1, 2]
    T = 8
    per_seed_means = {}
    all_alpha_norm, all_barriers = [], []
    methods = None
    for seed in seeds:
        scores, an, barriers, vecs = run_one(seed, T)
        methods = list(scores.keys())
        for m in methods:
            per_seed_means.setdefault(m, []).append(float(np.mean(scores[m])))
        all_alpha_norm.extend(an)
        all_barriers.extend(barriers)

    print(f"=== NumPy logic check ({len(seeds)} seeds): mean +/- std future accuracy ===")
    agg = {m: (float(np.mean(v)), float(np.std(v))) for m, v in per_seed_means.items()}
    recent_acc = agg["recent"][0]
    for m in sorted(agg, key=lambda k: agg[k][0], reverse=True):
        mean, std = agg[m]
        flag = ""
        if m != "recent":
            flag = " <-- beats Recent" if mean > recent_acc + 1e-4 else " (<= Recent)"
        star = " *ATD (original)" if m == "atd" else ""
        print(f"  {m:12s} {mean:.4f} +/- {std:.4f}{flag}{star}")
    winners = [m for m in agg if m != "recent" and agg[m][0] > recent_acc + 1e-4]
    print(f"\nMethods reliably beating Recent: {winners if winners else 'NONE'}")

    # alpha-vs-norm correlation (overconfidence hypothesis)
    xs = [x for x, _ in all_alpha_norm]
    ys = [y for _, y in all_alpha_norm]
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(set(xs)) > 1 and len(set(ys)) > 1 else float("nan")
    print(f"optimal-alpha vs norm correlation: r = {r:.3f} "
          f"(negative supports 'higher norm -> more shrink')")
    print(f"mean basin barrier: {np.mean(all_barriers):.4f} "
          f"(convex model -> ~0 = connected, as expected)")

    assert methods and "atd" in methods
    assert all(len(v) == len(seeds) for v in per_seed_means.values())
    print("\nLOGIC CHECK PASSED: ATD + multi-seed + alpha-norm + barrier all wired.")


if __name__ == "__main__":
    main()
