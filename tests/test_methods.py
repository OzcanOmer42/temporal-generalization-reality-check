"""Unit tests for the core methods and the past-only tuning guarantee."""
import torch

from src.tg import methods as M


def _vecs():
    torch.manual_seed(0)
    return [torch.randn(20) for _ in range(4)]


def test_recent_is_last():
    v = _vecs()
    assert torch.equal(M.recent(v), v[-1])


def test_average_in_convex_hull():
    v = _vecs()
    avg = M.average(v)
    stacked = torch.stack(v)
    # each coordinate of the average lies within [min, max] of the checkpoints
    assert torch.all(avg <= stacked.max(0).values + 1e-6)
    assert torch.all(avg >= stacked.min(0).values - 1e-6)


def test_downscale_reduces_norm():
    v = _vecs()
    assert M.downscale(v, 0.5).norm() < v[-1].norm()


def test_downscale_preserves_direction():
    v = _vecs()
    d = M.downscale(v, 0.7)
    cos = torch.dot(d, v[-1]) / (d.norm() * v[-1].norm())
    assert cos > 0.999


def test_taylor_extrapolation_direction():
    v = _vecs()
    out = M.taylor_extrapolate(v, alpha=1.0, dt=1.0)
    expected = v[-1] + (v[-1] - v[-2])
    assert torch.allclose(out, expected)


def test_per_layer_reduces_to_global_when_growth_uniform():
    v = _vecs()
    slices = {"a": (0, 10), "b": (10, 20)}
    # uniform growth -> per-layer with alpha_base equals global downscale
    growth = {"a": 1.0, "b": 1.0}
    out = M.per_layer_downscale(v, slices, alpha_base=0.8, growth=growth)
    assert torch.allclose(out, 0.8 * v[-1], atol=1e-5)


def test_atd_beta_one_is_recent():
    v = _vecs()
    out = M.anisotropic_trajectory_downscale(v, beta=1.0, k=2, m=3)
    assert torch.allclose(out, v[-1], atol=1e-5)


def test_atd_only_shrinks_inside_recency_subspace():
    v = _vecs()
    dirs = M.recency_subspace(v, k=2, m=3)
    out = M.anisotropic_trajectory_downscale(v, beta=0.0, k=2, m=3)
    # beta=0 removes the recency-subspace component entirely: projection of the
    # output onto those directions must be ~0.
    assert torch.allclose(dirs @ out, torch.zeros(dirs.shape[0]), atol=1e-4)
    # the orthogonal complement is preserved: component of (theta - out) lies in
    # the subspace, so removing it shouldn't touch orthogonal directions.
    removed = v[-1] - out
    # 'removed' should equal its own projection onto the subspace
    proj_removed = (dirs @ removed) @ dirs
    assert torch.allclose(removed, proj_removed, atol=1e-4)
