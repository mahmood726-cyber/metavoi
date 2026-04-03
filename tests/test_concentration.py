"""Tests for concentration inequality bounds."""

from metavoi.concentration import compute_concentration_bounds


def test_hoeffding_geq_bernstein(bcg_input):
    """Hoeffding bound must be >= Bernstein bound (Bernstein is tighter)."""
    res = compute_concentration_bounds(bcg_input)
    assert res["hoeffding_bound"] >= res["bernstein_bound"], (
        f"Hoeffding {res['hoeffding_bound']:.6f} < Bernstein {res['bernstein_bound']:.6f}"
    )


def test_bounds_positive(bcg_input):
    """All bounds and range must be strictly positive."""
    res = compute_concentration_bounds(bcg_input)
    assert res["hoeffding_bound"] > 0
    assert res["bernstein_bound"] > 0
    assert res["mcdiarmid_bound"] > 0
    assert res["sub_gaussian_sigma"] > 0
    assert res["range_nb"] > 0


def test_finite_sample_ci_contains_positive(bcg_input):
    """Finite-sample CI upper bound must be positive (EVPI >= 0)."""
    res = compute_concentration_bounds(bcg_input)
    ci_lo, ci_hi = res["finite_sample_ci"]
    assert ci_hi > 0, f"CI upper bound {ci_hi} should be positive"
    assert ci_hi > ci_lo, "CI must have positive width"


def test_required_n_positive(bcg_input):
    """Required n for epsilon=0.01 must be a positive integer."""
    res = compute_concentration_bounds(bcg_input)
    assert res["required_n_for_epsilon_001"] > 0
    assert isinstance(res["required_n_for_epsilon_001"], int)


def test_certain_input_tighter_bounds(certain_input):
    """Very certain evidence should have smaller bounds than uncertain."""
    from metavoi.concentration import compute_concentration_bounds
    res = compute_concentration_bounds(certain_input)
    # With very tight SE and zero tau2, bounds should be small
    assert res["hoeffding_bound"] < 0.5, (
        f"Certain input Hoeffding bound {res['hoeffding_bound']:.4f} unexpectedly large"
    )
    assert res["range_nb"] < 1.0, (
        f"Certain input range {res['range_nb']:.4f} unexpectedly large"
    )
