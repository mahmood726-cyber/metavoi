import pytest
from metavoi.evsi import compute_evsi, compute_evsi_curve
from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def test_evsi_nonnegative(bcg_input):
    evsi = compute_evsi(bcg_input, n_trial=500)
    assert evsi >= 0.0


def test_evsi_increases_with_n(bcg_input):
    evsi_100 = compute_evsi(bcg_input, n_trial=100)
    evsi_1000 = compute_evsi(bcg_input, n_trial=1000)
    assert evsi_1000 >= evsi_100 - 0.015  # MC noise tolerance


def test_evsi_bounded_by_evpi(bcg_input):
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    evsi = compute_evsi(bcg_input, n_trial=10000)
    assert evsi <= evpi + 0.005


def test_evsi_near_zero_when_certain(certain_input):
    evsi = compute_evsi(certain_input, n_trial=500)
    assert evsi < 0.001


def test_evsi_curve_returns_list(bcg_input):
    curve = compute_evsi_curve(bcg_input, n_values=[100, 500, 1000])
    assert len(curve) == 3
    assert all(pt.n > 0 for pt in curve)
    assert all(pt.evsi >= 0 for pt in curve)


def test_evsi_curve_monotonic(bcg_input):
    curve = compute_evsi_curve(bcg_input, n_values=[100, 500, 2000])
    evsis = [pt.evsi for pt in curve]
    for i in range(len(evsis) - 1):
        assert evsis[i + 1] >= evsis[i] - 0.015  # MC noise tolerance
