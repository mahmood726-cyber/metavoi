import pytest
from metavoi.evppi import compute_evppi
from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def test_evppi_theta_plus_tau2_lte_evpi(bcg_input):
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    evppi = compute_evppi(bcg_input)
    assert evppi["theta"] + evppi["tau2"] <= evpi + 0.01


def test_evppi_all_nonnegative(bcg_input):
    evppi = compute_evppi(bcg_input)
    assert evppi["theta"] >= 0.0
    assert evppi["tau2"] >= 0.0


def test_evppi_theta_dominant_when_low_tau2(statin_input):
    evppi = compute_evppi(statin_input)
    if evppi["theta"] + evppi["tau2"] > 0.001:
        assert evppi["theta"] >= evppi["tau2"]


def test_evppi_returns_fraction(bcg_input):
    evppi = compute_evppi(bcg_input)
    assert 0.0 <= evppi["theta_fraction"] <= 1.0


def test_evppi_dominant_parameter_is_string(bcg_input):
    evppi = compute_evppi(bcg_input)
    assert evppi["dominant"] in ("theta", "tau2")
