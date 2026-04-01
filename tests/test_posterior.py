import numpy as np
import pytest
from metavoi.models import VoIInput
from metavoi.posterior import predictive_distribution, p_wrong_decision, discount_factor


def test_predictive_mean_equals_theta(bcg_input):
    draws = predictive_distribution(bcg_input)
    assert abs(np.mean(draws) - bcg_input.theta) < 0.05


def test_predictive_var_includes_tau2(bcg_input):
    draws = predictive_distribution(bcg_input)
    expected_var = bcg_input.se ** 2 + bcg_input.tau2
    assert abs(np.var(draws) - expected_var) < 0.05


def test_predictive_length(bcg_input):
    draws = predictive_distribution(bcg_input)
    assert len(draws) == bcg_input.n_sim


def test_p_wrong_near_zero_when_certain(certain_input):
    draws = predictive_distribution(certain_input)
    p = p_wrong_decision(draws, certain_input.mcid)
    assert p < 0.01


def test_p_wrong_near_half_when_theta_equals_mcid():
    inp = VoIInput(theta=-0.2, se=0.10, tau2=0.0, k=10, mcid=-0.2,
                   n_sim=50_000, seed=42)
    draws = predictive_distribution(inp)
    p = p_wrong_decision(draws, inp.mcid)
    assert 0.40 < p < 0.60


def test_discount_factor_no_discount():
    assert abs(discount_factor(0.0, 5) - 5.0) < 1e-10


def test_discount_factor_positive_rate():
    df = discount_factor(0.035, 10)
    assert 8.0 < df < 9.0


def test_discount_factor_single_year():
    assert abs(discount_factor(0.035, 1) - 1.0) < 1e-10
