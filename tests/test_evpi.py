import pytest
from metavoi.evpi import compute_evpi, compute_evpi_population
from metavoi.posterior import predictive_distribution, discount_factor


def test_evpi_positive_when_uncertain(uncertain_input):
    draws = predictive_distribution(uncertain_input)
    evpi = compute_evpi(draws, uncertain_input.mcid)
    assert evpi > 0.01


def test_evpi_near_zero_when_certain(certain_input):
    draws = predictive_distribution(certain_input)
    evpi = compute_evpi(draws, certain_input.mcid)
    assert evpi < 0.001


def test_evpi_increases_with_uncertainty(certain_input, uncertain_input):
    draws_c = predictive_distribution(certain_input)
    draws_u = predictive_distribution(uncertain_input)
    evpi_c = compute_evpi(draws_c, certain_input.mcid)
    evpi_u = compute_evpi(draws_u, uncertain_input.mcid)
    assert evpi_u > evpi_c


def test_evpi_nonnegative(bcg_input):
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    assert evpi >= 0.0


def test_evpi_population_scales_linearly(bcg_input):
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    df = discount_factor(bcg_input.discount_rate, bcg_input.horizon_years)
    pop_evpi = compute_evpi_population(evpi, bcg_input.population, df)
    pop_evpi_half = compute_evpi_population(evpi, bcg_input.population // 2, df)
    assert abs(pop_evpi - 2 * pop_evpi_half) < 1.0


def test_evpi_bounded_by_max_nb(bcg_input):
    draws = predictive_distribution(bcg_input)
    evpi = compute_evpi(draws, bcg_input.mcid)
    max_possible = max(abs(draws.mean() - bcg_input.mcid), 0)
    assert evpi <= max_possible + 0.01
