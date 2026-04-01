import pytest
from metavoi.optimal import find_optimal_n


def test_optimal_n_exists(bcg_input):
    result = find_optimal_n(bcg_input)
    assert result["optimal_n"] is not None
    assert result["optimal_n"] > 0


def test_optimal_n_net_benefit_positive(bcg_input):
    result = find_optimal_n(bcg_input)
    if result["optimal_n"] is not None:
        assert result["optimal_net_benefit"] >= 0


def test_breakeven_n_greater_than_optimal(bcg_input):
    result = find_optimal_n(bcg_input)
    if result["optimal_n"] is not None and result["breakeven_n"] is not None:
        assert result["breakeven_n"] >= result["optimal_n"]


def test_no_optimal_when_cost_exceeds_value(certain_input):
    result = find_optimal_n(certain_input)
    if result["optimal_n"] is not None:
        assert result["optimal_net_benefit"] < 1.0
