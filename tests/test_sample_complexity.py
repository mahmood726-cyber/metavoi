"""Tests for sample complexity theory."""

from metavoi.sample_complexity import compute_sample_complexity


def test_pac_n_positive(bcg_input):
    """PAC sample size must be a positive integer."""
    res = compute_sample_complexity(bcg_input)
    assert res["pac_n"] > 0
    assert isinstance(res["pac_n"], int)


def test_all_n_positive(bcg_input):
    """All sample size estimates must be positive."""
    res = compute_sample_complexity(bcg_input)
    assert res["pac_n"] > 0
    assert res["minimax_n"] > 0
    assert res["bayesian_n"] > 0
    assert res["fano_lower_bound_n"] > 0
    assert res["adaptive_expected_n"] > 0


def test_comparison_dict_complete(bcg_input):
    """Comparison dict must have all 5 methods."""
    res = compute_sample_complexity(bcg_input)
    expected_keys = {"pac", "minimax", "bayesian", "fano", "adaptive"}
    assert set(res["comparison"].keys()) == expected_keys


def test_effective_sigma_positive(bcg_input):
    """Effective sigma must be positive."""
    res = compute_sample_complexity(bcg_input)
    assert res["effective_sigma"] > 0


def test_certain_needs_fewer_patients(certain_input, uncertain_input):
    """Certain evidence should require fewer patients than uncertain."""
    res_certain = compute_sample_complexity(certain_input)
    res_uncertain = compute_sample_complexity(uncertain_input)
    assert res_certain["pac_n"] < res_uncertain["pac_n"], (
        f"Certain PAC n ({res_certain['pac_n']}) >= uncertain ({res_uncertain['pac_n']})"
    )
