"""Tests for Renyi Entropy VoI module."""

import math
import pytest
from metavoi.renyi_voi import compute_renyi_voi, _shannon_entropy_binary


def test_renyi_at_alpha_1_equals_shannon(bcg_input):
    """H_alpha at alpha=1 should equal Shannon entropy."""
    result = compute_renyi_voi(bcg_input)
    h1 = result["renyi_entropies"][1.0]
    shannon = result["shannon_mi"]
    assert abs(h1 - shannon) < 1e-10, (
        f"H_1 ({h1}) != Shannon MI ({shannon})"
    )


def test_min_entropy_leq_shannon(bcg_input):
    """Min-entropy (alpha=inf) <= Shannon entropy (alpha=1)."""
    result = compute_renyi_voi(bcg_input)
    h_inf = result["min_entropy"]
    h_1 = result["renyi_entropies"][1.0]
    assert h_inf <= h_1 + 1e-10, (
        f"Min-entropy ({h_inf}) > Shannon ({h_1})"
    )


def test_tsallis_q2_is_gini(uncertain_input):
    """Tsallis at q=2 should equal Gini impurity: 2*p*(1-p)."""
    result = compute_renyi_voi(uncertain_input)
    # Recompute p_treat from the input
    from scipy import stats
    import numpy as np
    pred_sd = np.sqrt(uncertain_input.se**2 + uncertain_input.tau2)
    p_treat = stats.norm.cdf(uncertain_input.mcid, loc=uncertain_input.theta, scale=pred_sd)
    p_treat = np.clip(p_treat, 1e-15, 1.0 - 1e-15)
    expected_gini = 2.0 * p_treat * (1.0 - p_treat)
    actual = result["tsallis_entropies"][2.0]
    assert abs(actual - expected_gini) < 1e-10, (
        f"Tsallis q=2 ({actual}) != Gini ({expected_gini})"
    )


def test_entropy_spectrum_monotone_decreasing(bcg_input):
    """Renyi entropy H_alpha should decrease (or stay equal) as alpha increases."""
    result = compute_renyi_voi(bcg_input)
    spectrum = result["entropy_spectrum"]
    for i in range(1, len(spectrum)):
        assert spectrum[i]["H"] <= spectrum[i - 1]["H"] + 1e-10, (
            f"H not monotone at alpha={spectrum[i]['alpha']}: "
            f"{spectrum[i]['H']} > {spectrum[i-1]['H']}"
        )


def test_all_renyi_entropies_nonnegative(statin_input):
    """All Renyi entropies should be >= 0."""
    result = compute_renyi_voi(statin_input)
    for key, val in result["renyi_entropies"].items():
        assert val >= -1e-10, f"Renyi H_{key} = {val} is negative"
