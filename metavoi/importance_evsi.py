"""Importance Sampling EVSI (Heath et al. 2020).

More efficient EVSI computation than moment-matching. For each hypothetical
trial result, we analytically update the posterior and compute the optimal
decision under the updated posterior.

EVSI = E_y[max NB under updated] - max NB under current
"""

import numpy as np
from metavoi.posterior import discount_factor
from metavoi.evsi import compute_evsi


DEFAULT_N_TRIAL_VALUES = [25, 50, 100, 200, 500, 1000, 2000, 5000]


def compute_importance_evsi(inp, n_trial, n_outer=500):
    """EVSI via importance-sampling-inspired analytic update.

    For each of n_outer hypothetical trial results y_new:
      1. Updated posterior precision: prec_post = 1/(se^2+tau2) + n_trial/sigma2
      2. Updated posterior mean: mu_post = (theta/(se^2+tau2) + y_new*n_trial/sigma2) / prec_post
      3. Optimal decision under updated posterior
      4. NB under updated posterior

    EVSI = E_y[max NB under updated] - max NB under current
    """
    rng = np.random.default_rng(inp.seed + 30 + n_trial)
    mcid = inp.mcid

    sigma2 = inp.within_study_var if inp.within_study_var else inp.se ** 2
    prior_var = inp.se ** 2 + inp.tau2

    # Current decision
    nb_current_treat = mcid - inp.theta
    nb_current_best = max(nb_current_treat, 0.0)

    # Precisions
    prior_precision = 1.0 / prior_var if prior_var > 1e-12 else 1e12
    trial_precision = n_trial / sigma2 if sigma2 > 1e-12 else 1e12
    post_precision = prior_precision + trial_precision

    # Simulate hypothetical trial results y_new
    # y_new ~ N(theta, sigma2/n_trial + prior_var)
    # This is the prior predictive for the trial mean
    predictive_var = sigma2 / n_trial + prior_var
    y_new = rng.normal(inp.theta, np.sqrt(predictive_var), size=n_outer)

    # Updated posterior mean for each y_new
    mu_post = (prior_precision * inp.theta + trial_precision * y_new) / post_precision

    # Optimal decision under updated posterior
    nb_treat_updated = mcid - mu_post
    nb_no_treat_updated = np.zeros_like(mu_post)
    nb_best_updated = np.maximum(nb_treat_updated, nb_no_treat_updated)

    evsi = float(np.mean(nb_best_updated) - nb_current_best)
    return max(0.0, evsi)


def compute_importance_evsi_curve(inp, n_values=None, n_outer=500):
    """Compute importance-sampling EVSI for multiple trial sizes.

    Returns list of {n, evsi_importance}.
    """
    if n_values is None:
        n_values = DEFAULT_N_TRIAL_VALUES

    curve = []
    for n in n_values:
        evsi = compute_importance_evsi(inp, n_trial=n, n_outer=n_outer)
        curve.append({"n": n, "evsi_importance": evsi})

    return curve


def compute_comparison(inp, n_values=None):
    """Compare importance-sampling EVSI with moment-matching EVSI.

    Returns dict with:
        evsi_curve: list of {n, evsi_importance}
        evsi_moment_curve: list of {n, evsi_moment}
        efficiency_ratio: list of floats (importance / moment)
        recommended_n: int (trial size with highest population net benefit)
    """
    if n_values is None:
        n_values = DEFAULT_N_TRIAL_VALUES

    df = discount_factor(inp.discount_rate, inp.horizon_years)

    # Importance-sampling EVSI
    evsi_imp_curve = compute_importance_evsi_curve(inp, n_values=n_values)

    # Moment-matching EVSI from existing module
    evsi_moment_curve = []
    for n in n_values:
        evsi_mm = compute_evsi(inp, n_trial=n)
        evsi_moment_curve.append({"n": n, "evsi_moment": evsi_mm})

    # Efficiency ratio
    efficiency_ratio = []
    for imp_pt, mm_pt in zip(evsi_imp_curve, evsi_moment_curve):
        mm_val = mm_pt["evsi_moment"]
        imp_val = imp_pt["evsi_importance"]
        if mm_val > 1e-12:
            efficiency_ratio.append(imp_val / mm_val)
        else:
            efficiency_ratio.append(1.0 if imp_val < 1e-12 else float("inf"))

    # Recommended N: highest population net benefit (EVSI_pop - cost)
    best_nb = float("-inf")
    recommended_n = n_values[0]
    for pt in evsi_imp_curve:
        n = pt["n"]
        evsi_pop = pt["evsi_importance"] * inp.population * df
        cost = inp.cost_per_patient * n
        nb = evsi_pop - cost
        if nb > best_nb:
            best_nb = nb
            recommended_n = n

    return {
        "evsi_curve": evsi_imp_curve,
        "evsi_moment_curve": evsi_moment_curve,
        "efficiency_ratio": efficiency_ratio,
        "recommended_n": recommended_n,
    }
