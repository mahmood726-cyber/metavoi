"""Minimax Regret Analysis — decision-theoretic alternative to expected-value VoI.

Regret for decision d given true theta:
    R(d, theta) = max_{d'} NB(d', theta) - NB(d, theta)

Expected regret integrates R over the posterior.
Minimax regret chooses d that minimises max_theta R(d, theta).
Opportunity loss = E[regret of current optimal decision] = EVPI.
"""

import numpy as np
from metavoi.posterior import predictive_distribution, discount_factor
from metavoi.evpi import compute_evpi
from metavoi.evsi import compute_evsi


def _net_benefit(theta, mcid):
    """NB(treat) = mcid - theta; NB(no_treat) = 0."""
    return mcid - theta


def compute_regret(inp):
    """Full minimax regret analysis.

    Returns dict with:
        expected_regret_treat, expected_regret_no_treat,
        minimax_decision, regret_surface_treat, regret_surface_no_treat,
        theta_grid, regret_optimal_n, ev_optimal_n, opportunity_loss
    """
    rng = np.random.default_rng(inp.seed + 10)
    pred_var = inp.se ** 2 + inp.tau2
    pred_sd = np.sqrt(pred_var)
    mcid = inp.mcid

    # --- 1. Regret surfaces over theta grid ---
    theta_grid = np.linspace(inp.theta - 4 * pred_sd, inp.theta + 4 * pred_sd, 100)

    nb_treat = _net_benefit(theta_grid, mcid)  # mcid - theta
    nb_no_treat = np.zeros_like(theta_grid)     # 0

    # Best possible NB at each grid point
    nb_best = np.maximum(nb_treat, nb_no_treat)

    regret_treat = nb_best - nb_treat       # regret of choosing treat
    regret_no_treat = nb_best - nb_no_treat  # regret of choosing no_treat

    # --- 2. Expected regret under posterior (MC integration) ---
    draws = rng.normal(inp.theta, pred_sd, size=inp.n_sim)

    nb_treat_draws = _net_benefit(draws, mcid)
    nb_no_treat_draws = np.zeros_like(draws)
    nb_best_draws = np.maximum(nb_treat_draws, nb_no_treat_draws)

    expected_regret_treat = float(np.mean(nb_best_draws - nb_treat_draws))
    expected_regret_no_treat = float(np.mean(nb_best_draws - nb_no_treat_draws))

    # --- 3. Minimax decision ---
    max_regret_treat = float(np.max(regret_treat))
    max_regret_no_treat = float(np.max(regret_no_treat))

    if max_regret_treat <= max_regret_no_treat:
        minimax_decision = "treat"
    else:
        minimax_decision = "no_treat"

    # --- 4. Opportunity loss = E[regret of current optimal] = EVPI ---
    mean_theta = np.mean(draws)
    if mean_theta < mcid:
        # Current optimal = treat
        opportunity_loss = expected_regret_treat
    else:
        # Current optimal = no_treat
        opportunity_loss = expected_regret_no_treat

    # --- 5. Regret-based optimal N ---
    n_values = [25, 50, 100, 200, 500, 1000, 2000, 5000]
    df = discount_factor(inp.discount_rate, inp.horizon_years)

    best_regret_n = None
    best_regret_val = float("inf")

    for n in n_values:
        # Compute expected posterior regret after a trial of size n
        post_regret = _posterior_expected_regret(inp, n, rng)
        pop_regret = post_regret * inp.population * df
        net_regret = pop_regret + inp.cost_per_patient * n  # total loss
        if net_regret < best_regret_val:
            best_regret_val = net_regret
            best_regret_n = n

    # EV-optimal N from existing module
    from metavoi.optimal import find_optimal_n
    ev_result = find_optimal_n(inp, n_values=n_values)
    ev_optimal_n = ev_result["optimal_n"]

    return {
        "expected_regret_treat": expected_regret_treat,
        "expected_regret_no_treat": expected_regret_no_treat,
        "minimax_decision": minimax_decision,
        "regret_surface_treat": regret_treat.tolist(),
        "regret_surface_no_treat": regret_no_treat.tolist(),
        "theta_grid": theta_grid.tolist(),
        "regret_optimal_n": best_regret_n,
        "ev_optimal_n": ev_optimal_n,
        "opportunity_loss": opportunity_loss,
    }


def _posterior_expected_regret(inp, n_trial, rng):
    """Expected regret after updating with a trial of size n_trial."""
    mcid = inp.mcid
    prior_var = inp.se ** 2 + inp.tau2
    sigma2 = inp.within_study_var if inp.within_study_var else inp.se ** 2

    prior_precision = 1.0 / prior_var if prior_var > 1e-12 else 1e12
    trial_precision = n_trial / sigma2 if sigma2 > 1e-12 else 1e12
    post_precision = prior_precision + trial_precision

    n_inner = 2000
    # Draw true thetas from predictive
    theta_true = rng.normal(inp.theta, np.sqrt(prior_var), size=n_inner)
    trial_se = np.sqrt(sigma2 / n_trial) if n_trial > 0 else 1e6
    theta_trial = rng.normal(theta_true, trial_se)

    # Updated posterior mean
    theta_updated = (
        prior_precision * inp.theta + trial_precision * theta_trial
    ) / post_precision

    # Under updated decision
    nb_treat_updated = mcid - theta_updated
    nb_no_treat_updated = np.zeros_like(theta_updated)
    nb_best_updated = np.maximum(nb_treat_updated, nb_no_treat_updated)

    # Optimal decision under updated
    decision_treat = nb_treat_updated >= nb_no_treat_updated
    nb_chosen = np.where(decision_treat, nb_treat_updated, nb_no_treat_updated)

    # Regret = best - chosen (using true theta, not updated mean)
    nb_treat_true = mcid - theta_true
    nb_no_treat_true = np.zeros_like(theta_true)
    nb_best_true = np.maximum(nb_treat_true, nb_no_treat_true)
    nb_chosen_true = np.where(decision_treat, nb_treat_true, nb_no_treat_true)

    residual_regret = np.maximum(nb_best_true - nb_chosen_true, 0.0)
    return float(np.mean(residual_regret))
