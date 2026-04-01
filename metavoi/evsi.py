import numpy as np
from metavoi.models import EVSIPoint

DEFAULT_N_VALUES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]


def compute_evsi(inp, n_trial, n_sim=5000):
    """EVSI for a hypothetical trial of size n_trial.

    Bayesian preposterior analysis:
    1. Draw true theta from prior (posterior of current MA)
    2. Simulate trial data: theta_new ~ N(theta_true, sigma2/n_trial)
    3. Compute updated posterior mean (precision-weighted)
    4. For each updated posterior, find optimal decision
    5. EVSI = E[NB under updated decision] - NB under current decision
    """
    rng = np.random.default_rng(inp.seed + 100 + n_trial)
    mcid = inp.mcid
    sigma2 = inp.within_study_var if inp.within_study_var else inp.se ** 2

    # Current decision
    prior_var = inp.se ** 2 + inp.tau2
    nb_current_treat = mcid - inp.theta
    nb_current_best = max(nb_current_treat, 0.0)

    # Prior precision
    prior_precision = 1.0 / prior_var if prior_var > 1e-12 else 1e12
    # Trial precision
    trial_precision = n_trial / sigma2 if sigma2 > 1e-12 else 1e12
    # Updated precision
    post_precision = prior_precision + trial_precision

    # Simulate
    theta_true = rng.normal(inp.theta, np.sqrt(prior_var), size=n_sim)
    trial_se = np.sqrt(sigma2 / n_trial) if n_trial > 0 else 1e6
    theta_trial = rng.normal(theta_true, trial_se)
    theta_updated = (prior_precision * inp.theta + trial_precision * theta_trial) / post_precision

    # Optimal decision under updated posterior
    nb_treat_updated = mcid - theta_updated
    perfect_updated = np.maximum(nb_treat_updated, 0.0)

    evsi = float(np.mean(perfect_updated) - nb_current_best)
    return max(0.0, evsi)


def compute_evsi_curve(inp, n_values=None):
    """Compute EVSI for multiple trial sizes."""
    if n_values is None:
        n_values = DEFAULT_N_VALUES

    from metavoi.posterior import discount_factor
    df = discount_factor(inp.discount_rate, inp.horizon_years)

    curve = []
    for n in n_values:
        evsi = compute_evsi(inp, n_trial=n)
        evsi_pop = evsi * inp.population * df
        cost = inp.cost_per_patient * n
        nb = evsi_pop - cost
        curve.append(EVSIPoint(n=n, evsi=evsi, evsi_pop=evsi_pop, cost=cost, net_benefit=nb))

    return curve
