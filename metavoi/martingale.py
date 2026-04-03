"""Martingale Theory for Sequential Evidence Monitoring.

E-values and optional stopping theory for deciding when to stop
collecting evidence, based on Grünwald et al. (2020) safe testing.
"""

import numpy as np
from metavoi.posterior import predictive_distribution


def _compute_grow_lambda(theta_hat, mcid, sigma2):
    """GROW optimal lambda: maximizes expected growth rate.

    lambda* = (theta_hat - mcid) / sigma^2
    """
    return (theta_hat - mcid) / sigma2


def _compute_e_value(y, mcid, lam, sigma2):
    """Compute a single GROW e-value.

    e = exp(lambda * (y - mcid) - lambda^2 * sigma^2 / 2)
    """
    return float(np.exp(lam * (y - mcid) - lam ** 2 * sigma2 / 2.0))


def compute_martingale_voi(inp, n_sequential=20, n_per_trial=50, n_sims=200):
    """Compute martingale-based sequential evidence monitoring.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_sequential : int
        Number of sequential trials to simulate (default 20).
    n_per_trial : int
        Sample size per sequential trial (default 50).
    n_sims : int
        Number of simulation runs for expected stopping time (default 200).

    Returns
    -------
    dict with keys:
        expected_stopping_time: float
        e_process_curve: list of {t, mean_e, median_e}
        p_stop_by_t: list of {t, cumulative_prob}
        grow_lambda: float
        safe_evsi: float
        anytime_threshold: float (1/alpha = 20)
    """
    rng = np.random.default_rng(inp.seed + 50)
    alpha = 0.05
    anytime_threshold = 1.0 / alpha  # = 20

    # Variance for individual trial observations
    within_var = inp.within_study_var if inp.within_study_var is not None else inp.se ** 2
    # Each trial of n_per_trial produces a mean with variance:
    trial_var = within_var / n_per_trial + inp.tau2

    # GROW optimal lambda
    grow_lam = _compute_grow_lambda(inp.theta, inp.mcid, trial_var)

    # --- Simulate sequential evidence accumulation ---
    # e_processes[sim, t] = cumulative E_t for each simulation
    e_processes = np.ones((n_sims, n_sequential + 1))
    stopping_times = np.full(n_sims, n_sequential + 1, dtype=float)

    for sim in range(n_sims):
        e_t = 1.0
        for t in range(1, n_sequential + 1):
            # Generate trial mean from predictive distribution
            y_t = rng.normal(inp.theta, np.sqrt(trial_var))
            e_i = _compute_e_value(y_t, inp.mcid, grow_lam, trial_var)
            e_t *= e_i
            e_processes[sim, t] = e_t

            # Check stopping (Ville's inequality)
            if e_t >= anytime_threshold and stopping_times[sim] > n_sequential:
                stopping_times[sim] = float(t)

    # --- E-process curve (mean and median across simulations) ---
    e_process_curve = []
    for t in range(1, n_sequential + 1):
        vals = e_processes[:, t]
        e_process_curve.append({
            "t": t,
            "mean_e": float(np.mean(vals)),
            "median_e": float(np.median(vals)),
        })

    # --- Cumulative probability of stopping by time t ---
    p_stop_by_t = []
    for t in range(1, n_sequential + 1):
        cum_prob = float(np.mean(stopping_times <= t))
        p_stop_by_t.append({"t": t, "cumulative_prob": cum_prob})

    # --- Expected stopping time (among those that stopped) ---
    stopped = stopping_times[stopping_times <= n_sequential]
    if len(stopped) > 0:
        expected_stopping = float(np.mean(stopped))
    else:
        expected_stopping = float(n_sequential)  # Never stopped

    # --- Safe EVSI ---
    # Standard EVSI: expected value of sampling information
    # Multiply by probability of actually stopping = P(evidence sufficient)
    pred_var = inp.se ** 2 + inp.tau2
    prior_prec = 1.0 / pred_var
    trial_prec = 1.0 / trial_var

    # Standard EVSI for a single trial
    n_evsi_samples = 5000
    evsi_sum = 0.0
    for _ in range(n_evsi_samples):
        y_new = rng.normal(inp.theta, np.sqrt(trial_var))
        post_prec = prior_prec + trial_prec
        post_mean = (prior_prec * inp.theta + trial_prec * y_new) / post_prec
        post_sd = np.sqrt(1.0 / post_prec)

        post_draws = rng.normal(post_mean, post_sd, size=500)
        nb_treat_post = inp.mcid - post_draws
        evsi_this = float(np.mean(np.maximum(nb_treat_post, 0.0)) -
                          max(np.mean(nb_treat_post), 0.0))
        evsi_sum += max(0.0, evsi_this)

    standard_evsi = evsi_sum / n_evsi_samples

    # Prior EVPI baseline
    all_draws = predictive_distribution(inp)
    nb_prior = inp.mcid - all_draws
    evpi_baseline = float(np.mean(np.maximum(nb_prior, 0.0)) -
                          max(np.mean(nb_prior), 0.0))

    standard_evsi_net = max(0.0, standard_evsi - max(0.0, evpi_baseline - standard_evsi))

    # P(stop) = fraction of simulations that stopped
    p_stop = float(np.mean(stopping_times <= n_sequential))
    safe_evsi = standard_evsi * max(p_stop, 0.01)  # Floor at 1%

    return {
        "expected_stopping_time": expected_stopping,
        "e_process_curve": e_process_curve,
        "p_stop_by_t": p_stop_by_t,
        "grow_lambda": float(grow_lam),
        "safe_evsi": float(safe_evsi),
        "anytime_threshold": float(anytime_threshold),
    }
