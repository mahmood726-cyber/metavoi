"""Sample complexity theory for trial design.

Learning-theoretic perspective on how many patients are needed,
including PAC, minimax, Bayesian, and information-theoretic bounds.
"""

import numpy as np
from scipy import stats as sp_stats


def compute_sample_complexity(inp, n_adaptive_runs=200):
    """Compute sample complexity bounds from multiple theoretical perspectives.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_adaptive_runs : int
        Number of MC runs for adaptive sample size (default 200).

    Returns
    -------
    dict with keys:
        pac_n, minimax_n, bayesian_n, fano_lower_bound_n,
        adaptive_expected_n, comparison, effective_sigma
    """
    rng = np.random.default_rng(inp.seed + 100)

    # Effective sigma (total uncertainty)
    effective_sigma = np.sqrt(inp.se ** 2 + inp.tau2)
    epsilon = abs(inp.mcid) / 2.0 if abs(inp.mcid) > 0 else 0.1
    delta = 0.05

    # --- PAC-style bound ---
    # n >= (z_{1-delta/2} * sigma / epsilon)^2
    z_val = sp_stats.norm.ppf(1.0 - delta / 2.0)
    pac_n = int(np.ceil((z_val * effective_sigma / epsilon) ** 2))
    pac_n = max(1, pac_n)

    # --- Minimax sample complexity ---
    # R*(n) = sigma^2 / n; need R*(n) < tolerance = MCID^2 / 4
    tolerance = inp.mcid ** 2 / 4.0 if inp.mcid != 0 else 0.01
    if tolerance > 0:
        minimax_n = int(np.ceil(effective_sigma ** 2 / tolerance))
    else:
        minimax_n = pac_n
    minimax_n = max(1, minimax_n)

    # --- Bayesian sample complexity ---
    # n such that P(|theta_post - theta_true| > epsilon) < delta
    # Using posterior predictive: n >= (sigma_pred * z / epsilon)^2
    # sigma_pred is the predictive SD which shrinks as 1/sqrt(n)
    # For n new observations with known variance sigma^2/n:
    # posterior variance = 1/(1/prior_var + n/obs_var)
    # We solve for n such that sqrt(post_var) * z < epsilon
    # prior_var = effective_sigma^2
    # obs_var_per = effective_sigma^2 (per observation)
    # post_var = 1/(1/prior_var + n/obs_var_per) = prior_var * obs_var_per / (obs_var_per + n*prior_var)
    # Need sqrt(post_var) * z < epsilon
    # post_var < (epsilon/z)^2
    # prior_var * obs_var_per / (obs_var_per + n*prior_var) < (epsilon/z)^2
    # Solve for n:
    prior_var = effective_sigma ** 2
    obs_var_per = effective_sigma ** 2
    target_var = (epsilon / z_val) ** 2
    if target_var > 0 and prior_var > target_var:
        # n > (prior_var * obs_var_per / target_var - obs_var_per) / prior_var
        bayesian_n = int(np.ceil(
            (prior_var * obs_var_per / target_var - obs_var_per) / prior_var
        ))
    else:
        bayesian_n = 1
    bayesian_n = max(1, bayesian_n)

    # --- Information-theoretic lower bound (Fano-style) ---
    # n >= 2*sigma^2 / (epsilon^2) * log(1/delta)
    fano_n = int(np.ceil(
        2.0 * effective_sigma ** 2 / (epsilon ** 2) * np.log(1.0 / delta)
    ))
    fano_n = max(1, fano_n)

    # --- Adaptive sample size ---
    # Start with n_0=50, double until P(wrong decision) < 0.05
    # Wrong decision: sign of theta w.r.t. MCID
    adaptive_ns = []
    for _ in range(n_adaptive_runs):
        # True theta drawn from predictive
        theta_true = rng.normal(inp.theta, effective_sigma)
        n_current = 50
        max_n = 100000
        found = False
        while n_current <= max_n:
            # Simulate n_current observations
            obs_sigma = effective_sigma / np.sqrt(n_current)
            obs_mean = rng.normal(theta_true, obs_sigma)
            # Posterior: combine prior (theta, effective_sigma^2) with data
            post_var = 1.0 / (1.0 / prior_var + n_current / obs_var_per)
            post_mean = post_var * (inp.theta / prior_var + n_current * obs_mean / obs_var_per)
            post_sd = np.sqrt(post_var)
            # P(wrong decision): if post_mean < mcid, decision = treat
            # P(wrong) = P(theta > mcid | data)
            if post_mean < inp.mcid:
                p_wrong = 1.0 - sp_stats.norm.cdf(inp.mcid, loc=post_mean, scale=post_sd)
            else:
                p_wrong = sp_stats.norm.cdf(inp.mcid, loc=post_mean, scale=post_sd)
            if p_wrong < 0.05:
                adaptive_ns.append(n_current)
                found = True
                break
            n_current *= 2
        if not found:
            adaptive_ns.append(max_n)

    adaptive_expected_n = int(np.ceil(np.mean(adaptive_ns)))

    comparison = {
        "pac": pac_n,
        "minimax": minimax_n,
        "bayesian": bayesian_n,
        "fano": fano_n,
        "adaptive": adaptive_expected_n,
    }

    return {
        "pac_n": pac_n,
        "minimax_n": minimax_n,
        "bayesian_n": bayesian_n,
        "fano_lower_bound_n": fano_n,
        "adaptive_expected_n": adaptive_expected_n,
        "comparison": comparison,
        "effective_sigma": float(effective_sigma),
    }
