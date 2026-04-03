"""Concentration inequalities for VoI bounds.

Non-asymptotic bounds on EVPI estimation error using Hoeffding, Bernstein,
McDiarmid, and sub-Gaussian concentration inequalities.
"""

import numpy as np
from scipy import stats as sp_stats

from .posterior import predictive_distribution


def _nb_samples(draws, mcid):
    """Net benefit samples for treat vs no-treat decisions."""
    nb_treat = mcid - draws
    nb_no_treat = np.zeros_like(draws)
    nb_max = np.maximum(nb_treat, nb_no_treat)
    return nb_treat, nb_no_treat, nb_max


def compute_concentration_bounds(inp, n_mc=5000):
    """Compute concentration inequality bounds for EVPI estimation.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_mc : int
        Number of Monte Carlo samples (default 5000).

    Returns
    -------
    dict with keys:
        hoeffding_bound, bernstein_bound, mcdiarmid_bound,
        sub_gaussian_sigma, finite_sample_ci, required_n_for_epsilon_001,
        range_nb
    """
    rng = np.random.default_rng(inp.seed + 80)

    # Draw from predictive distribution using our own rng
    pred_var = inp.se ** 2 + inp.tau2
    draws = rng.normal(inp.theta, np.sqrt(pred_var), size=n_mc)

    nb_treat, nb_no_treat, nb_max = _nb_samples(draws, inp.mcid)

    # Current best decision
    current_best = max(np.mean(nb_treat), np.mean(nb_no_treat))

    # Per-sample EVPI contributions: max(NB) - current_best
    evpi_samples = nb_max - current_best

    # Range of NB max samples
    range_nb = float(np.max(nb_max) - np.min(nb_max))
    # Guard against zero range
    if range_nb < 1e-15:
        range_nb = 1e-15

    # Empirical variance of nb_max samples
    sigma2 = float(np.var(nb_max, ddof=1))
    sigma = np.sqrt(sigma2) if sigma2 > 0 else 1e-15

    # --- Hoeffding bound ---
    # P(|EVPI_hat - EVPI| > t) <= 2*exp(-2*n*t^2 / range^2)
    # Solve for t at confidence 0.95: delta=0.05
    delta = 0.05
    hoeffding_bound = range_nb * np.sqrt(np.log(2.0 / delta) / (2.0 * n_mc))

    # --- Bernstein bound ---
    # P(|X_bar - mu| > t) <= 2*exp(-n*t^2 / (2*sigma^2 + 2*b*t/3))
    # Solve: n*t^2 / (2*sigma^2 + 2*b*t/3) = log(2/delta)
    # This is a quadratic in t: n*t^2 - log(2/delta)*(2*b/3)*t - log(2/delta)*2*sigma^2 = 0
    b = range_nb
    log_term = np.log(2.0 / delta)
    # a_coeff * t^2 - b_coeff * t - c_coeff = 0
    a_coeff = n_mc
    b_coeff = log_term * 2.0 * b / 3.0
    c_coeff = log_term * 2.0 * sigma2
    discriminant = b_coeff ** 2 + 4.0 * a_coeff * c_coeff
    bernstein_bound = (b_coeff + np.sqrt(discriminant)) / (2.0 * a_coeff)

    # --- McDiarmid bound ---
    # EVPI is a function of k study estimates; bounded difference c_i
    # c_i = max change from perturbing study i
    # For simplicity: each study contributes se/sqrt(k) to the mean,
    # so perturbing study i changes theta by at most 2*se (within +/- 2 SE range).
    # The NB change is bounded by that perturbation.
    k = inp.k
    # Bounded difference: changing one study changes the pooled mean by at most
    # ~ 1/k of the study-level range. Conservative: c_i = range_nb / k
    c_i = range_nb / k if k > 0 else range_nb
    sum_ci_sq = k * c_i ** 2
    # P(|f - E[f]| > t) <= 2*exp(-2*t^2 / sum(c_i^2))
    mcdiarmid_bound = np.sqrt(sum_ci_sq * np.log(2.0 / delta) / 2.0)

    # --- Sub-Gaussian parameter ---
    # sigma_SG = max over lambda>0 of: (1/lambda) * sqrt(2*log(mean(exp(lambda*(X-mean(X))))))
    centered = nb_max - np.mean(nb_max)
    lambda_grid = [0.1, 0.5, 1.0, 2.0]
    sg_estimates = []
    for lam in lambda_grid:
        # Stabilize exp computation
        exponents = lam * centered
        max_exp = np.max(exponents)
        log_mgf = max_exp + np.log(np.mean(np.exp(exponents - max_exp)))
        if log_mgf > 0:
            sg_est = (1.0 / lam) * np.sqrt(2.0 * log_mgf)
            sg_estimates.append(sg_est)
    sub_gaussian_sigma = float(max(sg_estimates)) if sg_estimates else sigma

    # --- Finite-sample confidence interval ---
    evpi_hat = float(np.mean(nb_max) - current_best)
    evpi_hat = max(0.0, evpi_hat)
    ci_lo = evpi_hat - hoeffding_bound
    ci_hi = evpi_hat + hoeffding_bound
    finite_sample_ci = (float(ci_lo), float(ci_hi))

    # --- Required n for precision epsilon=0.01 ---
    epsilon = 0.01
    required_n = int(np.ceil(range_nb ** 2 * np.log(2.0 / delta) / (2.0 * epsilon ** 2)))

    return {
        "hoeffding_bound": float(hoeffding_bound),
        "bernstein_bound": float(bernstein_bound),
        "mcdiarmid_bound": float(mcdiarmid_bound),
        "sub_gaussian_sigma": float(sub_gaussian_sigma),
        "finite_sample_ci": finite_sample_ci,
        "required_n_for_epsilon_001": required_n,
        "range_nb": float(range_nb),
    }
