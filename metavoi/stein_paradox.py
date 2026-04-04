"""Stein's Paradox and Shrinkage Estimation for VoI.

James-Stein estimator shrinks individual study contributions toward the grand
mean, yielding strictly lower MSE than the MLE for k >= 3 dimensions.  This
module computes the shrinkage estimates, demonstrates MLE inadmissibility, and
recomputes EVPI using the shrunk posterior.
"""

import numpy as np
from scipy import stats

from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def compute_stein_paradox(inp, n_grid=50):
    """Compute James-Stein shrinkage analysis and shrunk-VoI.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_grid : int
        Number of grid points for inadmissibility MSE curves (default 50).

    Returns
    -------
    dict with keys:
        js_estimate: float — James-Stein estimator for the pooled effect.
        positive_part_estimate: float — Positive-part Stein estimator.
        shrinkage_factor: float — Raw shrinkage multiplier (1 - (k-2)*s2/S).
        stein_risk_ratio: float — MSE(JS) / MSE(MLE), always < 1 for k >= 3.
        mle_mse_curve: list[float] — MLE MSE at each grid point.
        js_mse_curve: list[float] — JS MSE at each grid point.
        theta_grid: list[float] — Grid of true-theta offsets for MSE curves.
        shrunk_evpi: float — EVPI computed from JS-shrunk posterior.
        standard_evpi: float — Standard EVPI (MLE-based).
        evpi_reduction_pct: float — Percent reduction from shrinkage.
    """
    rng = np.random.default_rng(inp.seed + 100)
    k = inp.k
    sigma2 = inp.se ** 2

    # ------------------------------------------------------------------
    # Generate k pseudo-study estimates around theta with variance sigma2
    # ------------------------------------------------------------------
    theta_i = rng.normal(inp.theta, inp.se, size=k)
    theta_bar = np.mean(theta_i)

    # Sum of squared deviations from grand mean
    S = float(np.sum((theta_i - theta_bar) ** 2))

    # ------------------------------------------------------------------
    # James-Stein shrinkage factor
    # ------------------------------------------------------------------
    if S > 0 and k >= 3:
        raw_factor = 1.0 - (k - 2) * sigma2 / S
    else:
        raw_factor = 1.0  # No shrinkage for k < 3

    shrinkage_factor = float(raw_factor)
    positive_factor = max(0.0, shrinkage_factor)

    # JS estimates
    js_estimates = theta_bar + shrinkage_factor * (theta_i - theta_bar)
    js_estimate = float(np.mean(js_estimates))

    positive_part_estimates = theta_bar + positive_factor * (theta_i - theta_bar)
    positive_part_estimate = float(np.mean(positive_part_estimates))

    # ------------------------------------------------------------------
    # Risk ratio: analytic MSE comparison
    # For N(mu, sigma2*I_k), JS has risk = k*sigma2 - (k-2)^2 * sigma4 / E[S]
    # MLE risk = k * sigma2
    # E[S] = (k-1)*sigma2 + ||mu - mu_bar||^2 (centred), approximate with S
    # ------------------------------------------------------------------
    mle_risk = k * sigma2
    if k >= 3 and S > 0:
        js_risk = k * sigma2 - ((k - 2) ** 2) * (sigma2 ** 2) / max(S, 1e-30)
        js_risk = max(js_risk, 0.0)  # Can't be negative
    else:
        js_risk = mle_risk
    stein_risk_ratio = js_risk / mle_risk if mle_risk > 0 else 1.0

    # ------------------------------------------------------------------
    # Inadmissibility demonstration: MSE curves over a grid of true theta
    # ------------------------------------------------------------------
    n_mc_risk = 2000
    half_range = max(3.0 * inp.se, 0.5)
    theta_grid = np.linspace(inp.theta - half_range, inp.theta + half_range, n_grid)

    mle_mse_curve = []
    js_mse_curve = []

    for mu_true in theta_grid:
        # Generate k observations from N(mu_true, sigma2)
        obs = rng.normal(mu_true, inp.se, size=(n_mc_risk, k))
        obs_bar = obs.mean(axis=1, keepdims=True)

        # MLE: just the observations themselves
        mle_mse = float(np.mean((obs - mu_true) ** 2))

        # JS shrinkage per MC replicate
        dev = obs - obs_bar
        S_mc = np.sum(dev ** 2, axis=1)  # shape (n_mc_risk,)
        # Avoid division by zero
        S_mc_safe = np.where(S_mc > 0, S_mc, 1.0)
        factor = 1.0 - (k - 2) * sigma2 / S_mc_safe
        factor = np.where(S_mc > 0, factor, 1.0)
        # Positive-part
        factor = np.maximum(factor, 0.0)

        js_obs = obs_bar + factor[:, None] * dev
        js_mse = float(np.mean((js_obs - mu_true) ** 2))

        mle_mse_curve.append(mle_mse)
        js_mse_curve.append(js_mse)

    # ------------------------------------------------------------------
    # EVPI with shrinkage: use JS-shrunk posterior
    # ------------------------------------------------------------------
    # Standard EVPI
    draws_std = predictive_distribution(inp)
    standard_evpi = compute_evpi(draws_std, inp.mcid)

    # Shrunk posterior: shift theta toward shrunk estimate, reduce variance
    se_js = inp.se * abs(positive_factor) if positive_factor > 0 else inp.se * 0.1
    shrunk_var = se_js ** 2 + inp.tau2
    draws_shrunk = rng.normal(positive_part_estimate, np.sqrt(shrunk_var), size=inp.n_sim)
    shrunk_evpi = compute_evpi(draws_shrunk, inp.mcid)

    # Reduction
    if standard_evpi > 0:
        evpi_reduction_pct = 100.0 * (standard_evpi - shrunk_evpi) / standard_evpi
    else:
        evpi_reduction_pct = 0.0

    return {
        "js_estimate": js_estimate,
        "positive_part_estimate": positive_part_estimate,
        "shrinkage_factor": shrinkage_factor,
        "stein_risk_ratio": stein_risk_ratio,
        "mle_mse_curve": [float(v) for v in mle_mse_curve],
        "js_mse_curve": [float(v) for v in js_mse_curve],
        "theta_grid": [float(v) for v in theta_grid],
        "shrunk_evpi": shrunk_evpi,
        "standard_evpi": standard_evpi,
        "evpi_reduction_pct": evpi_reduction_pct,
    }
