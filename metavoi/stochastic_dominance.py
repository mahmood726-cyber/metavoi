"""Stochastic Dominance Analysis for VoI.

Tests whether one treatment option dominates another across the entire
net-benefit distribution, beyond simple expected-value comparison.

- First-order stochastic dominance (FSD): F_treat(x) <= F_no_treat(x)
  for all x (universally preferred by all rational agents).
- Second-order stochastic dominance (SSD): integrated CDF of treat <=
  that of no-treat for all x (preferred by all risk-averse agents).
- Lorenz curve + Gini coefficient of NB distributions.
- Risk measures: VaR and CVaR (tail risk quantification).
"""

import numpy as np
from metavoi.posterior import predictive_distribution


def compute_stochastic_dominance(inp, n_samples=5000, n_grid=500, alpha=0.05):
    """Full stochastic dominance analysis: treat vs. no-treat.

    Parameters
    ----------
    inp : VoIInput
        Standard MetaVoI input.
    n_samples : int
        Monte Carlo samples for NB distributions.
    n_grid : int
        Grid points for CDF comparison.
    alpha : float
        Tail probability for VaR/CVaR.

    Returns
    -------
    dict with keys:
        fsd_treat_dominates  -- bool: treat FSD-dominates no-treat
        fsd_ratio            -- float [0,1]: fraction of grid where FSD holds
        ssd_treat_dominates  -- bool: treat SSD-dominates no-treat
        ssd_ratio            -- float [0,1]: fraction of grid where SSD holds
        gini_treat           -- Gini coefficient of treat NB
        gini_no_treat        -- Gini coefficient of no-treat NB
        var_treat            -- Value at Risk (alpha-quantile) for treat
        var_no_treat         -- Value at Risk for no-treat
        cvar_treat           -- Conditional VaR for treat
        cvar_no_treat        -- Conditional VaR for no-treat
        recommendation       -- str: decision recommendation
    """
    rng = np.random.default_rng(inp.seed + 30)

    # --- Generate NB distributions ---
    pred_var = inp.se ** 2 + inp.tau2
    theta_draws = rng.normal(inp.theta, np.sqrt(max(pred_var, 1e-16)),
                             size=n_samples)
    nb_treat = inp.mcid - theta_draws   # NB of treating
    nb_no_treat = np.zeros(n_samples)   # NB of not treating = 0

    # --- Evaluation grid ---
    all_vals = np.concatenate([nb_treat, nb_no_treat])
    x_min, x_max = float(np.min(all_vals)), float(np.max(all_vals))
    margin = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    grid = np.linspace(x_min - margin, x_max + margin, n_grid)

    # --- Empirical CDFs ---
    nb_treat_sorted = np.sort(nb_treat)
    nb_no_treat_sorted = np.sort(nb_no_treat)

    cdf_treat = np.searchsorted(nb_treat_sorted, grid, side="right") / n_samples
    cdf_no_treat = np.searchsorted(nb_no_treat_sorted, grid, side="right") / n_samples

    # --- First-order stochastic dominance ---
    # Treat FSD-dominates no-treat if F_treat(x) <= F_no_treat(x) for all x
    # (treat has less probability mass in the lower tails)
    fsd_holds = cdf_treat <= cdf_no_treat + 1e-12  # small tolerance
    fsd_ratio = float(np.mean(fsd_holds))
    fsd_treat_dominates = bool(np.all(fsd_holds))

    # --- Second-order stochastic dominance ---
    # Integrated CDF: I_F(x) = integral_{-inf}^{x} F(t) dt
    # Numerically: cumulative trapezoidal integral of the CDF
    dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
    int_cdf_treat = np.cumsum(cdf_treat) * dx
    int_cdf_no_treat = np.cumsum(cdf_no_treat) * dx

    ssd_holds = int_cdf_treat <= int_cdf_no_treat + 1e-12
    ssd_ratio = float(np.mean(ssd_holds))
    ssd_treat_dominates = bool(np.all(ssd_holds))

    # --- Lorenz curve and Gini ---
    gini_treat = _gini_coefficient(nb_treat)
    gini_no_treat = _gini_coefficient(nb_no_treat)

    # --- Risk measures: VaR and CVaR ---
    var_treat = float(np.percentile(nb_treat, 100 * alpha))
    var_no_treat = float(np.percentile(nb_no_treat, 100 * alpha))

    cvar_treat = _cvar(nb_treat, alpha)
    cvar_no_treat = _cvar(nb_no_treat, alpha)

    # --- Decision recommendation ---
    if fsd_treat_dominates:
        recommendation = "Strong: treat FSD-dominates no-treat"
    elif ssd_treat_dominates:
        recommendation = "Moderate: treat SSD-dominates (risk-averse preference)"
    elif np.mean(nb_treat) > np.mean(nb_no_treat):
        recommendation = "Weak: treat has higher expected NB but no dominance"
    else:
        recommendation = "No dominance: no-treat preferred on expected value"

    return {
        "fsd_treat_dominates": fsd_treat_dominates,
        "fsd_ratio": fsd_ratio,
        "ssd_treat_dominates": ssd_treat_dominates,
        "ssd_ratio": ssd_ratio,
        "gini_treat": gini_treat,
        "gini_no_treat": gini_no_treat,
        "var_treat": var_treat,
        "var_no_treat": var_no_treat,
        "cvar_treat": cvar_treat,
        "cvar_no_treat": cvar_no_treat,
        "recommendation": recommendation,
    }


def _gini_coefficient(values):
    """Compute the Gini coefficient from a sample.

    Gini = (2 * sum(i * y_sorted[i]) / (n * sum(y))) - (n+1)/n

    Handles negative values by shifting to non-negative range.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0

    # Shift to non-negative if needed
    shift = 0.0
    if np.min(v) < 0:
        shift = -np.min(v) + 1e-10
    v_shifted = v + shift

    total = np.sum(v_shifted)
    if total < 1e-15:
        return 0.0

    sorted_v = np.sort(v_shifted)
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * sorted_v) / (n * total)) - (n + 1.0) / n
    return float(max(0.0, gini))


def _cvar(values, alpha):
    """Conditional Value at Risk: expected value in the worst alpha fraction."""
    v = np.sort(values)
    n = len(v)
    cutoff = max(1, int(np.floor(n * alpha)))
    return float(np.mean(v[:cutoff]))
