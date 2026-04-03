"""Bayesian Bootstrap for VoI Uncertainty.

Quantifies uncertainty in VoI estimates via Rubin (1981) Bayesian bootstrap:
- Dirichlet-weighted resampling of MC draws
- Credible intervals for EVPI and EVSI
- Coefficient of variation for reliability assessment
- Probability that VoI justifies a trial at given cost
"""

import numpy as np

from metavoi.evsi import compute_evsi
from metavoi.posterior import predictive_distribution, discount_factor


def compute_bayesian_bootstrap(inp, n_boot=200, n_mc=5000, n_trial_evsi=500):
    """Bayesian bootstrap for EVPI and EVSI uncertainty.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis VoI input parameters.
    n_boot : int
        Number of bootstrap replications.
    n_mc : int
        Number of MC samples per bootstrap draw.
    n_trial_evsi : int
        Trial size N for the EVSI bootstrap.

    Returns
    -------
    dict with keys:
        evpi_mean : float
            Mean EVPI across bootstrap samples.
        evpi_ci : tuple[float, float]
            95% credible interval for EVPI.
        evpi_cv : float
            Coefficient of variation of EVPI bootstrap distribution.
        evpi_distribution : list[float]
            All B bootstrap EVPI values.
        evsi_mean : float
            Mean EVSI at n_trial_evsi.
        evsi_ci : tuple[float, float]
            95% credible interval for EVSI.
        p_justified : float
            P(EVPI_pop > trial cost) from bootstrap.
    """
    rng = np.random.default_rng(inp.seed + 20)

    # Generate base MC draws for EVPI
    draws = predictive_distribution(inp)
    if len(draws) > n_mc:
        draws = draws[:n_mc]
    n_samples = len(draws)

    mcid = inp.mcid
    df_sum = discount_factor(inp.discount_rate, inp.horizon_years)
    trial_cost = inp.cost_per_patient * n_trial_evsi

    # --- EVPI Bayesian bootstrap ---
    evpi_boot = []
    for _ in range(n_boot):
        weights = rng.dirichlet(np.ones(n_samples))
        evpi_b = _weighted_evpi(draws, weights, mcid)
        evpi_boot.append(evpi_b)

    evpi_boot = np.array(evpi_boot)
    evpi_mean = float(np.mean(evpi_boot))
    evpi_ci = (float(np.percentile(evpi_boot, 2.5)),
               float(np.percentile(evpi_boot, 97.5)))
    evpi_std = float(np.std(evpi_boot))
    evpi_cv = evpi_std / evpi_mean if evpi_mean > 1e-15 else 0.0

    # --- EVSI Bayesian bootstrap ---
    evsi_boot = _bootstrap_evsi(inp, rng, n_boot=n_boot, n_trial=n_trial_evsi,
                                n_sim=min(n_mc, 2000))
    evsi_mean = float(np.mean(evsi_boot))
    evsi_ci = (float(np.percentile(evsi_boot, 2.5)),
               float(np.percentile(evsi_boot, 97.5)))

    # --- P(VoI justifies trial) ---
    evpi_pop_boot = evpi_boot * inp.population * df_sum
    p_justified = float(np.mean(evpi_pop_boot > trial_cost))

    return {
        "evpi_mean": evpi_mean,
        "evpi_ci": evpi_ci,
        "evpi_cv": evpi_cv,
        "evpi_distribution": [float(v) for v in evpi_boot],
        "evsi_mean": evsi_mean,
        "evsi_ci": evsi_ci,
        "p_justified": p_justified,
    }


def _weighted_evpi(draws, weights, mcid):
    """Compute EVPI with Dirichlet weights.

    EVPI_w = sum(w_i * max(mcid - theta_i, 0)) - max(sum(w_i * (mcid - theta_i)), 0)
    """
    nb_treat = mcid - draws
    nb_no_treat = np.zeros_like(draws)

    # Weighted expected perfect information
    perfect = np.maximum(nb_treat, nb_no_treat)
    e_perfect = np.sum(weights * perfect)

    # Weighted current best
    e_treat = np.sum(weights * nb_treat)
    e_no_treat = 0.0
    current_best = max(e_treat, e_no_treat)

    evpi = e_perfect - current_best
    return max(0.0, evpi)


def _bootstrap_evsi(inp, rng, n_boot, n_trial, n_sim):
    """Bootstrap EVSI by varying seed across replications.

    Each bootstrap replication uses a different seed offset to generate
    independent EVSI estimates, providing a distribution of EVSI values.
    """
    from dataclasses import replace

    evsi_values = []
    for b in range(n_boot):
        inp_b = replace(inp, seed=inp.seed + 200 + b)
        evsi_b = compute_evsi(inp_b, n_trial=n_trial, n_sim=n_sim)
        evsi_values.append(evsi_b)

    return np.array(evsi_values)
