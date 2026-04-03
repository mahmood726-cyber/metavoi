"""Spectral risk measures for decision under uncertainty.

Generalized coherent risk measures (CVaR, exponential, Wang transform)
applied to Value-of-Information analysis.
"""

import numpy as np
from scipy import stats as sp_stats


def _nb_samples_from_draws(draws, mcid):
    """Net benefit arrays for treat and no-treat."""
    nb_treat = mcid - draws
    nb_no_treat = np.zeros_like(draws)
    return nb_treat, nb_no_treat


def _risk_neutral_spectrum(p):
    """phi(p) = 1 for all p."""
    return np.ones_like(p)


def _cvar_spectrum(p, alpha=0.05):
    """CVaR_alpha: phi(p) = 1/alpha for p <= alpha, 0 otherwise."""
    return np.where(p <= alpha, 1.0 / alpha, 0.0)


def _exponential_spectrum(p, gamma=1.0):
    """Exponential: phi(p) = gamma*exp(gamma*p) / (exp(gamma)-1)."""
    denom = np.exp(gamma) - 1.0
    if abs(denom) < 1e-15:
        return np.ones_like(p)
    return gamma * np.exp(gamma * p) / denom


def _wang_spectrum(p, lam=0.5):
    """Wang transform: phi(p) = d/dp Phi(Phi^{-1}(p) + lambda).

    Derivative of Phi(Phi^{-1}(p) + lambda) w.r.t. p:
    = phi(Phi^{-1}(p) + lambda) / phi(Phi^{-1}(p))
    where phi is the standard normal pdf.
    """
    # Clip p to avoid infinities at 0 and 1
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    z = sp_stats.norm.ppf(p_clip)
    result = np.exp(sp_stats.norm.logpdf(z + lam) - sp_stats.norm.logpdf(z))
    return result


def _spectral_risk_measure(samples, spectrum_fn, **spectrum_kwargs):
    """Compute spectral risk measure rho_phi(X).

    rho = sum(phi(p_i) * VaR(p_i) * delta_p) for p_i in [0.01, ..., 0.99].
    """
    p_grid = np.arange(0.01, 1.0, 0.01)  # 99 points
    delta_p = p_grid[1] - p_grid[0]  # 0.01
    sorted_samples = np.sort(samples)
    # VaR at quantile p = quantile(X, p)
    var_values = np.quantile(sorted_samples, p_grid)
    phi_values = spectrum_fn(p_grid, **spectrum_kwargs)
    # Normalize spectrum to integrate to 1 (sum * delta_p = 1)
    phi_sum = np.sum(phi_values) * delta_p
    if phi_sum > 1e-15:
        phi_values = phi_values / phi_sum
    return float(np.sum(phi_values * var_values * delta_p))


def _risk_adjusted_evpi(draws, mcid, spectrum_fn, **spectrum_kwargs):
    """Compute risk-adjusted EVPI for a given spectrum.

    Risk-adjusted EVPI = E[max rho(NB|theta)] - max E[rho(NB)]

    Since we have MC samples of theta, we compute:
    - For each draw: max(NB_treat(draw), NB_no_treat(draw)) — this gives perfect info
    - rho of the perfect-info NB
    - rho of NB_treat and rho of NB_no_treat — take the max
    - EVPI = rho(perfect_NB) - max(rho(NB_treat), rho(NB_no_treat))
    """
    nb_treat, nb_no_treat = _nb_samples_from_draws(draws, mcid)
    nb_perfect = np.maximum(nb_treat, nb_no_treat)

    rho_perfect = _spectral_risk_measure(nb_perfect, spectrum_fn, **spectrum_kwargs)
    rho_treat = _spectral_risk_measure(nb_treat, spectrum_fn, **spectrum_kwargs)
    rho_no_treat = _spectral_risk_measure(nb_no_treat, spectrum_fn, **spectrum_kwargs)

    evpi = rho_perfect - max(rho_treat, rho_no_treat)
    return max(0.0, evpi)


def compute_spectral_risk(inp, n_mc=5000):
    """Compute spectral risk measures and risk-adjusted EVPI.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_mc : int
        Number of Monte Carlo samples.

    Returns
    -------
    dict with keys:
        spectral_evpi_by_spectrum, risk_aversion_curve, wang_curve,
        risk_neutral_evpi, cvar_05_evpi, optimal_spectrum
    """
    rng = np.random.default_rng(inp.seed + 90)
    pred_var = inp.se ** 2 + inp.tau2
    draws = rng.normal(inp.theta, np.sqrt(pred_var), size=n_mc)

    # Built-in spectra
    spectra = {
        "risk_neutral": (_risk_neutral_spectrum, {}),
        "cvar_05": (_cvar_spectrum, {"alpha": 0.05}),
        "exponential_1": (_exponential_spectrum, {"gamma": 1.0}),
        "wang_05": (_wang_spectrum, {"lam": 0.5}),
    }

    spectral_evpi = {}
    for name, (fn, kwargs) in spectra.items():
        spectral_evpi[name] = _risk_adjusted_evpi(draws, inp.mcid, fn, **kwargs)

    risk_neutral_evpi = spectral_evpi["risk_neutral"]
    cvar_05_evpi = spectral_evpi["cvar_05"]

    # Risk aversion sensitivity (exponential spectrum)
    risk_aversion_curve = []
    for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        evpi_val = _risk_adjusted_evpi(draws, inp.mcid, _exponential_spectrum, gamma=gamma)
        risk_aversion_curve.append({"gamma": gamma, "evpi": evpi_val})

    # Wang transform sensitivity
    wang_curve = []
    for lam in [0.0, 0.5, 1.0, 1.5]:
        evpi_val = _risk_adjusted_evpi(draws, inp.mcid, _wang_spectrum, lam=lam)
        wang_curve.append({"lambda": lam, "evpi": evpi_val})

    # Find optimal spectrum (highest EVPI)
    optimal_spectrum = max(spectral_evpi, key=spectral_evpi.get)

    return {
        "spectral_evpi_by_spectrum": spectral_evpi,
        "risk_aversion_curve": risk_aversion_curve,
        "wang_curve": wang_curve,
        "risk_neutral_evpi": risk_neutral_evpi,
        "cvar_05_evpi": cvar_05_evpi,
        "optimal_spectrum": optimal_spectrum,
    }
