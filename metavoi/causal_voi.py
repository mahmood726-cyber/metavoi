"""Causal Value of Information.

Adjusts VoI for unmeasured confounding, instrumental variable strength,
and E-value sensitivity analysis.
"""

import numpy as np
from scipy import stats
from metavoi.posterior import predictive_distribution
from metavoi.evpi import compute_evpi


def compute_causal_voi(inp, bias_var_factor=0.1):
    """Compute causal-adjusted VoI measures.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    bias_var_factor : float
        Confounding bias variance as fraction of tau2 (default 0.1).

    Returns
    -------
    dict with keys:
        causal_evpi : float
            EVPI adjusted for confounding bias.
        standard_evpi : float
            Nominal (unadjusted) EVPI.
        confounding_component : float
            Difference between causal and standard EVPI.
        e_value : float
            E-value for the observed effect.
        iv_evpi_curve : list of {F, iv_evpi}
            EVPI under IV adjustment for F-statistics [1, 2, 5, 10].
        bias_sensitivity_curve : list of {bias_var, causal_evpi}
            EVPI for different confounding bias variances.
        is_causally_robust : bool
            Whether causal EVPI < 2x standard EVPI.
    """
    rng = np.random.default_rng(inp.seed + 80)

    # --- Standard EVPI ---
    draws = predictive_distribution(inp)
    standard_evpi = compute_evpi(draws, inp.mcid)

    # --- Causal EVPI with confounding bias ---
    # E_gamma[EVPI(theta + gamma)] where gamma ~ N(0, bias_var)
    bias_var = bias_var_factor * inp.tau2 if inp.tau2 > 0 else bias_var_factor * inp.se ** 2
    n_mc = 2000
    gamma_samples = rng.normal(0.0, np.sqrt(max(bias_var, 1e-12)), size=n_mc)

    # For each bias realization, compute EVPI with shifted draws
    causal_evpis = []
    for gamma in gamma_samples:
        shifted_draws = draws + gamma
        evpi_g = compute_evpi(shifted_draws, inp.mcid)
        causal_evpis.append(evpi_g)

    causal_evpi = float(np.mean(causal_evpis))
    confounding_component = causal_evpi - standard_evpi

    # --- E-value ---
    # For log-scale effects: E-value = exp(|theta|) + sqrt(exp(|theta|) * (exp(|theta|) - 1))
    abs_theta = abs(inp.theta)
    exp_abs = np.exp(abs_theta)
    e_value = exp_abs + np.sqrt(exp_abs * (exp_abs - 1.0))

    # --- IV-EVPI curve ---
    # Instrumental variable: inflated SE by 1/sqrt(F)
    f_values = [1, 2, 5, 10]
    iv_evpi_curve = []
    for f_stat in f_values:
        iv_se = inp.se / np.sqrt(f_stat)
        iv_pred_var = iv_se ** 2 + inp.tau2
        iv_draws = rng.normal(inp.theta, np.sqrt(iv_pred_var), size=len(draws))
        iv_evpi = compute_evpi(iv_draws, inp.mcid)
        iv_evpi_curve.append({
            "F": f_stat,
            "iv_evpi": iv_evpi,
        })

    # --- Bias sensitivity curve ---
    bias_var_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    bias_sensitivity_curve = []
    for bv_factor in bias_var_values:
        bv = bv_factor * inp.tau2 if inp.tau2 > 0 else bv_factor * inp.se ** 2
        if bv < 1e-15:
            # No bias: just standard EVPI
            bias_sensitivity_curve.append({
                "bias_var": bv_factor,
                "causal_evpi": standard_evpi,
            })
        else:
            gamma_s = rng.normal(0.0, np.sqrt(bv), size=500)
            evpis = [compute_evpi(draws + g, inp.mcid) for g in gamma_s]
            bias_sensitivity_curve.append({
                "bias_var": bv_factor,
                "causal_evpi": float(np.mean(evpis)),
            })

    # --- Is causally robust? ---
    is_causally_robust = causal_evpi < 2.0 * standard_evpi if standard_evpi > 0 else True

    return {
        "causal_evpi": causal_evpi,
        "standard_evpi": standard_evpi,
        "confounding_component": confounding_component,
        "e_value": e_value,
        "iv_evpi_curve": iv_evpi_curve,
        "bias_sensitivity_curve": bias_sensitivity_curve,
        "is_causally_robust": is_causally_robust,
    }
