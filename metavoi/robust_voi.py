"""Distributionally Robust Value of Information.

Wasserstein robust EVPI, Chebyshev worst-case bounds, and contamination
robustness analysis for meta-analytic decision problems.
"""

import numpy as np
from scipy import stats
from metavoi.posterior import predictive_distribution
from metavoi.evpi import compute_evpi


def compute_robust_voi(inp):
    """Compute distributionally robust VoI measures.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.

    Returns
    -------
    dict with keys:
        robust_evpi_curve : list of {eps, robust_evpi}
            Wasserstein robust EVPI for eps in [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0].
        breakeven_epsilon : float
            Epsilon where P(wrong) + eps * factor crosses 0.5.
        chebyshev_p_wrong : float
            Chebyshev worst-case P(theta < mcid).
        chebyshev_evpi : float
            EVPI computed from Chebyshev worst-case probability.
        contamination_curve : list of {eps, contamination_evpi}
            Contamination robust EVPI for eps in [0, 0.1, 0.2, 0.3, 0.4, 0.5].
        nominal_evpi : float
            Standard (non-robust) EVPI for reference.
    """
    draws = predictive_distribution(inp)
    nominal_evpi = compute_evpi(draws, inp.mcid)

    # --- Wasserstein robust EVPI ---
    # For binary NB with Lipschitz constant = 1,
    # robust_EVPI(eps) = EVPI + eps
    wasserstein_eps = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    robust_evpi_curve = []
    for eps in wasserstein_eps:
        robust_evpi_curve.append({
            "eps": eps,
            "robust_evpi": nominal_evpi + eps,
        })

    # --- Breakeven epsilon ---
    # P(wrong decision) under nominal model
    mean_theta = np.mean(draws)
    if mean_theta < inp.mcid:
        p_wrong = float(np.mean(draws > inp.mcid))
    else:
        p_wrong = float(np.mean(draws <= inp.mcid))

    # Breakeven: p_wrong + eps * factor = 0.5
    # Factor = 1 (Lipschitz bound on decision loss)
    factor = 1.0
    if factor > 0 and p_wrong < 0.5:
        breakeven_epsilon = (0.5 - p_wrong) / factor
    else:
        breakeven_epsilon = 0.0

    # --- Chebyshev worst-case ---
    # P(theta < mcid) <= sigma^2 / (sigma^2 + (theta - mcid)^2)
    # when theta > mcid (effect on wrong side of threshold)
    pred_var = inp.se ** 2 + inp.tau2
    gap = inp.theta - inp.mcid  # signed distance
    gap_sq = gap ** 2

    if gap_sq > 0:
        # One-sided Chebyshev (Cantelli's inequality)
        chebyshev_p_wrong = pred_var / (pred_var + gap_sq)
    else:
        # theta == mcid: worst case is 0.5
        chebyshev_p_wrong = 0.5

    # Chebyshev EVPI: use worst-case probability in EVPI formula
    # EVPI = p_wrong * |E[theta] - mcid| (for binary decision, simplified)
    # More precisely: EVPI = min(p_treat, p_no_treat) * |gap| approximately
    # Use the Chebyshev probability as the worst-case p_wrong
    chebyshev_evpi = chebyshev_p_wrong * abs(gap)

    # --- Contamination robust EVPI ---
    # (1 - eps) * EVPI + eps * max_EVPI
    # max_EVPI occurs when theta is exactly at mcid: EVPI_max ~ 0.5 * max(|range|)
    # For contamination model: adversary can place eps mass anywhere
    # max_EVPI for the contaminating distribution = abs(gap) + pred_sd * phi(0)/Phi(0)
    # Simplified: use a generous upper bound
    pred_sd = np.sqrt(pred_var)
    # Maximum per-decision EVPI: expected absolute deviation from mcid
    max_evpi = abs(gap) + pred_sd * float(stats.norm.pdf(0))  # ~0.4 * pred_sd

    contamination_eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    contamination_curve = []
    for eps in contamination_eps:
        cont_evpi = (1.0 - eps) * nominal_evpi + eps * max_evpi
        contamination_curve.append({
            "eps": eps,
            "contamination_evpi": cont_evpi,
        })

    return {
        "robust_evpi_curve": robust_evpi_curve,
        "breakeven_epsilon": breakeven_epsilon,
        "chebyshev_p_wrong": chebyshev_p_wrong,
        "chebyshev_evpi": chebyshev_evpi,
        "contamination_curve": contamination_curve,
        "nominal_evpi": nominal_evpi,
    }
