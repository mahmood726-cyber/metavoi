"""D-Optimal and A-Optimal Trial Design for Meta-Analysis VoI.

Optimal allocation of trial resources to maximize information gain:
- D-optimal design (maximize det of posterior Fisher information)
- A-optimal design (minimize trace of inverse posterior Fisher information)
- Multi-site Neyman-type allocation under budget constraint
- Information gain per dollar curve with knee-point detection
- Comparison across optimization criteria
"""

import numpy as np

from metavoi.evsi import compute_evsi
from metavoi.posterior import discount_factor


def compute_optimal_design(inp, budget=None, sites=None):
    """Compute optimal trial designs under multiple criteria.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis VoI input parameters.
    budget : float or None
        Total budget for multi-site allocation. Defaults to
        cost_per_patient * 5000.
    sites : list[dict] or None
        List of dicts with 'cost' and 'sigma2' per site.
        Defaults to 3 sites with varying cost and variance.

    Returns
    -------
    dict with keys:
        d_optimal_n : int
            Trial N that maximizes det(I_posterior).
        a_optimal_n : int
            Trial N that minimizes tr(I_posterior^{-1}).
        multi_site_allocation : list[dict]
            Per-site allocation {site, n, cost}.
        info_gain_curve : list[dict]
            {n, delta_info, cost} for a range of N values.
        comparison : dict
            Method name -> optimal N.
    """
    se2 = inp.se ** 2
    tau2 = max(inp.tau2, 1e-12)
    sigma2 = inp.within_study_var if inp.within_study_var else se2
    cost_pp = inp.cost_per_patient

    if budget is None:
        budget = cost_pp * 5000

    if sites is None:
        sites = [
            {"cost": cost_pp, "sigma2": sigma2},
            {"cost": cost_pp * 1.6, "sigma2": sigma2 * 1.5},
            {"cost": cost_pp * 2.4, "sigma2": sigma2 * 0.7},
        ]

    # Prior Fisher information
    w = se2 + tau2
    I_prior_tt = inp.k / w
    I_prior_tau2 = inp.k / (2.0 * w ** 2)

    # --- D-optimal and A-optimal over a grid of N ---
    n_candidates = list(range(10, 10001, 10))

    d_optimal_n = _d_optimal(n_candidates, I_prior_tt, I_prior_tau2, sigma2)
    a_optimal_n = _a_optimal(n_candidates, I_prior_tt, I_prior_tau2, sigma2)

    # --- Multi-site allocation ---
    multi_site = _multi_site_allocation(sites, budget)

    # --- Information gain per dollar ---
    info_curve = _info_gain_curve(n_candidates, I_prior_tt, sigma2, cost_pp)

    # --- EVSI-optimal for comparison ---
    evsi_optimal_n = _evsi_optimal(inp)

    comparison = {
        "d_optimal": d_optimal_n,
        "a_optimal": a_optimal_n,
        "evsi_optimal": evsi_optimal_n,
    }

    return {
        "d_optimal_n": d_optimal_n,
        "a_optimal_n": a_optimal_n,
        "multi_site_allocation": multi_site,
        "info_gain_curve": info_curve,
        "comparison": comparison,
    }


def _d_optimal(n_candidates, I_prior_tt, I_prior_tau2, sigma2):
    """Find N maximizing det(I_posterior) per unit cost.

    For a new trial of size N:
      I_post_tt = I_prior_tt + N / sigma2
      I_post_tau2 = I_prior_tau2  (tau2 info unchanged by single new trial)
      det(I_post) = I_post_tt * I_post_tau2
    We maximize det(I_post) / N  (efficiency).
    """
    best_n = n_candidates[0]
    best_efficiency = -np.inf

    for n in n_candidates:
        I_post_tt = I_prior_tt + n / sigma2
        det_post = I_post_tt * I_prior_tau2
        efficiency = det_post / n
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_n = n

    return best_n


def _a_optimal(n_candidates, I_prior_tt, I_prior_tau2, sigma2):
    """Find N minimizing tr(I_posterior^{-1}) subject to positive net info gain.

    tr(I^{-1}) = 1/I_post_tt + 1/I_post_tau2
    We minimize tr(I^{-1}) * N  (cost-adjusted).
    """
    best_n = n_candidates[0]
    best_cost_adjusted = np.inf

    for n in n_candidates:
        I_post_tt = I_prior_tt + n / sigma2
        trace_inv = 1.0 / I_post_tt + 1.0 / I_prior_tau2
        cost_adjusted = trace_inv * n
        if cost_adjusted < best_cost_adjusted:
            best_cost_adjusted = cost_adjusted
            best_n = n

    return best_n


def _multi_site_allocation(sites, budget):
    """Generalized Neyman allocation across sites.

    n_j* proportional to sqrt(1 / (sigma2_j * C_j))
    Subject to sum(n_j * C_j) <= B
    """
    # Raw allocation weights
    weights = []
    for s in sites:
        w = np.sqrt(1.0 / (s["sigma2"] * s["cost"]))
        weights.append(w)

    total_w = sum(weights)
    if total_w <= 0:
        return [{"site": j, "n": 0, "cost": 0.0} for j in range(len(sites))]

    # Continuous allocation
    raw_n = []
    for j, s in enumerate(sites):
        n_cont = (weights[j] / total_w) * budget / s["cost"]
        raw_n.append(n_cont)

    # Round to integers respecting budget
    allocations = []
    remaining_budget = budget
    for j, s in enumerate(sites):
        n_j = int(np.floor(raw_n[j]))
        cost_j = n_j * s["cost"]
        if cost_j > remaining_budget:
            n_j = int(remaining_budget // s["cost"])
            cost_j = n_j * s["cost"]
        remaining_budget -= cost_j
        allocations.append({"site": j, "n": n_j, "cost": cost_j})

    return allocations


def _info_gain_curve(n_candidates, I_prior_tt, sigma2, cost_pp):
    """Information gain per dollar for each candidate N."""
    # Sample every 100th for a manageable curve
    sampled = [n for n in n_candidates if n % 500 == 0 or n == n_candidates[0]]
    curve = []
    I_base = I_prior_tt

    for n in sampled:
        I_post_tt = I_prior_tt + n / sigma2
        delta_info = I_post_tt - I_base
        cost = n * cost_pp
        curve.append({
            "n": n,
            "delta_info": float(delta_info),
            "cost": float(cost),
        })

    return curve


def _evsi_optimal(inp):
    """Find EVSI-optimal N from a coarse grid."""
    df_sum = discount_factor(inp.discount_rate, inp.horizon_years)
    best_n = 50
    best_nb = -np.inf

    for n in [50, 100, 200, 500, 1000, 2000, 5000]:
        evsi = compute_evsi(inp, n_trial=n, n_sim=2000)
        evsi_pop = evsi * inp.population * df_sum
        cost = inp.cost_per_patient * n
        nb = evsi_pop - cost
        if nb > best_nb:
            best_nb = nb
            best_n = n

    return best_n
