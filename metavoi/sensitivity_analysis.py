"""Probabilistic Sensitivity Analysis for VoI.

Quantifies which input parameters drive decision uncertainty via:
- Partial Rank Correlation Coefficients (PRCC)
- Tornado diagram data (one-at-a-time ±20%)
- Scatter plot data for most influential parameters
"""

import numpy as np
from metavoi.posterior import predictive_distribution, discount_factor
from metavoi.evpi import compute_evpi
from metavoi.models import VoIInput


def _compute_evpi_for_params(theta, se, tau2, mcid, population, horizon_years,
                              cost_per_patient, discount_rate, k, seed, n_sim):
    """Compute EVPI for a given set of parameter values."""
    inp = VoIInput(
        theta=theta, se=se, tau2=tau2, k=k, mcid=mcid,
        population=int(population), horizon_years=int(horizon_years),
        cost_per_patient=cost_per_patient, discount_rate=discount_rate,
        seed=seed, n_sim=n_sim,
    )
    draws = predictive_distribution(inp)
    evpi_per = compute_evpi(draws, mcid)
    df = discount_factor(discount_rate, int(horizon_years))
    return evpi_per * population * df


def _rankdata(x):
    """Simple rank transform (average rank for ties)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def _partial_correlation_from_corr(R, i, j):
    """Partial correlation between i and j controlling for all others.

    rho_ij.k = -P_ij / sqrt(P_ii * P_jj) where P = inv(R)
    """
    try:
        P = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # Singular matrix — use pseudoinverse
        P = np.linalg.pinv(R)
    denom = np.sqrt(abs(P[i, i] * P[j, j]))
    if denom < 1e-15:
        return 0.0
    return float(-P[i, j] / denom)


def compute_sensitivity(inp, n_samples=5000):
    """Run probabilistic sensitivity analysis.

    Returns dict with:
        prcc: dict of param_name -> correlation
        tornado: list of {param, low, high, range}
        scatter_x1, scatter_y1: most influential param scatter
        scatter_x2, scatter_y2: second most influential
        most_influential: str
    """
    rng = np.random.default_rng(inp.seed + 20)

    # --- 1. Sample parameter sets (±20% perturbation) ---
    param_names = ["theta", "tau2", "mcid", "population"]
    base_values = {
        "theta": inp.theta,
        "tau2": inp.tau2,
        "mcid": inp.mcid,
        "population": float(inp.population),
    }

    # Generate perturbed samples
    samples = {}
    for name in param_names:
        base = base_values[name]
        if abs(base) < 1e-12:
            # For near-zero base, use absolute perturbation
            samples[name] = rng.uniform(base - 0.1, base + 0.1, size=n_samples)
        else:
            low = base * 0.8
            high = base * 1.2
            if low > high:
                low, high = high, low
            samples[name] = rng.uniform(low, high, size=n_samples)

    # Compute EVPI for each sample
    evpi_values = np.zeros(n_samples)
    # Use smaller n_sim for speed in sensitivity loop
    inner_n_sim = 2000
    for i in range(n_samples):
        evpi_values[i] = _compute_evpi_for_params(
            theta=samples["theta"][i],
            se=inp.se,
            tau2=max(samples["tau2"][i], 0.0),
            mcid=samples["mcid"][i],
            population=max(samples["population"][i], 1),
            horizon_years=inp.horizon_years,
            cost_per_patient=inp.cost_per_patient,
            discount_rate=inp.discount_rate,
            k=inp.k,
            seed=inp.seed + i,
            n_sim=inner_n_sim,
        )

    # --- 2. PRCC via rank transform + partial correlation ---
    n_params = len(param_names)
    # Build rank matrix: [n_samples x (n_params + 1)] — params + EVPI
    rank_matrix = np.column_stack(
        [_rankdata(samples[name]) for name in param_names] + [_rankdata(evpi_values)]
    )

    # Spearman rank correlation matrix
    R = np.corrcoef(rank_matrix, rowvar=False)
    # Add small regularization to ensure invertibility
    R += np.eye(R.shape[0]) * 1e-8

    evpi_idx = n_params  # last column
    prcc = {}
    for j, name in enumerate(param_names):
        prcc[name] = _partial_correlation_from_corr(R, j, evpi_idx)

    # --- 3. Tornado diagram (one-at-a-time) ---
    base_evpi = _compute_evpi_for_params(
        theta=inp.theta, se=inp.se, tau2=inp.tau2, mcid=inp.mcid,
        population=float(inp.population), horizon_years=inp.horizon_years,
        cost_per_patient=inp.cost_per_patient, discount_rate=inp.discount_rate,
        k=inp.k, seed=inp.seed, n_sim=inp.n_sim,
    )

    tornado = []
    for name in param_names:
        base = base_values[name]
        if abs(base) < 1e-12:
            low_val, high_val = base - 0.1, base + 0.1
        else:
            low_val, high_val = base * 0.8, base * 1.2
            if low_val > high_val:
                low_val, high_val = high_val, low_val

        kwargs = {
            "theta": inp.theta, "se": inp.se, "tau2": inp.tau2,
            "mcid": inp.mcid, "population": float(inp.population),
            "horizon_years": inp.horizon_years,
            "cost_per_patient": inp.cost_per_patient,
            "discount_rate": inp.discount_rate,
            "k": inp.k, "seed": inp.seed, "n_sim": inp.n_sim,
        }

        kwargs[name] = low_val
        if name == "tau2":
            kwargs[name] = max(kwargs[name], 0.0)
        evpi_low = _compute_evpi_for_params(**kwargs)

        kwargs[name] = high_val
        evpi_high = _compute_evpi_for_params(**kwargs)

        tornado.append({
            "param": name,
            "low": evpi_low,
            "high": evpi_high,
            "range": abs(evpi_high - evpi_low),
        })

    tornado.sort(key=lambda x: x["range"], reverse=True)

    # --- 4. Scatter data for top 2 influential parameters ---
    sorted_by_abs_prcc = sorted(param_names, key=lambda n: abs(prcc[n]), reverse=True)
    most_influential = sorted_by_abs_prcc[0]
    second_influential = sorted_by_abs_prcc[1] if len(sorted_by_abs_prcc) > 1 else sorted_by_abs_prcc[0]

    return {
        "prcc": prcc,
        "tornado": tornado,
        "scatter_x1": samples[most_influential].tolist(),
        "scatter_y1": evpi_values.tolist(),
        "scatter_x2": samples[second_influential].tolist(),
        "scatter_y2": evpi_values.tolist(),
        "most_influential": most_influential,
    }
