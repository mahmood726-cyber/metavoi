"""Multi-alternative decision Value of Information.

Extends the binary (treat / don't treat) framework to K alternatives,
each with its own effect, uncertainty, and cost.

    EVPI_multi = E[max_k NB(k, theta)] - max_k E[NB(k, theta)]
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Alternative:
    """One decision alternative in a multi-arm comparison."""
    label: str
    effect: float       # point estimate (e.g. logRR)
    se: float           # standard error
    cost: float         # cost per patient
    tau2: float = 0.0   # between-study variance (predictive distribution)


def compute_multi_evpi(alternatives, mcid, population=100_000,
                       horizon_years=10, discount_rate=0.035,
                       n_sim=10_000, seed=42):
    """Multi-alternative EVPI.

    Parameters
    ----------
    alternatives : list[Alternative]
        K decision alternatives.
    mcid : float
        Minimum clinically important difference (threshold).
    population, horizon_years, discount_rate : scalars
        For population-level scaling.
    n_sim : int
        Monte Carlo draws.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: evpi_multi, p_optimal (list), current_best (str),
                    pairwise_evpi (K x K matrix as list of lists)
    """
    rng = np.random.default_rng(seed)
    K = len(alternatives)

    if K == 0:
        return {
            "evpi_multi": 0.0,
            "p_optimal": [],
            "current_best": "",
            "pairwise_evpi": [],
        }

    # ------------------------------------------------------------------
    # Sample theta_k ~ N(effect_k, se_k^2) for each alternative
    # NB(k, theta) = mcid - theta_k - cost_k_scaled
    # We normalise cost to per-patient net-benefit units.
    # ------------------------------------------------------------------
    # Draw effects for each arm from predictive: N(effect, se^2 + tau2)
    effects = np.zeros((K, n_sim))
    for i, alt in enumerate(alternatives):
        pred_var = alt.se ** 2 + alt.tau2
        effects[i] = rng.normal(alt.effect, np.sqrt(max(pred_var, 1e-16)), size=n_sim)

    # Net benefit for each arm: clinical benefit minus cost
    # Cost is normalised to the NB scale (per decision) via cost / population
    nb = np.zeros((K, n_sim))
    for i, alt in enumerate(alternatives):
        clinical_nb = mcid - effects[i]  # higher when effect more negative
        cost_scaled = alt.cost / max(population, 1)
        nb[i] = clinical_nb - cost_scaled

    # ------------------------------------------------------------------
    # EVPI_multi = E[max_k NB_k] - max_k E[NB_k]
    # ------------------------------------------------------------------
    expected_nb = nb.mean(axis=1)  # (K,)
    best_idx = int(np.argmax(expected_nb))
    current_best = alternatives[best_idx].label

    perfect_value = nb.max(axis=0).mean()  # E[max_k NB_k]
    current_value = expected_nb.max()       # max_k E[NB_k]
    evpi_multi = max(0.0, float(perfect_value - current_value))

    # ------------------------------------------------------------------
    # P(optimal) for each arm
    # ------------------------------------------------------------------
    best_per_sim = nb.argmax(axis=0)  # (n_sim,)
    p_optimal = [float(np.mean(best_per_sim == i)) for i in range(K)]

    # ------------------------------------------------------------------
    # Pairwise EVPI: for each pair (i, j), compute binary EVPI
    # ------------------------------------------------------------------
    pairwise = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            diff = nb[i] - nb[j]  # NB advantage of i over j
            e_diff = float(np.mean(diff))
            # Binary EVPI between i and j
            pw_evpi = max(0.0, float(np.mean(np.maximum(diff, 0.0))) - max(e_diff, 0.0))
            pairwise[i][j] = pw_evpi
            pairwise[j][i] = pw_evpi

    return {
        "evpi_multi": evpi_multi,
        "p_optimal": p_optimal,
        "current_best": current_best,
        "pairwise_evpi": pairwise,
    }
