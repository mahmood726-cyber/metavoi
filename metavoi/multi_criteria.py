"""Multi-Criteria Decision Analysis for VoI.

Extends VoI to multiple outcomes simultaneously, supporting:
- Per-outcome EVPI computation
- Weighted aggregate EVPI
- Pareto frontier over trial sizes
- TOPSIS ranking (Technique for Order of Preference by
  Similarity to Ideal Solution)
"""

import numpy as np
from metavoi.posterior import predictive_distribution, discount_factor
from metavoi.evpi import compute_evpi
from metavoi.evsi import compute_evsi


def compute_multi_criteria(inp, outcomes=None, n_values=None):
    """Multi-criteria VoI analysis.

    Parameters
    ----------
    inp : VoIInput
        Base MetaVoI input (used for population, horizon, discount, seed).
    outcomes : list[dict] or None
        Each dict has {name, theta, se, tau2, mcid, weight}.
        If None, creates two synthetic outcomes from inp:
        efficacy (from inp directly) and safety (attenuated).
    n_values : list[int] or None
        Trial sizes to evaluate for Pareto/TOPSIS. Default: [50..5000].

    Returns
    -------
    dict with keys:
        per_outcome_evpi   -- dict name -> float
        weighted_evpi      -- float
        pareto_optimal_n   -- list of int (Pareto-optimal trial sizes)
        topsis_ranking     -- list of {n, score}
        best_n_topsis      -- int
        outcome_names      -- list of str
    """
    if n_values is None:
        n_values = [50, 100, 200, 500, 1000, 2000, 5000]

    # --- Build outcome list ---
    if outcomes is None or len(outcomes) == 0:
        outcomes = [
            {
                "name": "efficacy",
                "theta": inp.theta,
                "se": inp.se,
                "tau2": inp.tau2,
                "mcid": inp.mcid,
                "weight": 1.0,
            },
            {
                "name": "safety",
                "theta": inp.theta * 0.3,
                "se": inp.se * 1.5,
                "tau2": inp.tau2 * 0.5,
                "mcid": inp.mcid * 0.5,
                "weight": 0.5,
            },
        ]

    outcome_names = [o["name"] for o in outcomes]
    n_outcomes = len(outcomes)
    rng = np.random.default_rng(inp.seed + 50)

    # --- Per-outcome EVPI ---
    per_outcome_evpi = {}
    for o in outcomes:
        pred_var = o["se"] ** 2 + o["tau2"]
        draws = rng.normal(o["theta"], np.sqrt(max(pred_var, 1e-16)),
                           size=inp.n_sim)
        evpi = compute_evpi(draws, o["mcid"])
        per_outcome_evpi[o["name"]] = evpi

    # --- Weighted EVPI ---
    weighted_evpi = sum(
        o["weight"] * per_outcome_evpi[o["name"]] for o in outcomes
    )

    # --- EVSI per outcome per trial size (for Pareto + TOPSIS) ---
    # evsi_matrix[i][j] = EVSI of outcome j at trial size n_values[i]
    from dataclasses import replace
    evsi_matrix = np.zeros((len(n_values), n_outcomes))
    df_sum = discount_factor(inp.discount_rate, inp.horizon_years)

    for j, o in enumerate(outcomes):
        inp_o = replace(
            inp,
            theta=o["theta"],
            se=o["se"],
            tau2=o["tau2"],
            mcid=o["mcid"],
            seed=inp.seed + 50 + j,
        )
        for i, n in enumerate(n_values):
            evsi_val = compute_evsi(inp_o, n_trial=n, n_sim=min(inp.n_sim, 3000))
            evsi_matrix[i, j] = evsi_val

    # --- Cost column ---
    costs = np.array([inp.cost_per_patient * n for n in n_values])

    # --- Pareto frontier ---
    # Criteria: maximise each weighted EVSI_pop, minimise cost
    # A point dominates B if better on ALL criteria
    weighted_evsi_pop = np.zeros(len(n_values))
    for i in range(len(n_values)):
        weighted_evsi_pop[i] = sum(
            outcomes[j]["weight"] * evsi_matrix[i, j] * inp.population * df_sum
            for j in range(n_outcomes)
        )

    # Net benefit = weighted_evsi_pop - cost
    net_benefits = weighted_evsi_pop - costs

    pareto_optimal_n = _pareto_frontier(n_values, evsi_matrix, costs, outcomes,
                                        inp.population, df_sum)

    # --- TOPSIS ranking ---
    topsis_ranking = _topsis(n_values, evsi_matrix, costs, outcomes,
                             inp.population, df_sum)

    best_n_topsis = topsis_ranking[0]["n"] if topsis_ranking else n_values[0]

    return {
        "per_outcome_evpi": per_outcome_evpi,
        "weighted_evpi": float(weighted_evpi),
        "pareto_optimal_n": pareto_optimal_n,
        "topsis_ranking": topsis_ranking,
        "best_n_topsis": best_n_topsis,
        "outcome_names": outcome_names,
    }


def _pareto_frontier(n_values, evsi_matrix, costs, outcomes, population, df_sum):
    """Find Pareto-optimal trial sizes.

    Criteria per trial size:
    - For each outcome: weighted EVSI_pop (maximise)
    - Cost (minimise, i.e. we negate it to maximise -cost)
    """
    n_points = len(n_values)
    n_outcomes = evsi_matrix.shape[1]

    # Build objective matrix: [evsi_pop_1, ..., evsi_pop_J, -cost]
    # All to be maximised.
    obj = np.zeros((n_points, n_outcomes + 1))
    for j in range(n_outcomes):
        obj[:, j] = (outcomes[j]["weight"] * evsi_matrix[:, j]
                      * population * df_sum)
    obj[:, -1] = -costs  # maximise = minimise cost

    pareto_idx = []
    for i in range(n_points):
        dominated = False
        for k in range(n_points):
            if k == i:
                continue
            # k dominates i if k >= i on all criteria and k > i on at least one
            if np.all(obj[k] >= obj[i]) and np.any(obj[k] > obj[i]):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    return [n_values[i] for i in pareto_idx]


def _topsis(n_values, evsi_matrix, costs, outcomes, population, df_sum):
    """TOPSIS ranking of trial sizes.

    Criteria:
    - Per-outcome weighted EVSI_pop (benefit, higher = better)
    - Cost (cost, lower = better)
    """
    n_points = len(n_values)
    n_outcomes = evsi_matrix.shape[1]
    n_criteria = n_outcomes + 1  # outcomes + cost

    # Build decision matrix
    dm = np.zeros((n_points, n_criteria))
    for j in range(n_outcomes):
        dm[:, j] = (outcomes[j]["weight"] * evsi_matrix[:, j]
                     * population * df_sum)
    dm[:, -1] = costs

    # Direction: +1 = benefit (higher better), -1 = cost (lower better)
    direction = np.ones(n_criteria)
    direction[-1] = -1.0

    # --- Normalise columns to [0, 1] ---
    norm_dm = np.zeros_like(dm)
    for c in range(n_criteria):
        col = dm[:, c]
        col_min, col_max = col.min(), col.max()
        rng = col_max - col_min
        if rng < 1e-15:
            norm_dm[:, c] = 0.5  # all same
        else:
            if direction[c] > 0:
                norm_dm[:, c] = (col - col_min) / rng  # higher = better
            else:
                norm_dm[:, c] = (col_max - col) / rng  # lower = better

    # --- Ideal and anti-ideal points ---
    ideal = np.ones(n_criteria)      # best = 1 on all normalised
    anti_ideal = np.zeros(n_criteria) # worst = 0 on all normalised

    # --- Distance to ideal and anti-ideal ---
    dist_ideal = np.sqrt(np.sum((norm_dm - ideal) ** 2, axis=1))
    dist_anti = np.sqrt(np.sum((norm_dm - anti_ideal) ** 2, axis=1))

    # --- TOPSIS score ---
    scores = np.zeros(n_points)
    for i in range(n_points):
        denom = dist_ideal[i] + dist_anti[i]
        scores[i] = dist_anti[i] / denom if denom > 1e-15 else 0.5

    # --- Rank by descending score ---
    ranking = sorted(
        [{"n": n_values[i], "score": float(scores[i])} for i in range(n_points)],
        key=lambda x: -x["score"],
    )
    return ranking
