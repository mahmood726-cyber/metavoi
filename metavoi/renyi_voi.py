"""Renyi Entropy Generalization of Value of Information.

Extends Shannon-based VoI to the full Renyi/Tsallis entropy family, enabling
min-entropy (worst-case) and collision-entropy perspectives on the decision
value of resolving meta-analytic uncertainty.
"""

import numpy as np
from scipy import stats

from metavoi.posterior import predictive_distribution


def _shannon_entropy_binary(p):
    """Shannon binary entropy H_1(p) in nats (natural log)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def _renyi_entropy_binary(p, alpha):
    """Renyi entropy of order alpha for a binary distribution (p, 1-p).

    H_alpha = 1/(1 - alpha) * log(p^alpha + (1-p)^alpha)  for alpha != 1.
    Limit alpha -> 1 gives Shannon entropy.
    alpha = inf gives min-entropy: -log(max(p, 1-p)).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0

    q = 1.0 - p

    if alpha == np.inf or alpha > 1e6:
        # Min-entropy
        return -np.log(max(p, q))

    if abs(alpha - 1.0) < 1e-10:
        return _shannon_entropy_binary(p)

    # General case
    val = p ** alpha + q ** alpha
    if val <= 0:
        return 0.0
    return (1.0 / (1.0 - alpha)) * np.log(val)


def _tsallis_entropy_binary(p, q_param):
    """Tsallis entropy S_q for binary distribution (p, 1-p).

    S_q = (1 - (p^q + (1-p)^q)) / (q - 1)  for q != 1.
    q=2 yields Gini index: 2*p*(1-p).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0

    r = 1.0 - p

    if abs(q_param - 1.0) < 1e-10:
        return _shannon_entropy_binary(p)

    val = p ** q_param + r ** q_param
    return (1.0 - val) / (q_param - 1.0)


def compute_renyi_voi(inp, n_mc=5000):
    """Compute Renyi-entropy generalized VoI measures.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_mc : int
        Number of Monte Carlo samples (default 5000).

    Returns
    -------
    dict with keys:
        renyi_entropies: dict alpha -> H_alpha (nats).
        renyi_mi: dict alpha -> I_alpha (mutual information).
        min_entropy: float — H_inf of the decision.
        tsallis_entropies: dict q -> S_q.
        entropy_spectrum: list of {alpha, H} over a fine grid.
        shannon_mi: float — Shannon mutual information I_1.
        min_entropy_voi: float — Worst-case VoI from min-entropy.
    """
    rng = np.random.default_rng(inp.seed + 110)

    pred_var = inp.se ** 2 + inp.tau2
    pred_sd = np.sqrt(pred_var)

    # P(treat) = P(theta < mcid) under predictive
    p_treat = float(stats.norm.cdf(inp.mcid, loc=inp.theta, scale=pred_sd))
    p_treat = np.clip(p_treat, 1e-15, 1.0 - 1e-15)

    # ------------------------------------------------------------------
    # Renyi entropies for the decision at various alpha
    # ------------------------------------------------------------------
    alphas = [0.5, 1.0, 2.0, 5.0, np.inf]
    renyi_entropies = {}
    for alpha in alphas:
        key = "inf" if alpha == np.inf or alpha > 1e6 else float(alpha)
        renyi_entropies[key] = _renyi_entropy_binary(p_treat, alpha)

    # ------------------------------------------------------------------
    # Renyi mutual information: I_alpha(D; theta)
    # For binary decision with perfect information, H_alpha(D|theta) = 0
    # (decision becomes deterministic), so I_alpha = H_alpha(D)
    # ------------------------------------------------------------------
    renyi_mi = {}
    for alpha in alphas:
        key = "inf" if alpha == np.inf or alpha > 1e6 else float(alpha)
        renyi_mi[key] = renyi_entropies[key]

    # Shannon MI is the alpha=1 case
    shannon_mi = renyi_entropies[1.0]

    # Min-entropy and min-entropy VoI
    min_entropy = renyi_entropies["inf"]

    # Min-entropy VoI: the value of resolving all uncertainty in worst-case
    # With perfect info, H_inf(D|theta) = 0, so VoI = H_inf(D)
    # Scale by population-level impact: use same MC approach
    # VoI_inf = H_inf(D) (in nats; this is the information-theoretic value)
    min_entropy_voi = min_entropy

    # ------------------------------------------------------------------
    # Tsallis entropies
    # ------------------------------------------------------------------
    q_values = [0.5, 1.0, 2.0, 5.0]
    tsallis_entropies = {}
    for q_val in q_values:
        tsallis_entropies[float(q_val)] = _tsallis_entropy_binary(p_treat, q_val)

    # ------------------------------------------------------------------
    # Entropy spectrum: fine grid of alpha from 0.1 to 20
    # ------------------------------------------------------------------
    alpha_grid = np.concatenate([
        np.linspace(0.1, 1.0, 10),
        np.linspace(1.1, 5.0, 10),
        np.linspace(5.5, 20.0, 10),
    ])
    entropy_spectrum = []
    for a in alpha_grid:
        entropy_spectrum.append({
            "alpha": float(a),
            "H": _renyi_entropy_binary(p_treat, a),
        })

    return {
        "renyi_entropies": renyi_entropies,
        "renyi_mi": renyi_mi,
        "min_entropy": min_entropy,
        "tsallis_entropies": tsallis_entropies,
        "entropy_spectrum": entropy_spectrum,
        "shannon_mi": shannon_mi,
        "min_entropy_voi": min_entropy_voi,
    }
