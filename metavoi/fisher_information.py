"""Fisher Information for Meta-Analysis VoI.

Information-geometric analysis of the random-effects model, providing:
- Observed Fisher information matrix for (theta, tau2)
- Cramer-Rao lower bound on VoI estimation precision
- Effective sample size and information ratio
- Jeffreys prior for (theta, tau2) on a tau2 grid
"""

import numpy as np

from metavoi.evpi import compute_evpi
from metavoi.posterior import predictive_distribution


def compute_fisher_information(inp):
    """Compute Fisher information matrix and derived quantities.

    For the random-effects model:
        y_i ~ N(theta, v_i + tau2),  i = 1..k

    Since individual v_i are unavailable, we approximate v_i ~ se^2.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis VoI input parameters.

    Returns
    -------
    dict with keys:
        fisher_matrix : list[list[float]]
            2x2 Fisher information matrix [[I_tt, I_t_tau2], [I_t_tau2, I_tau2_tau2]].
        cramer_rao_bound : float
            Lower bound on Var(EVPI_hat) via delta method.
        effective_sample_size : float
            Number of ideal single-study equivalents.
        information_ratio : float
            Proportion of decision-relevant information from theta vs tau2.
        jeffreys_prior_tau2 : list[float]
            Jeffreys prior density (unnormalized) evaluated on tau2_grid.
        tau2_grid : list[float]
            Grid of tau2 values for Jeffreys prior.
        det_fisher : float
            Determinant of the Fisher information matrix.
    """
    k = inp.k
    se2 = inp.se ** 2
    tau2 = max(inp.tau2, 1e-12)  # avoid division by zero
    w = se2 + tau2  # approximate total variance per study

    # Fisher information entries
    I_tt = k / w                      # I_theta_theta
    I_tau2 = k / (2.0 * w ** 2)      # I_tau2_tau2
    I_cross = 0.0                     # orthogonal in normal model

    fisher_matrix = [[I_tt, I_cross], [I_cross, I_tau2]]
    det_fisher = I_tt * I_tau2 - I_cross ** 2

    # Effective sample size: how many ideal single-study equivalents
    I_single = 1.0 / w
    effective_sample_size = I_tt / I_single if I_single > 0 else 0.0

    # Information ratio: theta info / (theta info + tau2 info)
    # Both dimensions contribute to decision uncertainty. Normalize.
    total_info = I_tt + I_tau2
    information_ratio = I_tt / total_info if total_info > 0 else 0.5

    # Cramer-Rao lower bound on Var(EVPI_hat) via delta method
    # EVPI depends on theta primarily. dEVPI/dtheta ~ 1 near decision boundary.
    # Var(EVPI_hat) >= (dEVPI/dtheta)^2 / I_tt
    # We estimate the gradient numerically.
    cr_bound = _cramer_rao_evpi(inp, I_tt)

    # Jeffreys prior: proportional to sqrt(det(I(tau2)))
    tau2_upper = max(5.0 * tau2, 0.01)
    tau2_grid = np.linspace(0.001, tau2_upper, 50)
    jeffreys_values = []
    for t2 in tau2_grid:
        w_t = se2 + t2
        I_tt_t = k / w_t
        I_tau2_t = k / (2.0 * w_t ** 2)
        jeffreys_values.append(np.sqrt(I_tt_t * I_tau2_t))

    return {
        "fisher_matrix": fisher_matrix,
        "cramer_rao_bound": cr_bound,
        "effective_sample_size": effective_sample_size,
        "information_ratio": information_ratio,
        "jeffreys_prior_tau2": [float(v) for v in jeffreys_values],
        "tau2_grid": [float(v) for v in tau2_grid],
        "det_fisher": det_fisher,
    }


def _cramer_rao_evpi(inp, I_theta_theta):
    """Compute Cramer-Rao bound on EVPI estimator variance via delta method.

    Uses numerical differentiation of EVPI w.r.t. theta.
    """
    from dataclasses import replace

    delta = 0.01 * max(abs(inp.theta), 0.01)

    # EVPI at theta + delta
    inp_plus = replace(inp, theta=inp.theta + delta)
    draws_plus = predictive_distribution(inp_plus)
    evpi_plus = compute_evpi(draws_plus, inp_plus.mcid)

    # EVPI at theta - delta
    inp_minus = replace(inp, theta=inp.theta - delta)
    draws_minus = predictive_distribution(inp_minus)
    evpi_minus = compute_evpi(draws_minus, inp_minus.mcid)

    # Numerical gradient
    d_evpi_d_theta = (evpi_plus - evpi_minus) / (2.0 * delta)

    # Cramer-Rao: Var >= (dEVPI/dtheta)^2 / I_theta_theta
    if I_theta_theta > 0:
        return d_evpi_d_theta ** 2 / I_theta_theta
    return 0.0
