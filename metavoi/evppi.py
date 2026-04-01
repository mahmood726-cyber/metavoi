import numpy as np
from metavoi.posterior import predictive_distribution


def compute_evppi(inp, n_outer=2000):
    """EVPPI for theta and tau2 using nested Monte Carlo.

    For theta: condition on theta, the decision is deterministic
    (if theta < mcid, treat; otherwise don't). So EVPPI_theta captures
    how much knowing theta exactly would help.

    For tau2: residual = total EVPI - EVPPI_theta.
    """
    rng = np.random.default_rng(inp.seed + 1)
    mcid = inp.mcid

    # EVPPI for theta: sample theta from posterior N(theta_hat, se^2)
    theta_outer = rng.normal(inp.theta, inp.se, size=n_outer)

    # With perfect knowledge of theta, NB(treat) = mcid - theta, NB(no_treat) = 0
    nb_treat_theta = mcid - theta_outer
    perfect_given_theta = np.maximum(nb_treat_theta, 0.0)
    evppi_theta = float(np.mean(perfect_given_theta) - max(np.mean(nb_treat_theta), 0.0))
    evppi_theta = max(0.0, evppi_theta)

    # Total EVPI for comparison
    all_draws = predictive_distribution(inp)
    nb_treat_all = mcid - all_draws
    perfect_all = np.maximum(nb_treat_all, 0.0)
    total_evpi = float(np.mean(perfect_all) - max(np.mean(nb_treat_all), 0.0))
    total_evpi = max(0.0, total_evpi)

    # EVPPI_tau2 is the residual
    evppi_tau2 = max(0.0, total_evpi - evppi_theta)

    total = evppi_theta + evppi_tau2
    theta_frac = evppi_theta / total if total > 1e-12 else 0.5
    dominant = "theta" if evppi_theta >= evppi_tau2 else "tau2"

    return {
        "theta": evppi_theta,
        "tau2": evppi_tau2,
        "theta_fraction": theta_frac,
        "dominant": dominant,
    }
