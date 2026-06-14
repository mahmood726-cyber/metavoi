"""dgp_decision.py -- standalone seeded known-truth decision problem for VoI.

Decision: treat vs no-treat. Net benefit (matching metavoi/evpi.py convention,
log-scale effect where lower/more-negative = more beneficial):
    NB(treat, theta)    = mcid - theta
    NB(no_treat, theta) = 0
The uncertain quantity is the true effect theta, with posterior
    theta ~ Normal(mu, sigma^2),  sigma = sqrt(se^2 + tau^2).

CLOSED FORM (the ground truth we test against):
EVPI = E[max(mcid - theta, 0)] - max(mcid - mu, 0)
For theta ~ N(mu, sigma^2), with z = (mcid - mu)/sigma:
    E[max(mcid - theta, 0)] = sigma * ( phi(z) + z * Phi(z) )
    max(mcid - mu, 0)       = sigma * max(z, 0)
=> EVPI = sigma * ( phi(z) + z*Phi(z) - max(z,0) )
        = sigma * ( phi(z) - z*Phi(-z) )       [since z*Phi(z)-max(z,0) = -z*Phi(-z)]
        = sigma * L(|z|)                        where L is the unit normal loss integral
                                                 L(u) = phi(u) - u*Phi(-u).
This is the standard EVPI-for-a-normal-posterior result and is the closed form
the recipe asks us to match.
"""
import numpy as np
from math import erf, exp, sqrt, pi


def _phi(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def evpi_closed_form(mu, sigma, mcid):
    """Exact EVPI for the treat/no-treat normal-posterior problem above."""
    if sigma <= 0:
        return 0.0
    z = (mcid - mu) / sigma
    # EVPI = sigma * (phi(z) + z*Phi(z) - max(z,0))
    return sigma * (_phi(z) + z * _Phi(z) - max(z, 0.0))


def posterior_draws(mu, sigma, n_sim=200000, seed=0):
    """Seeded Monte-Carlo draws from the posterior Normal(mu, sigma^2)."""
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, size=n_sim)


def scenario(mu, se, tau2, mcid, n_sim=200000, seed=0):
    """Bundle a known-truth scenario: returns draws + closed-form EVPI + sigma."""
    sigma = sqrt(se * se + tau2)
    draws = posterior_draws(mu, sigma, n_sim=n_sim, seed=seed)
    cf = evpi_closed_form(mu, sigma, mcid)
    return {"mu": mu, "sigma": sigma, "mcid": mcid, "draws": draws,
            "evpi_closed_form": cf}
