"""Variational Inference for VoI.

Mean-field variational Bayes as a fast, deterministic alternative to
Monte Carlo EVPI estimation.

Model:
    y_i ~ N(theta, v_i + tau2),  i = 1..k
    theta ~ N(0, 100)            (weakly informative prior)
    tau2  ~ InvGamma(0.001, 0.001)

Variational family:
    q(theta) = N(mu_q, sigma_q^2)
    q(tau2)  = InvGamma(a_q, b_q)

CAVI (Coordinate Ascent Variational Inference) iterates until the ELBO
converges, then uses the variational posterior to compute VB-EVPI.
"""

import numpy as np
from scipy.special import gammaln, digamma

from metavoi.posterior import predictive_distribution
from metavoi.evpi import compute_evpi


# ---------------------------------------------------------------------------
# ELBO components
# ---------------------------------------------------------------------------

def _elbo(yi, vi, mu_q, sigma_q2, a_q, b_q, k, prior_var=100.0,
          a0=0.001, b0=0.001):
    """Compute the Evidence Lower Bound.

    ELBO = E_q[log p(y|theta,tau2)]
         + E_q[log p(theta)]
         + E_q[log p(tau2)]
         - E_q[log q(theta)]
         - E_q[log q(tau2)]
    """
    # E_q[tau2] and E_q[log(tau2)] under InvGamma(a_q, b_q)
    e_tau2 = b_q / (a_q - 1.0) if a_q > 1.0 else b_q  # mean of InvGamma
    e_log_tau2 = np.log(b_q) - digamma(a_q)  # E[log(tau2)]

    # --- E_q[log p(y|theta, tau2)] ---
    # Each y_i ~ N(theta, v_i + tau2)
    # E_q[log N(y_i; theta, v_i + tau2)]
    #   = -0.5 * log(2*pi) - 0.5 * E[log(v_i+tau2)]
    #     - 0.5 * E[(y_i - theta)^2 / (v_i + tau2)]
    # Approximation: E[log(v+tau2)] ~ log(v + E[tau2]),
    #                E[1/(v+tau2)] ~ 1/(v + E[tau2])
    total_var = vi + e_tau2
    log_lik = 0.0
    for i in range(k):
        tv = total_var[i] if k > 1 else total_var
        e_sq = (yi[i] - mu_q) ** 2 + sigma_q2
        log_lik += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(tv) - 0.5 * e_sq / tv

    # --- E_q[log p(theta)] ---
    # theta ~ N(0, prior_var)
    log_prior_theta = (-0.5 * np.log(2 * np.pi * prior_var)
                       - 0.5 * (mu_q ** 2 + sigma_q2) / prior_var)

    # --- E_q[log p(tau2)] ---
    # tau2 ~ InvGamma(a0, b0)
    # log p(tau2) = a0*log(b0) - gammaln(a0) - (a0+1)*log(tau2) - b0/tau2
    e_inv_tau2 = a_q / b_q  # E[1/tau2] for InvGamma
    log_prior_tau2 = (a0 * np.log(b0) - gammaln(a0)
                      - (a0 + 1.0) * e_log_tau2
                      - b0 * e_inv_tau2)

    # --- Entropy of q(theta) = N(mu_q, sigma_q2) ---
    h_theta = 0.5 * np.log(2 * np.pi * np.e * sigma_q2)

    # --- Entropy of q(tau2) = InvGamma(a_q, b_q) ---
    h_tau2 = (a_q + np.log(b_q) + gammaln(a_q)
              - (1.0 + a_q) * digamma(a_q))

    elbo = log_lik + log_prior_theta + log_prior_tau2 + h_theta + h_tau2
    return float(elbo)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_variational_bayes(inp, max_iter=200, tol=1e-6):
    """Run mean-field CAVI and compute VB-EVPI.

    Since we don't have individual study-level y_i, we create k
    pseudo-observations centred at theta with within-study variance se^2.

    Parameters
    ----------
    inp : VoIInput
        Standard MetaVoI input.
    max_iter : int
        Maximum CAVI iterations.
    tol : float
        ELBO convergence tolerance (absolute change).

    Returns
    -------
    dict with keys:
        mu_q, sigma_q, a_q, b_q           -- variational parameters
        elbo_trace                          -- list of ELBO values per iteration
        converged                           -- bool
        iterations                          -- int
        vb_evpi                             -- EVPI from variational posterior
        kl_divergence_approx                -- approximate KL(q || p)
        mc_evpi                             -- standard MC-EVPI for comparison
    """
    k = inp.k
    prior_var = 100.0
    a0, b0 = 0.001, 0.001

    # --- Pseudo-observations ---
    # k studies each reporting theta with variance se^2
    rng = np.random.default_rng(inp.seed + 40)
    yi = np.full(k, inp.theta) + rng.normal(0, inp.se * 0.1, size=k)
    vi = np.full(k, inp.se ** 2)

    # --- Initialise variational parameters ---
    mu_q = inp.theta
    sigma_q2 = inp.se ** 2
    a_q = a0 + k / 2.0
    b_q = b0 + 0.5 * k * inp.tau2 if inp.tau2 > 0 else b0 + 0.1

    elbo_trace = []
    converged = False
    iterations = 0

    for it in range(max_iter):
        # E[tau2] under current q(tau2) = InvGamma(a_q, b_q)
        e_tau2 = b_q / (a_q - 1.0) if a_q > 1.0 else b_q

        # --- Update q(theta) = N(mu_q, sigma_q2) ---
        precision_lik = np.sum(1.0 / (vi + e_tau2))
        precision_prior = 1.0 / prior_var
        total_precision = precision_lik + precision_prior
        sigma_q2 = 1.0 / total_precision
        mu_q = sigma_q2 * np.sum(yi / (vi + e_tau2))

        # --- Update q(tau2) = InvGamma(a_q, b_q) ---
        a_q = a0 + k / 2.0
        sum_sq = np.sum((yi - mu_q) ** 2 + sigma_q2)
        # b_q = b0 + 0.5 * sum(E[(yi-theta)^2] - vi)
        # but clamp to avoid negative b_q when vi dominates
        b_q_candidate = b0 + 0.5 * max(sum_sq - np.sum(vi), 0.0)
        b_q = max(b_q_candidate, 1e-10)

        # --- ELBO ---
        elbo = _elbo(yi, vi, mu_q, sigma_q2, a_q, b_q, k,
                      prior_var=prior_var, a0=a0, b0=b0)
        elbo_trace.append(elbo)
        iterations = it + 1

        if len(elbo_trace) >= 2 and abs(elbo_trace[-1] - elbo_trace[-2]) < tol:
            converged = True
            break

    sigma_q = np.sqrt(sigma_q2)

    # --- VB-EVPI: sample from q(theta) and compute EVPI ---
    rng_vb = np.random.default_rng(inp.seed + 41)
    theta_samples = rng_vb.normal(mu_q, sigma_q, size=inp.n_sim)
    nb_treat = inp.mcid - theta_samples
    nb_no_treat = np.zeros_like(theta_samples)
    perfect = np.maximum(nb_treat, nb_no_treat)
    current_best = max(float(np.mean(nb_treat)), 0.0)
    vb_evpi = max(0.0, float(np.mean(perfect) - current_best))

    # --- MC-EVPI for comparison ---
    draws = predictive_distribution(inp)
    mc_evpi = compute_evpi(draws, inp.mcid)

    # --- Approximate KL(q || p) ---
    # KL = log(mc_evidence) - ELBO  (intractable, so approximate via
    # |VB-EVPI - MC-EVPI| / max(MC-EVPI, 1e-10) as a practical measure)
    kl_approx = abs(vb_evpi - mc_evpi) / max(mc_evpi, 1e-10)

    return {
        "mu_q": float(mu_q),
        "sigma_q": float(sigma_q),
        "a_q": float(a_q),
        "b_q": float(b_q),
        "elbo_trace": [float(e) for e in elbo_trace],
        "converged": converged,
        "iterations": iterations,
        "vb_evpi": float(vb_evpi),
        "kl_divergence_approx": float(kl_approx),
        "mc_evpi": float(mc_evpi),
    }
