"""Approximate Bayesian Computation for Value of Information.

ABC rejection sampler for posterior inference when the likelihood is
intractable, with tolerance schedule and model comparison via Bayes factors.
"""

import numpy as np
from metavoi.evpi import compute_evpi


def compute_abc_voi(inp, n_proposals=30000):
    """Compute ABC-based posterior and VoI estimates.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_proposals : int
        Number of ABC proposals (default 30000).

    Returns
    -------
    dict with keys:
        abc_posterior_mean : float
            Mean of the ABC posterior at the finest tolerance.
        abc_posterior_sd : float
            SD of the ABC posterior at the finest tolerance.
        abc_evpi_by_tolerance : list of {tolerance, evpi, acceptance_rate}
            EVPI and acceptance rate at each tolerance level.
        analytic_evpi : float
            Standard analytic EVPI for comparison.
        bayes_factor_re_vs_fe : float
            Bayes factor comparing random-effects to fixed-effects model.
        n_accepted : int
            Number of accepted proposals at finest tolerance.
    """
    rng = np.random.default_rng(inp.seed + 70)

    # Observed summary statistics
    s_obs_mean = inp.theta
    s_obs_var = inp.se ** 2 + inp.tau2

    # Analytic EVPI for comparison
    pred_sd = np.sqrt(inp.se ** 2 + inp.tau2)
    analytic_draws = rng.normal(inp.theta, pred_sd, size=inp.n_sim)
    analytic_evpi = compute_evpi(analytic_draws, inp.mcid)

    # --- ABC rejection sampling ---
    # Propose theta from N(0, 5)
    proposed_theta = rng.normal(0.0, 5.0, size=n_proposals)

    # For each proposal, simulate k studies and compute summary statistics
    # Under the random-effects model:
    #   y_i ~ N(theta_true, within_var + tau2)
    within_var = inp.within_study_var if inp.within_study_var is not None else inp.se ** 2
    sim_var = within_var + inp.tau2

    # Simulate k study means for each proposal
    # sim_means[i] = mean of k studies from N(proposed_theta[i], sim_var)
    sim_studies = rng.normal(
        proposed_theta[:, np.newaxis],
        np.sqrt(sim_var),
        size=(n_proposals, inp.k),
    )
    sim_means = np.mean(sim_studies, axis=1)
    sim_vars = np.var(sim_studies, axis=1, ddof=1)

    # Distance: normalized Euclidean on (mean, var) summary statistics
    se_ref = inp.se
    var_ref = inp.se ** 2 + inp.tau2
    dist_mean = (sim_means - s_obs_mean) / se_ref
    dist_var = (sim_vars - s_obs_var) / var_ref
    distances = np.sqrt(dist_mean ** 2 + dist_var ** 2)

    # --- Tolerance schedule ---
    tolerance_schedule = [2.0, 1.0, 0.5, 0.2]
    abc_evpi_by_tolerance = []
    final_accepted_theta = None

    for tol in tolerance_schedule:
        accepted_mask = distances < tol
        n_acc = int(np.sum(accepted_mask))
        acc_rate = n_acc / n_proposals

        if n_acc >= 10:
            accepted_theta = proposed_theta[accepted_mask]
            # Compute EVPI from ABC posterior draws
            abc_draws = rng.normal(
                accepted_theta,
                pred_sd,
            )
            abc_evpi = compute_evpi(abc_draws, inp.mcid)
            final_accepted_theta = accepted_theta
        else:
            abc_evpi = analytic_evpi  # fallback

        abc_evpi_by_tolerance.append({
            "tolerance": tol,
            "evpi": abc_evpi,
            "acceptance_rate": acc_rate,
        })

    # --- ABC posterior statistics (finest tolerance with enough acceptances) ---
    if final_accepted_theta is not None and len(final_accepted_theta) >= 2:
        abc_posterior_mean = float(np.mean(final_accepted_theta))
        abc_posterior_sd = float(np.std(final_accepted_theta, ddof=1))
        n_accepted = len(final_accepted_theta)
    else:
        # Fallback to widest tolerance
        widest_mask = distances < tolerance_schedule[0]
        if np.sum(widest_mask) >= 2:
            accepted_theta = proposed_theta[widest_mask]
            abc_posterior_mean = float(np.mean(accepted_theta))
            abc_posterior_sd = float(np.std(accepted_theta, ddof=1))
            n_accepted = len(accepted_theta)
        else:
            abc_posterior_mean = inp.theta
            abc_posterior_sd = pred_sd
            n_accepted = 0

    # --- Bayes factor: RE vs FE ---
    # Under FE: y_i ~ N(theta, within_var) -> sim_var_fe = within_var
    # Under RE: y_i ~ N(theta, within_var + tau2) -> sim_var_re = within_var + tau2
    # Simulate under FE model
    sim_studies_fe = rng.normal(
        proposed_theta[:, np.newaxis],
        np.sqrt(within_var),
        size=(n_proposals, inp.k),
    )
    sim_means_fe = np.mean(sim_studies_fe, axis=1)
    sim_vars_fe = np.var(sim_studies_fe, axis=1, ddof=1)

    dist_mean_fe = (sim_means_fe - s_obs_mean) / se_ref
    dist_var_fe = (sim_vars_fe - s_obs_var) / var_ref
    distances_fe = np.sqrt(dist_mean_fe ** 2 + dist_var_fe ** 2)

    # Use a moderate tolerance for Bayes factor comparison
    bf_tol = 1.0
    n_acc_re = int(np.sum(distances < bf_tol))
    n_acc_fe = int(np.sum(distances_fe < bf_tol))

    if n_acc_fe > 0:
        bayes_factor_re_vs_fe = n_acc_re / n_acc_fe
    else:
        bayes_factor_re_vs_fe = float("inf") if n_acc_re > 0 else 1.0

    return {
        "abc_posterior_mean": abc_posterior_mean,
        "abc_posterior_sd": abc_posterior_sd,
        "abc_evpi_by_tolerance": abc_evpi_by_tolerance,
        "analytic_evpi": analytic_evpi,
        "bayes_factor_re_vs_fe": bayes_factor_re_vs_fe,
        "n_accepted": n_accepted,
    }
