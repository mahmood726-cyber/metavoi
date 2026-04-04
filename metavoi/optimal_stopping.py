"""Optimal Stopping Theory for Evidence Collection.

Applies classical stopping rules — the secretary problem, CUSUM, and SPRT —
to the sequential accumulation of meta-analytic evidence, answering: "When
should we stop collecting studies and commit to a treatment decision?"
"""

import math

import numpy as np
from scipy import stats


def compute_optimal_stopping(inp, n_runs=200):
    """Compute optimal stopping analysis for sequential evidence collection.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_runs : int
        Number of simulation runs for value-of-continuing (default 200).

    Returns
    -------
    dict with keys:
        secretary_threshold: int — Number of studies to observe before committing.
        secretary_expected_rank: float — Expected rank of selected study.
        cusum_arl: float — Average run length to decision under CUSUM.
        cusum_threshold: float — CUSUM stopping boundary h.
        sprt_boundaries: dict with 'upper' and 'lower' log-likelihood bounds.
        sprt_expected_n_h0: float — Expected sample size under H0 (ineffective).
        sprt_expected_n_h1: float — Expected sample size under H1 (effective).
        optimal_stopping_n: int — Optimal number of studies before stopping.
        value_of_continuing: float — Expected net benefit of one more study.
    """
    rng = np.random.default_rng(inp.seed + 120)
    k = max(inp.k, 3)  # Need at least 3 studies for meaningful stopping

    # ==================================================================
    # 1. Secretary problem analogue
    # ==================================================================
    # Optimal strategy: observe first k/e studies, then stop at next best
    secretary_threshold = max(1, int(round(k / math.e)))

    # Expected rank of the selected candidate under optimal 1/e strategy
    # For the classical problem, E[rank] ~ 1 + ln(k)/k (approximately)
    # Exact: the probability of selecting the best is ~1/e ≈ 0.368
    # Expected rank can be computed by simulation
    n_secretary_sim = 5000
    ranks_selected = []
    for _ in range(n_secretary_sim):
        # Generate k "quality" scores (lower P(wrong) = better)
        qualities = rng.uniform(0, 1, size=k)
        # Observe first `secretary_threshold` studies
        if secretary_threshold >= k:
            # Must pick the last one
            selected_idx = k - 1
        else:
            best_in_sample = np.min(qualities[:secretary_threshold])
            selected_idx = None
            for j in range(secretary_threshold, k):
                if qualities[j] < best_in_sample:
                    selected_idx = j
                    break
            if selected_idx is None:
                selected_idx = k - 1  # Forced to pick last

        # Rank of selected (1 = best)
        rank = 1 + int(np.sum(qualities < qualities[selected_idx]))
        ranks_selected.append(rank)

    secretary_expected_rank = float(np.mean(ranks_selected))

    # ==================================================================
    # 2. CUSUM stopping rule
    # ==================================================================
    # z-scores: standardised evidence from each study
    # Under H0 (no effect beyond MCID): z ~ N(0, 1)
    # Under H1 (effect = theta): z ~ N(delta, 1) where delta = (theta - mcid) / se
    se_study = inp.se  # per-study SE
    delta = (inp.theta - inp.mcid) / se_study if se_study > 0 else 0.0

    # CUSUM parameter: k_cusum = delta/2 (optimal for detecting shift of size delta)
    k_cusum = abs(delta) / 2.0 if abs(delta) > 0 else 0.5

    # Threshold h: set for ARL0 ≈ 500 under H0 (standard choice)
    # For one-sided CUSUM, h ≈ 5 gives ARL0 ~ 500 at k_cusum = 0.5
    h = 5.0

    # Simulate CUSUM runs to get ARL
    n_cusum_sim = 2000
    run_lengths = []
    max_steps = k * 10  # Cap to prevent infinite loops

    for _ in range(n_cusum_sim):
        S_t = 0.0
        for step in range(1, max_steps + 1):
            # Evidence z-score under H1 (true effect exists)
            z = rng.normal(abs(delta), 1.0)
            S_t = max(0.0, S_t + z - k_cusum)
            if S_t > h:
                run_lengths.append(step)
                break
        else:
            run_lengths.append(max_steps)

    cusum_arl = float(np.mean(run_lengths))
    cusum_threshold = h

    # ==================================================================
    # 3. SPRT (Sequential Probability Ratio Test)
    # ==================================================================
    alpha_sprt = 0.05
    beta_sprt = 0.20

    # Wald boundaries
    upper_boundary = math.log((1.0 - beta_sprt) / alpha_sprt)  # Reject H0
    lower_boundary = math.log(beta_sprt / (1.0 - alpha_sprt))  # Accept H0

    sprt_boundaries = {
        "upper": float(upper_boundary),
        "lower": float(lower_boundary),
    }

    # Expected sample sizes under each hypothesis (Wald's approximation)
    # E[N | H0] ≈ ((1-alpha)*log(beta/(1-alpha)) + alpha*log((1-beta)/alpha)) / E_0[log LR]
    # E[N | H1] ≈ ((1-beta)*log((1-beta)/alpha) + beta*log(beta/(1-alpha))) / E_1[log LR]
    # For Normal with known variance:
    # log LR_i under H0: -delta * z_i + delta^2 / 2 where z_i ~ N(0,1)
    # E_0[log LR] = -delta^2 / 2
    # E_1[log LR] = delta^2 / 2

    delta2 = delta ** 2
    if delta2 > 1e-10:
        # Wald's formulas
        e0_llr = -delta2 / 2.0  # E[log LR] under H0
        e1_llr = delta2 / 2.0   # E[log LR] under H1

        # OC and ASN from Wald
        sprt_expected_n_h0 = abs(
            ((1.0 - alpha_sprt) * lower_boundary + alpha_sprt * upper_boundary) / e0_llr
        )
        sprt_expected_n_h1 = abs(
            ((1.0 - beta_sprt) * upper_boundary + beta_sprt * lower_boundary) / e1_llr
        )
    else:
        # No effect difference: SPRT will take very long
        sprt_expected_n_h0 = float(k * 10)
        sprt_expected_n_h1 = float(k * 10)

    # ==================================================================
    # 4. Value of stopping vs continuing
    # ==================================================================
    # Net benefit of current decision
    pred_var = inp.se ** 2 + inp.tau2
    pred_sd = np.sqrt(pred_var)

    # Current expected NB
    p_treat_correct = float(stats.norm.cdf(inp.mcid, loc=inp.theta, scale=pred_sd))
    nb_treat = inp.mcid - inp.theta  # Expected NB of treating
    nb_no_treat = 0.0
    current_nb = max(nb_treat, nb_no_treat)

    # Simulate adding one more study and recomputing
    within_var = inp.within_study_var if inp.within_study_var is not None else inp.se ** 2
    study_var = within_var + inp.tau2
    marginal_nbs = []

    for _ in range(n_runs):
        # New study observation
        y_new = rng.normal(inp.theta, np.sqrt(study_var))

        # Bayesian update
        prior_prec = 1.0 / pred_var
        study_prec = 1.0 / within_var if within_var > 0 else 1.0 / (inp.se ** 2)
        post_prec = prior_prec + study_prec
        post_var = 1.0 / post_prec
        post_mean = (prior_prec * inp.theta + study_prec * y_new) / post_prec

        # Optimal decision with updated info
        nb_treat_new = inp.mcid - post_mean
        updated_nb = max(nb_treat_new, nb_no_treat)
        marginal_nbs.append(updated_nb)

    expected_nb_after = float(np.mean(marginal_nbs))
    value_of_continuing = max(0.0, expected_nb_after - current_nb)

    # Optimal stopping: simulate sequential accumulation
    # Stop when marginal value drops below marginal cost
    cost_per_study = inp.cost_per_patient  # Proxy: cost per patient ~ cost per study
    best_stopping_n = 1
    best_cumulative_nb = -np.inf

    for n_stop in range(1, k + 1):
        # Simulate accumulating n_stop studies
        run_nbs = []
        for _ in range(n_runs):
            # Accumulate n_stop observations
            current_mean = inp.theta
            current_prec = 1.0 / pred_var
            for _ in range(n_stop):
                y_s = rng.normal(inp.theta, np.sqrt(study_var))
                study_prec_s = 1.0 / within_var if within_var > 0 else 1.0 / (inp.se ** 2)
                new_prec = current_prec + study_prec_s
                current_mean = (current_prec * current_mean + study_prec_s * y_s) / new_prec
                current_prec = new_prec

            nb_after_n = max(inp.mcid - current_mean, 0.0)
            total_cost = n_stop * cost_per_study / inp.population  # Normalize
            run_nbs.append(nb_after_n - total_cost)

        avg_nb = float(np.mean(run_nbs))
        if avg_nb > best_cumulative_nb:
            best_cumulative_nb = avg_nb
            best_stopping_n = n_stop

    optimal_stopping_n = best_stopping_n

    return {
        "secretary_threshold": secretary_threshold,
        "secretary_expected_rank": secretary_expected_rank,
        "cusum_arl": cusum_arl,
        "cusum_threshold": cusum_threshold,
        "sprt_boundaries": sprt_boundaries,
        "sprt_expected_n_h0": float(sprt_expected_n_h0),
        "sprt_expected_n_h1": float(sprt_expected_n_h1),
        "optimal_stopping_n": optimal_stopping_n,
        "value_of_continuing": float(value_of_continuing),
    }
