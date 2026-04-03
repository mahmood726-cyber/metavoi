"""Kernel Methods for Value of Information.

RKHS embedding of posterior distributions for VoI computation using
Gaussian (RBF) kernels, Maximum Mean Discrepancy, and kernel regression.
"""

import numpy as np
from metavoi.posterior import predictive_distribution


def _median_heuristic(samples):
    """Bandwidth via median heuristic: h = median of pairwise distances."""
    n = len(samples)
    # Use a subsample for efficiency if large
    if n > 1000:
        idx = np.linspace(0, n - 1, 1000, dtype=int)
        sub = samples[idx]
    else:
        sub = samples
    dists = np.abs(sub[:, None] - sub[None, :])
    # Upper triangle only (exclude diagonal zeros)
    triu_idx = np.triu_indices(len(sub), k=1)
    median_dist = np.median(dists[triu_idx])
    return max(float(median_dist), 1e-8)


def _rbf_kernel(x, y, h):
    """RBF kernel: k(x, y) = exp(-||x-y||^2 / (2*h^2))."""
    return np.exp(-((x - y) ** 2) / (2.0 * h ** 2))


def _gram_matrix(samples, h):
    """Compute Gram matrix K[i,j] = k(x_i, x_j)."""
    diff = samples[:, None] - samples[None, :]
    return np.exp(-(diff ** 2) / (2.0 * h ** 2))


def _mmd_squared(x, y, h):
    """Unbiased estimate of MMD^2 between samples x and y.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    n = len(x)
    m = len(y)

    Kxx = _gram_matrix(x, h)
    Kyy = _gram_matrix(y, h)
    Kxy = _rbf_kernel(x[:, None], y[None, :], h)

    # Unbiased: exclude diagonal for same-sample terms
    sum_xx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = np.mean(Kxy)

    return float(sum_xx + sum_yy - 2.0 * sum_xy)


def _simulate_post_trial_posterior(inp, rng, n_trial, n_samples):
    """Simulate a post-trial posterior given a new trial of size n_trial.

    The new trial observes y_new ~ N(theta, within_var / n_trial).
    Bayesian update: posterior precision = prior_precision + trial_precision.
    """
    prior_var = inp.se ** 2 + inp.tau2
    within_var = inp.within_study_var if inp.within_study_var is not None else inp.se ** 2
    trial_var = within_var / n_trial

    # Simulate the observed trial mean
    y_new = rng.normal(inp.theta, np.sqrt(trial_var + inp.tau2))

    # Bayesian update (Normal-Normal)
    prior_prec = 1.0 / prior_var
    trial_prec = 1.0 / trial_var
    post_prec = prior_prec + trial_prec
    post_var = 1.0 / post_prec
    post_mean = (prior_prec * inp.theta + trial_prec * y_new) / post_prec

    return rng.normal(post_mean, np.sqrt(post_var), size=n_samples)


def compute_kernel_voi(inp, n_mc=3000):
    """Compute kernel-based VoI measures.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_mc : int
        Number of Monte Carlo samples (default 3000).

    Returns
    -------
    dict with keys:
        mmd_curve: list of {n, mmd} for trial sizes
        kernel_evppi_theta: float
        kernel_evppi_tau2: float
        bandwidth: float
        two_sample_p_value: float
        kernel_mean_norm: float
    """
    rng = np.random.default_rng(inp.seed + 40)

    # --- Prior/current predictive samples ---
    pred_var = inp.se ** 2 + inp.tau2
    pre_samples = rng.normal(inp.theta, np.sqrt(pred_var), size=n_mc)

    # Bandwidth via median heuristic
    h = _median_heuristic(pre_samples)

    # --- Kernel mean embedding norm: ||mu_P||^2 = (1/n^2) sum_ij k(x_i, x_j) ---
    K_pre = _gram_matrix(pre_samples, h)
    kernel_mean_norm = float(np.mean(K_pre))

    # --- MMD curve: for each trial size, simulate post-trial and compute MMD ---
    trial_sizes = [50, 100, 200, 500, 1000, 2000]
    mmd_curve = []
    post_samples_for_test = None

    for n_trial in trial_sizes:
        post_samples = _simulate_post_trial_posterior(inp, rng, n_trial, n_mc)
        mmd2 = _mmd_squared(pre_samples, post_samples, h)
        mmd_val = float(np.sqrt(max(0.0, mmd2)))
        mmd_curve.append({"n": n_trial, "mmd": mmd_val})

        if n_trial == 200:
            post_samples_for_test = post_samples.copy()

    # --- Kernel two-sample test (permutation test on MMD) ---
    if post_samples_for_test is None:
        post_samples_for_test = _simulate_post_trial_posterior(inp, rng, 200, n_mc)

    observed_mmd2 = _mmd_squared(pre_samples, post_samples_for_test, h)
    combined = np.concatenate([pre_samples, post_samples_for_test])
    n_perm = 199
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(combined))
        x_perm = combined[perm[:n_mc]]
        y_perm = combined[perm[n_mc:]]
        perm_mmd2 = _mmd_squared(x_perm, y_perm, h)
        if perm_mmd2 >= observed_mmd2:
            count_ge += 1
    two_sample_p = (count_ge + 1) / (n_perm + 1)

    # --- Kernel EVPPI via Nadaraya-Watson estimator ---
    # EVPPI for theta: use kernel regression of NB on theta
    theta_samples = rng.normal(inp.theta, inp.se, size=n_mc)
    nb_samples = inp.mcid - theta_samples  # NB(treat) = mcid - theta

    # Nadaraya-Watson: E[NB | theta=t] = sum(K_h(t-theta_i)*NB_i) / sum(K_h(t-theta_i))
    # EVPPI = E[max(E[NB|theta], 0)] - max(E[NB], 0)
    h_theta = _median_heuristic(theta_samples)
    weights = _gram_matrix(theta_samples, h_theta)
    # Each row i: conditional expectation at theta_i
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    cond_nb = weights @ nb_samples / row_sums.ravel()
    kernel_evppi_theta = float(np.mean(np.maximum(cond_nb, 0.0)) - max(np.mean(nb_samples), 0.0))
    kernel_evppi_theta = max(0.0, kernel_evppi_theta)

    # EVPPI for tau2: residual approach (total - theta component)
    all_draws = predictive_distribution(inp)
    nb_all = inp.mcid - all_draws
    total_evpi = float(np.mean(np.maximum(nb_all, 0.0)) - max(np.mean(nb_all), 0.0))
    total_evpi = max(0.0, total_evpi)
    kernel_evppi_tau2 = max(0.0, total_evpi - kernel_evppi_theta)

    return {
        "mmd_curve": mmd_curve,
        "kernel_evppi_theta": kernel_evppi_theta,
        "kernel_evppi_tau2": kernel_evppi_tau2,
        "bandwidth": h,
        "two_sample_p_value": two_sample_p,
        "kernel_mean_norm": kernel_mean_norm,
    }
