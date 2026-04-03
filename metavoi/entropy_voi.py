"""Information-Theoretic Value of Information.

Entropy-based VoI decomposition using decision entropy, mutual information,
KL divergence, and channel capacity concepts.
"""

import numpy as np
from scipy import stats
from metavoi.posterior import predictive_distribution


def _binary_entropy(p):
    """Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p), in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def _normal_kl(mu_prior, var_prior, mu_post, var_post):
    """KL(posterior || prior) for univariate Normals.

    KL = 0.5 * (var_prior/var_post - 1 + (mu_post - mu_prior)^2/var_post + log(var_post/var_prior))
    """
    if var_post <= 0 or var_prior <= 0:
        return 0.0
    return 0.5 * (var_prior / var_post - 1.0 +
                  (mu_post - mu_prior) ** 2 / var_post +
                  np.log(var_post / var_prior))


def compute_entropy_voi(inp, n_mc=5000):
    """Compute information-theoretic VoI measures.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input parameters.
    n_mc : int
        Number of Monte Carlo samples (default 5000).

    Returns
    -------
    dict with keys:
        decision_entropy: float (bits)
        mutual_information_theta: float (bits)
        mutual_information_tau2: float (bits)
        entropy_reduction_curve: list of {n, delta_h}
        kl_gain_curve: list of {n, kl}
        channel_capacity: float (bits)
        capacity_utilization: float (fraction)
    """
    rng = np.random.default_rng(inp.seed + 60)

    pred_var = inp.se ** 2 + inp.tau2
    pred_sd = np.sqrt(pred_var)

    # --- Decision entropy ---
    # P(treat) = P(theta < mcid) where theta ~ N(theta_hat, pred_var)
    p_treat = float(stats.norm.cdf(inp.mcid, loc=inp.theta, scale=pred_sd))
    p_no_treat = 1.0 - p_treat
    decision_entropy = _binary_entropy(p_treat)

    # --- Mutual information I(D; theta) = H(D) - H(D|theta) ---
    # With perfect info H(D|theta) = 0, so I(D; theta) = H(D)
    mi_theta = decision_entropy

    # --- Mutual information I(D; tau2 | theta) ---
    # This captures the additional info from knowing tau2 beyond theta
    # Approach: compute I(D; theta, tau2) - I(D; theta)
    # I(D; theta, tau2) = H(D) since knowing both fully determines decision
    # But tau2 doesn't directly determine treatment; it affects uncertainty
    # Use MC: sample theta from posterior on theta alone, compute residual entropy

    # Entropy from theta uncertainty alone (marginal over tau2)
    theta_samples = rng.normal(inp.theta, inp.se, size=n_mc)
    # For each theta sample, decision is deterministic: treat if theta < mcid
    p_treat_given_theta = float(np.mean(theta_samples < inp.mcid))
    h_d_given_theta_marginal = _binary_entropy(p_treat_given_theta)

    # With perfect theta knowledge, H(D|theta) = 0
    # So I(D; theta) as computed via MC = H(D) - E_theta[H(D|theta)] = H(D) - 0 = H(D)
    # The tau2 contribution is about how tau2 affects the predictive uncertainty
    # I(D; tau2 | theta) = H(D | theta) - H(D | theta, tau2)
    # Since both are 0 with perfect knowledge, tau2 info is in the predictive spread

    # Practical approach: information about tau2 is the fraction of EVPI
    # driven by heterogeneity uncertainty. Use entropy decomposition:
    # Compute how much entropy remains if we only resolve theta vs full resolution

    # When only theta is known (but not tau2), there's still uncertainty from tau2
    # Generate samples from predictive with tau2 uncertainty
    if inp.tau2 > 0:
        # Sample tau2 from a scaled chi-squared approximation
        # Prior on tau2: use moment-matching from the DL estimate
        # Shape/scale from method-of-moments: mean=tau2, var proportional
        tau2_shape = max(1.0, inp.k - 1)
        tau2_scale = inp.tau2 / tau2_shape if tau2_shape > 0 else inp.tau2
        tau2_samples = rng.gamma(tau2_shape, tau2_scale, size=n_mc)

        # For each tau2 sample, compute P(treat) with this heterogeneity
        p_treat_per_tau2 = np.array([
            float(stats.norm.cdf(inp.mcid, loc=inp.theta,
                                 scale=np.sqrt(inp.se ** 2 + t2)))
            for t2 in tau2_samples
        ])
        # E_tau2[H(D | tau2)] = mean of binary entropies
        h_d_given_tau2 = float(np.mean([_binary_entropy(p) for p in p_treat_per_tau2]))
        mi_tau2 = max(0.0, decision_entropy - h_d_given_tau2)
    else:
        mi_tau2 = 0.0

    # --- Entropy reduction curve: H_prior(D) - E[H_post(D|trial)] ---
    trial_sizes = [50, 100, 200, 500, 1000, 2000]
    within_var = inp.within_study_var if inp.within_study_var is not None else inp.se ** 2
    entropy_reduction_curve = []
    kl_gain_curve = []

    for n_trial in trial_sizes:
        trial_var = within_var / n_trial

        # Simulate n_mc_sub post-trial posteriors
        n_mc_sub = 1000
        post_entropies = []
        kl_values = []

        for _ in range(n_mc_sub):
            # Simulate trial observation
            y_new = rng.normal(inp.theta, np.sqrt(trial_var + inp.tau2))

            # Bayesian update
            prior_prec = 1.0 / pred_var
            trial_prec = 1.0 / trial_var
            post_prec = prior_prec + trial_prec
            post_var = 1.0 / post_prec
            post_mean = (prior_prec * inp.theta + trial_prec * y_new) / post_prec
            post_sd = np.sqrt(post_var)

            # Post-trial decision entropy
            p_treat_post = float(stats.norm.cdf(inp.mcid, loc=post_mean, scale=post_sd))
            post_entropies.append(_binary_entropy(p_treat_post))

            # KL divergence (posterior || prior)
            kl = _normal_kl(inp.theta, pred_var, post_mean, post_var)
            kl_values.append(kl)

        expected_post_entropy = float(np.mean(post_entropies))
        delta_h = max(0.0, decision_entropy - expected_post_entropy)
        entropy_reduction_curve.append({"n": n_trial, "delta_h": delta_h})

        mean_kl = float(np.mean(kl_values))
        kl_gain_curve.append({"n": n_trial, "kl": mean_kl})

    # --- Channel capacity ---
    # For binary decision: capacity = 1 bit (achieved when p_treat = 0.5)
    channel_capacity = 1.0  # bits

    # Current utilization: how much of the channel capacity is used
    capacity_utilization = decision_entropy / channel_capacity

    return {
        "decision_entropy": decision_entropy,
        "mutual_information_theta": mi_theta,
        "mutual_information_tau2": mi_tau2,
        "entropy_reduction_curve": entropy_reduction_curve,
        "kl_gain_curve": kl_gain_curve,
        "channel_capacity": channel_capacity,
        "capacity_utilization": capacity_utilization,
    }
