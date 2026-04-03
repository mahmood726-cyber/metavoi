"""GP-regression EVPPI (Strong-Oakley-Brennan 2014).

Instead of nested Monte Carlo, fit a Gaussian Process to approximate
the conditional expectation E[NB | parameter], then compute:

    EVPPI = E[max(E[NB|param], 0)] - max(E[NB], 0)

Uses a simple squared-exponential (RBF) kernel GP with Cholesky
decomposition -- no sklearn dependency.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


# ---------------------------------------------------------------------------
# Squared-exponential (RBF) kernel GP
# ---------------------------------------------------------------------------

def _rbf_kernel(x1, x2, length_scale, variance):
    """Compute RBF kernel matrix K(x1, x2)."""
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    sq_dist = (x1 - x2.T) ** 2
    return variance * np.exp(-0.5 * sq_dist / length_scale ** 2)


def _fit_gp(x_train, y_train, length_scale, variance, nugget=1e-6):
    """Fit GP: return Cholesky factor and alpha = K_inv @ y."""
    K = _rbf_kernel(x_train, x_train, length_scale, variance)
    K += nugget * np.eye(len(K))
    L, low = cho_factor(K, lower=True)
    alpha = cho_solve((L, low), y_train)
    return L, low, alpha


def _predict_gp(x_train, x_test, alpha, length_scale, variance):
    """GP mean prediction at x_test."""
    K_star = _rbf_kernel(x_test, x_train, length_scale, variance)
    return K_star @ alpha


def _choose_length_scale(x):
    """Heuristic: use the standard deviation of x as the length scale."""
    sd = float(np.std(x))
    return max(sd, 1e-8)


def _r_squared(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gp_evppi(inp, n_outer=2000):
    """GP-regression EVPPI for theta and tau2.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input data.
    n_outer : int
        Number of outer Monte Carlo samples.

    Returns
    -------
    dict with keys: evppi_theta_gp, evppi_tau2_gp,
                    gp_r2_theta, gp_r2_tau2
    """
    rng = np.random.default_rng(inp.seed + 200)
    mcid = inp.mcid

    # ------------------------------------------------------------------
    # Sample from joint posterior of (theta, tau2)
    # theta ~ N(theta_hat, se^2)
    # tau2 ~ scaled-chi2 approximation => Gamma approximation
    # ------------------------------------------------------------------
    theta_samples = rng.normal(inp.theta, inp.se, size=n_outer)

    # Approximate posterior for tau2: Gamma with matched mean/var
    # mean = tau2, var ~ 2*tau2^2 / (k - 1)   (DL-like approximation)
    tau2_mean = max(inp.tau2, 1e-8)
    tau2_var = 2.0 * tau2_mean ** 2 / max(inp.k - 1, 1)
    shape = tau2_mean ** 2 / tau2_var
    scale = tau2_var / tau2_mean
    tau2_samples = rng.gamma(shape, scale, size=n_outer)

    # ------------------------------------------------------------------
    # Compute NB for each sample from predictive
    # For each (theta_i, tau2_i), draw one predictive theta_pred
    # NB(treat) = mcid - theta_pred;  NB(no_treat) = 0
    # ------------------------------------------------------------------
    pred_sd = np.sqrt(tau2_samples)  # predictive SD beyond mean uncertainty
    theta_pred = rng.normal(theta_samples, np.maximum(pred_sd, 1e-10))
    nb_all = mcid - theta_pred  # NB of treating

    # ------------------------------------------------------------------
    # GP-EVPPI for theta
    # ------------------------------------------------------------------
    ls_theta = _choose_length_scale(theta_samples)
    var_theta = max(float(np.var(nb_all)), 1e-8)

    L_t, low_t, alpha_t = _fit_gp(
        theta_samples, nb_all, ls_theta, var_theta, nugget=1e-6
    )
    mu_theta = _predict_gp(theta_samples, theta_samples, alpha_t, ls_theta, var_theta)
    r2_theta = _r_squared(nb_all, mu_theta)

    # EVPPI_theta = E[max(mu_theta, 0)] - max(E[nb], 0)
    evppi_theta_gp = float(np.mean(np.maximum(mu_theta, 0.0)) - max(np.mean(nb_all), 0.0))
    evppi_theta_gp = max(0.0, evppi_theta_gp)

    # ------------------------------------------------------------------
    # GP-EVPPI for tau2
    # ------------------------------------------------------------------
    ls_tau2 = _choose_length_scale(tau2_samples)
    var_tau2 = max(float(np.var(nb_all)), 1e-8)

    L_tau, low_tau, alpha_tau = _fit_gp(
        tau2_samples, nb_all, ls_tau2, var_tau2, nugget=1e-6
    )
    mu_tau2 = _predict_gp(tau2_samples, tau2_samples, alpha_tau, ls_tau2, var_tau2)
    r2_tau2 = _r_squared(nb_all, mu_tau2)

    evppi_tau2_gp = float(np.mean(np.maximum(mu_tau2, 0.0)) - max(np.mean(nb_all), 0.0))
    evppi_tau2_gp = max(0.0, evppi_tau2_gp)

    return {
        "evppi_theta_gp": evppi_theta_gp,
        "evppi_tau2_gp": evppi_tau2_gp,
        "gp_r2_theta": r2_theta,
        "gp_r2_tau2": r2_tau2,
    }
