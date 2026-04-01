import numpy as np


def predictive_distribution(inp):
    """Draw from the predictive distribution: N(theta, se^2 + tau^2)."""
    rng = np.random.default_rng(inp.seed)
    pred_var = inp.se ** 2 + inp.tau2
    return rng.normal(inp.theta, np.sqrt(pred_var), size=inp.n_sim)


def p_wrong_decision(draws, mcid):
    """Probability that the optimal decision based on E[theta] is wrong.

    If E[draws] < mcid (effect exceeds threshold in beneficial direction),
    optimal decision is 'treat'. P(wrong) = P(theta > mcid).
    Otherwise optimal is 'no treat'. P(wrong) = P(theta <= mcid).
    """
    mean_theta = np.mean(draws)
    if mean_theta < mcid:
        return float(np.mean(draws > mcid))
    else:
        return float(np.mean(draws <= mcid))


def discount_factor(rate, years):
    """Sum of discounted life-years: sum(1/(1+r)^t for t in 0..years-1)."""
    if rate <= 0:
        return float(years)
    return sum(1.0 / (1.0 + rate) ** t for t in range(years))
