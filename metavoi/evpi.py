import numpy as np


def compute_evpi(draws, mcid):
    """Expected Value of Perfect Information via Monte Carlo.

    EVPI = E[max(NB_treat, NB_no_treat)] - max(E[NB_treat], E[NB_no_treat])

    For log-scale effects where lower is better (negative = beneficial):
      NB(treat, theta) = mcid - theta  (how much effect exceeds threshold)
      NB(no_treat) = 0
    """
    nb_treat = mcid - draws
    nb_no_treat = np.zeros_like(draws)

    perfect = np.maximum(nb_treat, nb_no_treat)
    current_best = max(np.mean(nb_treat), np.mean(nb_no_treat))

    evpi = float(np.mean(perfect) - current_best)
    return max(0.0, evpi)


def compute_evpi_population(evpi_per_decision, population, discount_factor_sum):
    """Scale per-decision EVPI to population level."""
    return evpi_per_decision * population * discount_factor_sum
