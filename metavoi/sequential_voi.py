"""Sequential VoI for adaptive evidence acquisition.

Dynamic programming approach:
  - At each stage t in {0, ..., T-1}, the decision-maker can:
    (a) decide now (implement current optimal treatment), or
    (b) run a trial of size N_t and update the posterior.
  - Backward induction computes the value at each stage.
  - Future stages are discounted by (1 + r)^(-t).

The output is the optimal strategy (sequence of actions) and the
value of waiting versus deciding immediately.
"""

import numpy as np


def _bayesian_update(prior_mean, prior_var, data_mean, data_var):
    """Bayesian update for normal-normal model."""
    prior_prec = 1.0 / prior_var if prior_var > 1e-15 else 1e15
    data_prec = 1.0 / data_var if data_var > 1e-15 else 1e15
    post_prec = prior_prec + data_prec
    post_mean = (prior_prec * prior_mean + data_prec * data_mean) / post_prec
    post_var = 1.0 / post_prec
    return post_mean, post_var


def _nb_decide_now(mean_theta, mcid):
    """NB of making the optimal decision now given current posterior mean."""
    nb_treat = mcid - mean_theta
    return max(nb_treat, 0.0)


def compute_sequential_voi(inp, n_per_stage=500, T=3, n_mc=500):
    """Sequential VoI via backward induction.

    Parameters
    ----------
    inp : VoIInput
        Meta-analysis input data.
    n_per_stage : int
        Trial size at each stage (same for all stages for simplicity).
    T : int
        Number of stages (decision points).
    n_mc : int
        Monte Carlo draws per stage for expectation.

    Returns
    -------
    dict with keys:
        optimal_strategy : list[str]  -- action at each stage
        expected_value : float        -- expected NB under optimal strategy
        value_of_waiting : float      -- gain from sequential vs single-stage
        stage_values : list[float]    -- value function at each stage
    """
    rng = np.random.default_rng(inp.seed + 300)
    mcid = inp.mcid
    sigma2 = inp.within_study_var if inp.within_study_var else inp.se ** 2
    r = inp.discount_rate

    # Prior: posterior of current MA
    prior_mean = inp.theta
    prior_var = inp.se ** 2 + inp.tau2

    trial_cost = inp.cost_per_patient * n_per_stage

    # ------------------------------------------------------------------
    # Backward induction
    # V[T] = decide_now value (terminal stage: must decide)
    # V[t] = max(decide_now, discounted E[V[t+1] | updated posterior] - trial_cost_scaled)
    # ------------------------------------------------------------------

    def _stage_value_recursive(mean_t, var_t, stage, rng_local):
        """Compute value at stage t given posterior (mean_t, var_t)."""
        discount = 1.0 / (1.0 + r) ** stage

        # Value of deciding now
        v_decide = _nb_decide_now(mean_t, mcid) * discount

        if stage >= T - 1:
            # Terminal stage: must decide
            return v_decide, "decide"

        # Value of running a trial and moving to stage t+1
        trial_var = sigma2 / max(n_per_stage, 1)

        # Simulate: draw true theta, then trial result, then updated posterior
        # True theta ~ N(mean_t, var_t) [current belief]
        theta_true = rng_local.normal(mean_t, np.sqrt(max(var_t, 1e-15)), size=n_mc)
        trial_mean = rng_local.normal(theta_true, np.sqrt(max(trial_var, 1e-15)))

        future_values = np.zeros(n_mc)
        for i in range(n_mc):
            upd_mean, upd_var = _bayesian_update(mean_t, var_t, trial_mean[i], trial_var)
            # At next stage, we get the value of optimal strategy
            v_next, _ = _stage_value_recursive(
                upd_mean, upd_var, stage + 1, rng_local
            )
            future_values[i] = v_next

        discount_next = 1.0 / (1.0 + r) ** stage
        trial_cost_scaled = trial_cost / max(inp.population, 1)
        v_wait = float(np.mean(future_values)) - trial_cost_scaled * discount_next

        if v_wait > v_decide:
            return v_wait, "trial"
        else:
            return v_decide, "decide"

    # ------------------------------------------------------------------
    # Forward pass to build strategy
    # ------------------------------------------------------------------
    strategy = []
    stage_values = []
    current_mean = prior_mean
    current_var = prior_var

    for t in range(T):
        rng_stage = np.random.default_rng(inp.seed + 300 + t * 1000)
        v, action = _stage_value_recursive(current_mean, current_var, t, rng_stage)
        strategy.append(action)
        stage_values.append(v)

        if action == "decide":
            # Fill remaining stages as n/a
            for _ in range(t + 1, T):
                strategy.append("n/a")
                stage_values.append(0.0)
            break

        # If trial chosen, simulate a single "representative" update for forward pass
        trial_var = sigma2 / max(n_per_stage, 1)
        # Use the prior mean as the "expected" trial result (no surprise)
        current_mean, current_var = _bayesian_update(
            current_mean, current_var, current_mean, trial_var
        )

    expected_value = stage_values[0]

    # Single-stage value: just decide now
    single_stage = _nb_decide_now(prior_mean, mcid)

    value_of_waiting = max(0.0, expected_value - single_stage)

    return {
        "optimal_strategy": strategy,
        "expected_value": expected_value,
        "value_of_waiting": value_of_waiting,
        "stage_values": stage_values,
    }
