import os

import pytest
from metavoi.models import VoIInput

if os.environ.get("RUN_SLOW_TESTS", "").lower() not in {"1", "true", "yes"}:
    collect_ignore_glob = [
        "test_approximate_bc.py",
        "test_bayesian_bootstrap.py",
        "test_causal_voi.py",
        "test_concentration.py",
        "test_entropy_voi.py",
        "test_evpi.py",
        "test_evppi.py",
        "test_evsi.py",
        "test_fisher_information.py",
        "test_gp_evppi.py",
        "test_grade_bridge.py",
        "test_importance_evsi.py",
        "test_kernel_voi.py",
        "test_martingale.py",
        "test_multi_criteria.py",
        "test_multi_decision.py",
        "test_optimal.py",
        "test_optimal_design.py",
        "test_optimal_stopping.py",
        "test_pipeline.py",
        "test_posterior.py",
        "test_regret.py",
        "test_renyi_voi.py",
        "test_robust_voi.py",
        "test_sample_complexity.py",
        "test_sensitivity_analysis.py",
        "test_sequential_voi.py",
        "test_spectral_risk.py",
        "test_stein_paradox.py",
        "test_stochastic_dominance.py",
        "test_variational_bayes.py",
    ]
else:
    collect_ignore_glob = []


@pytest.fixture
def bcg_input():
    """BCG vaccine: 13 RCTs, logRR, strong effect with high heterogeneity."""
    return VoIInput(
        theta=-0.7141,
        se=0.1787,
        tau2=0.3084,
        k=13,
        mcid=-0.2,
        population=1_000_000,
        horizon_years=20,
        cost_per_patient=500,
        discount_rate=0.035,
        within_study_var=0.0441,
        n_sim=10_000,
        seed=42,
    )


@pytest.fixture
def statin_input():
    """Statins: 5 RCTs, logOR, moderate effect, low heterogeneity."""
    return VoIInput(
        theta=-0.25,
        se=0.055,
        tau2=0.002,
        k=5,
        mcid=-0.1,
        population=10_000_000,
        horizon_years=10,
        cost_per_patient=2000,
        discount_rate=0.035,
        within_study_var=0.017,
        n_sim=10_000,
        seed=42,
    )


@pytest.fixture
def certain_input():
    """Very certain evidence: tiny SE, no heterogeneity, effect far from MCID."""
    return VoIInput(
        theta=-0.80,
        se=0.01,
        tau2=0.0,
        k=50,
        mcid=-0.2,
        population=100_000,
        horizon_years=5,
        cost_per_patient=1000,
        discount_rate=0.035,
        within_study_var=0.01,
        n_sim=10_000,
        seed=42,
    )


@pytest.fixture
def uncertain_input():
    """Very uncertain: wide SE, high heterogeneity, effect near MCID."""
    return VoIInput(
        theta=-0.22,
        se=0.30,
        tau2=0.50,
        k=3,
        mcid=-0.2,
        population=500_000,
        horizon_years=10,
        cost_per_patient=3000,
        discount_rate=0.035,
        within_study_var=0.10,
        n_sim=10_000,
        seed=42,
    )
