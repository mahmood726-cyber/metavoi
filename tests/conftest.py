import pytest
from metavoi.models import VoIInput


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
