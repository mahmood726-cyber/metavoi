from dataclasses import dataclass, field


@dataclass
class VoIInput:
    theta: float
    se: float
    tau2: float
    k: int
    mcid: float
    population: int = 100000
    horizon_years: int = 10
    cost_per_patient: float = 5000.0
    discount_rate: float = 0.035
    within_study_var: float | None = None
    n_sim: int = 10000
    seed: int = 42


@dataclass
class EVSIPoint:
    n: int
    evsi: float
    evsi_pop: float
    cost: float
    net_benefit: float


@dataclass
class VoIResult:
    current_optimal: str
    p_wrong: float
    expected_nb_treat: float
    expected_nb_no_treat: float
    evpi: float
    evpi_pop: float
    evppi_theta: float
    evppi_tau2: float
    dominant_parameter: str
    theta_fraction: float
    evsi_curve: list[EVSIPoint] = field(default_factory=list)
    optimal_n: int | None = None
    optimal_evsi_pop: float = 0.0
    optimal_net_benefit: float = 0.0
    breakeven_n: int | None = None
    implied_certainty: str = ""
    grade_recommendation: str = ""
    input_hash: str = ""
    certification: str = ""
