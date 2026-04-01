from metavoi.models import VoIInput, VoIResult
from metavoi.posterior import predictive_distribution, p_wrong_decision, discount_factor
from metavoi.evpi import compute_evpi, compute_evpi_population
from metavoi.evppi import compute_evppi
from metavoi.optimal import find_optimal_n
from metavoi.grade_bridge import grade_from_p_wrong
from metavoi.certifier import compute_input_hash, certify


def run_voi(inp: VoIInput) -> VoIResult:
    """End-to-end VoI computation from MA results."""
    # Phase 1: Posterior
    draws = predictive_distribution(inp)
    p_wrong = p_wrong_decision(draws, inp.mcid)

    mean_theta = float(draws.mean())
    nb_treat = inp.mcid - mean_theta
    current_optimal = "treat" if nb_treat > 0 else "no_treat"

    # Phase 2: EVPI
    evpi = compute_evpi(draws, inp.mcid)
    df = discount_factor(inp.discount_rate, inp.horizon_years)
    evpi_pop = compute_evpi_population(evpi, inp.population, df)

    # Phase 3: EVPPI
    evppi = compute_evppi(inp)

    # Phase 4: EVSI + Optimal N
    opt = find_optimal_n(inp)

    # Phase 5: GRADE
    grade = grade_from_p_wrong(p_wrong)

    # Phase 6: Certify
    input_hash = compute_input_hash(inp)
    cert = certify(evpi, opt["curve"], inp.n_sim)

    return VoIResult(
        current_optimal=current_optimal,
        p_wrong=p_wrong,
        expected_nb_treat=max(nb_treat, 0.0),
        expected_nb_no_treat=0.0,
        evpi=evpi,
        evpi_pop=evpi_pop,
        evppi_theta=evppi["theta"],
        evppi_tau2=evppi["tau2"],
        dominant_parameter=evppi["dominant"],
        theta_fraction=evppi["theta_fraction"],
        evsi_curve=opt["curve"],
        optimal_n=opt["optimal_n"],
        optimal_evsi_pop=opt["optimal_evsi_pop"],
        optimal_net_benefit=opt["optimal_net_benefit"],
        breakeven_n=opt["breakeven_n"],
        implied_certainty=grade["certainty"],
        grade_recommendation=grade["recommendation"],
        input_hash=input_hash,
        certification=cert,
    )
