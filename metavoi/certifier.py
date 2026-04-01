import hashlib
import json


def compute_input_hash(inp):
    """SHA-256 hash of input parameters (first 16 hex chars)."""
    d = {
        "theta": inp.theta, "se": inp.se, "tau2": inp.tau2, "k": inp.k,
        "mcid": inp.mcid, "population": inp.population,
        "horizon_years": inp.horizon_years, "cost_per_patient": inp.cost_per_patient,
        "discount_rate": inp.discount_rate, "n_sim": inp.n_sim, "seed": inp.seed,
    }
    raw = json.dumps(d, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def certify(evpi, evsi_curve, n_sim):
    """Certification: PASS if computation completed with enough simulations."""
    if n_sim < 1000:
        return "REJECT"
    if n_sim < 5000:
        return "WARN"
    return "PASS"
