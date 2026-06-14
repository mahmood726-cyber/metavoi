"""test_truth_recovery.py -- assertions over metavoi's OWN compute_evpi vs a
known-truth normal-posterior decision problem with a closed-form EVPI.

Monte-Carlo tolerances are relaxed (per stochastic-test discipline); the
closed-form values are exact. Run: python test_truth_recovery.py
or: python -m pytest test_truth_recovery.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from metavoi.evpi import compute_evpi
from dgp_decision import scenario, evpi_closed_form


def _mc(mu, se, tau2, mcid, seed=0, n_sim=400000):
    sc = scenario(mu, se, tau2, mcid, n_sim=n_sim, seed=seed)
    return compute_evpi(sc["draws"], mcid), sc["evpi_closed_form"], sc["sigma"]


def test_evpi_zero_when_decision_never_changes():
    # CI entirely on the beneficial side of the threshold (|z|=10): EVPI ~ 0.
    mc, cf, sig = _mc(-0.5, 0.05, 0.0, 0.0)
    assert mc < 1e-3, f"EVPI should be ~0 when decision is fixed, got {mc}"
    assert cf < 1e-3


def test_evpi_positive_when_spanning_threshold():
    mc, cf, sig = _mc(0.0, 0.20, 0.0, 0.0)
    assert mc > 0.0, "EVPI must be positive when uncertainty spans threshold"
    assert cf > 0.0


def test_evpi_matches_closed_form():
    # Match the unit-normal-loss closed form across a grid; MC tolerance atol.
    max_abs = 0.0
    for mu in (-0.3, -0.1, 0.0, 0.1, 0.3):
        for se in (0.08, 0.15, 0.25):
            mc, cf, sig = _mc(mu, se, 0.0, 0.0, seed=11)
            max_abs = max(max_abs, abs(mc - cf))
    assert max_abs < 5e-3, f"max abs error vs closed form {max_abs} exceeds MC tol"


def test_evpi_non_negative_always():
    rng = np.random.default_rng(123)
    for _ in range(100):
        mu = float(rng.uniform(-0.6, 0.6))
        se = float(rng.uniform(0.03, 0.4))
        tau2 = float(rng.uniform(0.0, 0.05))
        mcid = float(rng.uniform(-0.2, 0.2))
        mc, cf, sig = _mc(mu, se, tau2, mcid, seed=int(rng.integers(0, 1_000_000)), n_sim=40000)
        assert mc >= 0.0, f"EVPI negative: {mc}"


def test_evpi_scales_linearly_with_sigma_at_threshold():
    # At mu == mcid (z=0), EVPI = sigma * phi(0) = sigma * 0.398942.
    for sig in (0.1, 0.2, 0.4):
        mc, cf, _ = _mc(0.0, sig, 0.0, 0.0, seed=5)
        expected = sig * 0.3989422804
        assert abs(mc - expected) < 5e-3, f"sigma={sig}: {mc} vs {expected}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print("  PASS", fn.__name__); p += 1
        except AssertionError as e:
            print("  FAIL", fn.__name__, "->", e); 
    print(f"\n{p}/{len(fns)} assertions passed.")
    sys.exit(0 if p == len(fns) else 1)
