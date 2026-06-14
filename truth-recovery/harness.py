"""harness.py -- wire metavoi's OWN compute_evpi against the known-truth
decision DGP and MEASURE correctness vs the closed form.

Run: python harness.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from metavoi.evpi import compute_evpi          # the repo's OWN function
from dgp_decision import scenario, evpi_closed_form


def measure(mu, se, tau2, mcid, seed=0, n_sim=400000):
    sc = scenario(mu, se, tau2, mcid, n_sim=n_sim, seed=seed)
    mc = compute_evpi(sc["draws"], mcid)         # tool's Monte-Carlo EVPI
    cf = sc["evpi_closed_form"]                   # ground truth
    err = abs(mc - cf)
    return mc, cf, err, sc["sigma"]


def main():
    print("=== metavoi -- EVPI truth-recovery harness ===\n")

    print("--- A) EVPI = 0 when decision NEVER changes (CI one side of threshold) ---")
    # mu far below mcid=0 relative to sigma: P(theta>mcid) ~ 0
    a_cases = [(-0.50, 0.05, 0.0), (-0.40, 0.06, 0.0), (0.50, 0.05, 0.0)]
    a_ok = True
    for mu, se, tau2 in a_cases:
        mc, cf, err, sig = measure(mu, se, tau2, 0.0)
        zdist = abs(mu) / sig
        print(f"  mu={mu:+.2f} sigma={sig:.3f} (|z|={zdist:.1f})  MC_EVPI={mc:.3e}  CF={cf:.3e}")
        if mc > 1e-3:
            a_ok = False
    print(f"  -> all near zero: {a_ok}\n")

    print("--- B) EVPI > 0 when uncertainty SPANS the threshold ---")
    b_cases = [(0.0, 0.20, 0.0), (0.05, 0.25, 0.0), (-0.05, 0.30, 0.0)]
    b_ok = True
    for mu, se, tau2 in b_cases:
        mc, cf, err, sig = measure(mu, se, tau2, 0.0)
        print(f"  mu={mu:+.2f} sigma={sig:.3f}  MC_EVPI={mc:.4f}  CF={cf:.4f}  absErr={err:.2e}")
        if mc <= 0:
            b_ok = False
    print(f"  -> all strictly positive: {b_ok}\n")

    print("--- C) MATCH closed form (unit normal loss integral) across a grid ---")
    maxrel = 0.0
    maxabs = 0.0
    for mu in [-0.3, -0.1, 0.0, 0.1, 0.3]:
        for se in [0.08, 0.15, 0.25]:
            for tau2 in [0.0, 0.01]:
                mc, cf, err, sig = measure(mu, se, tau2, 0.0, seed=11)
                maxabs = max(maxabs, err)
                if cf > 1e-4:
                    maxrel = max(maxrel, err / cf)
    print(f"  grid of 30 cells, n_sim=400k")
    print(f"  max abs error = {maxabs:.2e}")
    print(f"  max rel error (cf>1e-4) = {maxrel*100:.3f}%\n")

    print("--- D) EVPI >= 0 ALWAYS (non-negativity) ---")
    neg = 0
    rng = np.random.default_rng(123)
    for _ in range(200):
        mu = float(rng.uniform(-0.6, 0.6))
        se = float(rng.uniform(0.03, 0.4))
        tau2 = float(rng.uniform(0.0, 0.05))
        mcid = float(rng.uniform(-0.2, 0.2))
        mc, cf, err, sig = measure(mu, se, tau2, mcid, seed=int(rng.integers(0, 1e6)), n_sim=50000)
        if mc < 0:
            neg += 1
    print(f"  random cases with MC_EVPI < 0: {neg}/200\n")

    print("--- E) EVPI scales ~linearly with sigma at fixed z (closed-form property) ---")
    # at mu=mcid, z=0, EVPI = sigma*phi(0) = sigma*0.39894
    for sig_target in [0.1, 0.2, 0.4]:
        mc, cf, err, sig = measure(0.0, sig_target, 0.0, 0.0, seed=5)
        print(f"  sigma={sig_target}: MC={mc:.4f}  expect sigma*0.39894={sig_target*0.398942:.4f}")

    return {"a_ok": a_ok, "b_ok": b_ok, "maxabs": maxabs, "maxrel": maxrel, "neg": neg}


if __name__ == "__main__":
    main()
