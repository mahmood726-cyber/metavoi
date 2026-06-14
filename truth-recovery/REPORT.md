# Truth-Recovery Validation -- metavoi (Value of Information)

**Repo:** mahmood726-cyber/metavoi
**Engine under test:** metavoi/evpi.py::compute_evpi (imported VERBATIM at runtime)
**Method:** drove the repo's own EVPI function against a seeded known-truth
normal-posterior decision problem whose EVPI has an exact closed form (the unit
normal loss integral), and checked the four properties the recipe requires.

## Verdict

**PASS -- the EVPI computation is CORRECT.** It matches the closed form, is zero
when the decision never changes, is positive when uncertainty spans the threshold,
and is non-negative always.

## Decision problem (known truth, dgp_decision.py)

Treat vs no-treat; net benefit (matching evpi.py's sign convention, log-scale
effect where more-negative = more beneficial):
  NB(treat, theta) = mcid - theta ;  NB(no_treat) = 0.
Posterior theta ~ Normal(mu, sigma^2), sigma = sqrt(se^2 + tau^2).

Closed form (ground truth):
  EVPI = sigma * ( phi(z) + z*Phi(z) - max(z,0) ),  z = (mcid - mu)/sigma
       = sigma * L(|z|),  L = unit normal loss integral, L(u)=phi(u)-u*Phi(-u).
This is the standard EVPI-for-a-normal-posterior result.

## Measured results (harness.py, n_sim up to 400k)

| Property tested | Result |
|---|---|
| EVPI = 0 when CI entirely one side of threshold (decision fixed) | MC_EVPI = 0.000 at |z|=6.7-10 (closed form 1e-13..1e-24). PASS |
| EVPI > 0 when uncertainty spans threshold | 0.077-0.097, all strictly positive. PASS |
| Matches closed form across 30-cell grid | max ABS error 2.0e-4; max rel error 3.7% only on tiniest near-zero cells (pure MC noise). PASS |
| EVPI >= 0 always (200 random cases) | 0/200 negative (formula also clamps max(0,.)). PASS |
| Linear scaling EVPI = sigma*phi(0)=0.39894*sigma at z=0 | sigma 0.1/0.2/0.4 -> 0.0398/0.0797/0.1594 vs 0.0399/0.0798/0.1596. PASS |

The ~3.7% max RELATIVE error appears only where the closed form is ~1e-4 (decision
essentially fixed); the ABSOLUTE agreement there is ~1e-5, i.e. Monte-Carlo noise,
not a formula error. The meaningful (absolute) agreement is tight everywhere.

## Findings

- `compute_evpi` is a correct, textbook Monte-Carlo EVPI:
  E[max over decisions] - max[E over decisions], with the right net-benefit
  parameterisation and a non-negativity clamp.
- The closed-form match confirms both the estimator AND the net-benefit sign
  convention (mcid - theta) are right; a sign error would have produced EVPI that
  is positive on the wrong side of the threshold -- it does not.
- No bug found in the EVPI core. (Scope: only compute_evpi was validated against a
  closed form. EVSI/EVPPI and the population scaler were not closed-form-checked.)

## Recommendation

Ship as-is. The EVPI engine is sound. Suggested follow-up (out of scope here):
add the same closed-form regression test to the repo's own suite, and extend
known-truth checks to EVPPI (single-parameter) which also has tractable cases.

## Files

- engine_note.md   -- which function is under test and why it is imported, not copied.
- dgp_decision.py  -- seeded known-truth decision problem + exact closed-form EVPI.
- harness.py       -- wires metavoi's own compute_evpi; measures all four properties.
- test_truth_recovery.py -- 5 assertions, all passing (relaxed MC tolerances).
