# MetaVoI — Value of Information from Meta-Analysis

**Date**: 2026-04-01
**Status**: Approved
**Target Journal**: Medical Decision Making
**Location**: `C:\Models\MetaVoI\`

## Summary

Browser-based tool that computes Value of Information (VoI) directly from meta-analysis output. Answers: "Is there enough evidence, or should we fund another trial?" No existing tool combines MA output with VoI — current tools (Sheffield SAVI, BCEA R package) require building full decision models from scratch. MetaVoI uses the MA posterior as the decision model.

## Architecture

- **Python engine** (`metavoi/`): Pure-function VoI computations, numpy/scipy only
- **Browser app** (`app/metavoi.html`): Single-file HTML with Plotly.js, ports Python logic to JS
- **Test suite** (`tests/`): pytest, 25+ tests, validated against BCEA R package
- **TruthCert**: Hash-linked provenance on all outputs

## Decision Model

Binary decision: treat (d=1) vs don't treat (d=0) based on whether the true treatment effect exceeds a Minimal Clinically Important Difference (MCID).

- **Net benefit under treatment**: `NB(d=1) = theta * WTP - cost_treatment` (simplified to: `NB = theta - mcid` when WTP is absorbed)
- **Optimal decision**: treat if `E[theta] > mcid`, don't treat otherwise
- **Decision uncertainty**: `P(wrong) = P(theta < mcid)` if treating, or `P(theta > mcid)` if not treating

## Statistical Methods

### 1. EVPI (Expected Value of Perfect Information)

The maximum amount a decision-maker would pay to eliminate ALL parameter uncertainty.

```
EVPI = E_theta[max(NB(d=1, theta), NB(d=0, theta))] - max(E_theta[NB(d=1, theta)], E_theta[NB(d=0, theta)])
```

**Computation**: Monte Carlo (10,000 draws from posterior):
1. Draw `theta_i ~ N(theta_hat, se^2 + tau^2)` (predictive distribution)
2. For each draw, compute `max(NB(d=1, theta_i), NB(d=0, theta_i))`
3. `EVPI = mean(max values) - max(mean NB under d=1, mean NB under d=0)`

**Population EVPI**: `EVPI_pop = EVPI * population * horizon_years * discount_factor`

Where `discount_factor = sum(1/(1+r)^t for t in 0..horizon-1)`, r = discount rate (default 3.5%).

### 2. EVPPI (Expected Value of Partial Perfect Information)

How much of the total EVPI is attributable to uncertainty in theta vs tau^2.

**For theta** (treatment effect):
- Outer loop: draw `theta_j` from prior
- Inner loop: for each `theta_j`, compute optimal decision
- `EVPPI_theta = E_theta[max(NB(d, theta))] - max(E_theta[NB(d, theta)])`

**For tau^2** (heterogeneity):
- Same structure but condition on tau^2 values
- Quantifies: "Would knowing the true heterogeneity change our decision?"

**Simplified approach (GAM regression)**: Strong & Oakley (2014) method using GAM to estimate conditional expectations. More efficient than nested Monte Carlo.

### 3. EVSI (Expected Value of Sample Information)

"What if we ran a new trial with N patients?"

**Computation** (moment matching, Ades et al. 2004):
1. A trial of size N with known sigma^2 (within-study variance) produces a new estimate `theta_new ~ N(theta_true, sigma^2/N)`
2. Updated posterior (Bayesian update): 
   - `precision_post = precision_prior + N/sigma^2`
   - `theta_post = (precision_prior * theta_prior + (N/sigma^2) * theta_new) / precision_post`
3. For each simulated `(theta_true, theta_new)` pair, compute updated optimal decision
4. `EVSI(N) = E[max(NB under updated posterior)] - max(E[NB under current posterior])`

**Inputs needed**: Within-study variance estimate (can default to median vi from MA studies).

### 4. Optimal Trial Size

Grid search over N in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
- `Net_benefit(N) = EVSI_pop(N) - cost_per_patient * N`
- Optimal N* = argmax(Net_benefit)
- Report: N*, EVSI at N*, net benefit, break-even N (where EVSI_pop = cost)

### 5. GRADE Bridge

Map GRADE certainty to VoI interpretation:
| GRADE | Typical P(wrong) | Interpretation |
|-------|-------------------|---------------|
| High | < 5% | Low EVPI — additional trials unlikely to change decision |
| Moderate | 5-20% | Moderate EVPI — consider targeted trials |
| Low | 20-40% | High EVPI — strong case for new trial |
| Very Low | > 40% | Very high EVPI — decision highly uncertain |

Computed dynamically from the posterior, not from GRADE labels.

## Input Specification

### Option A: Study-level data
```json
{
  "studies": [
    {"yi": -0.71, "vi": 0.05, "label": "Study 1"},
    {"yi": -0.43, "vi": 0.08, "label": "Study 2"}
  ],
  "mcid": 0.2,
  "population": 100000,
  "horizon_years": 10,
  "cost_per_patient": 5000,
  "discount_rate": 0.035,
  "within_study_var": null
}
```

### Option B: Summary MA results
```json
{
  "theta": -0.71,
  "se": 0.18,
  "tau2": 0.31,
  "k": 13,
  "mcid": 0.2,
  "population": 100000,
  "horizon_years": 10,
  "cost_per_patient": 5000,
  "discount_rate": 0.035,
  "within_study_var": 0.05
}
```

## Output Specification

```json
{
  "decision": {
    "current_optimal": "treat",
    "p_wrong": 0.12,
    "expected_nb_treat": 0.51,
    "expected_nb_no_treat": 0.0
  },
  "evpi": {
    "per_decision": 0.032,
    "population": 28400000,
    "currency_note": "Units match net benefit scale"
  },
  "evppi": {
    "theta": 0.028,
    "tau2": 0.004,
    "dominant_parameter": "theta",
    "theta_fraction": 0.875
  },
  "evsi": {
    "curve": [
      {"n": 100, "evsi": 0.008, "evsi_pop": 7100000, "net_benefit": 6600000},
      {"n": 500, "evsi": 0.022, "evsi_pop": 19500000, "net_benefit": 17000000}
    ],
    "optimal_n": 2000,
    "optimal_evsi_pop": 26800000,
    "optimal_net_benefit": 16800000,
    "breakeven_n": 8500
  },
  "grade_bridge": {
    "implied_certainty": "Moderate",
    "p_wrong": 0.12,
    "recommendation": "Moderate EVPI — targeted trial of ~2000 patients would maximize value"
  },
  "certification": {
    "input_hash": "a3f8...",
    "n_simulations": 10000,
    "seed": 42,
    "status": "PASS"
  }
}
```

## Browser App Tabs (5)

### Tab 1: Input
- Toggle: study-level data vs summary results
- Paste/type study data (yi, vi, label) or summary (theta, se, tau2, k)
- Decision parameters: MCID, population size, time horizon, cost per patient, discount rate
- Built-in examples: BCG vaccine, statin therapy, SGLT2i in HF
- "Compute VoI" button

### Tab 2: Decision Space
- Posterior density plot (normal, shaded regions for treat/don't-treat)
- P(wrong decision) prominently displayed
- Net benefit comparison bar chart
- GRADE certainty badge (computed from P(wrong))

### Tab 3: EVPI Analysis
- Population EVPI in currency units (or "NB units" if no WTP specified)
- EVPPI tornado chart: horizontal bars showing theta vs tau2 contribution
- Interpretation text: "X% of decision uncertainty comes from the treatment effect estimate"

### Tab 4: Trial Planning (EVSI)
- EVSI curve: line plot of EVSI_pop vs N, with cost line overlay
- Optimal N marker on curve
- Break-even N marker
- Net benefit curve (EVSI - cost)
- Summary verdict: "A trial of N=2000 would provide value of $X per patient-year"

### Tab 5: Report & Certify
- Plain-text structured report (copy-paste for manuscripts)
- TruthCert JSON bundle with hashes
- Methods paragraph generator (for paper supplements)

## Visualizations (4 Plotly charts)

1. **Posterior density**: Normal curve with MCID threshold line, shaded P(wrong)
2. **EVPPI tornado**: Horizontal bars (theta, tau2) showing parameter contributions
3. **EVSI curve**: Line plot (N vs EVSI_pop) with cost overlay and optimal/breakeven markers
4. **Net benefit curve**: EVSI_pop - cost(N), highlighting positive region

## Test Coverage

### Python engine tests (25+)
- EVPI: zero when P(wrong)=0, positive when uncertain, bounded by max(NB)
- EVPI: increases as SE increases (more uncertainty = more value of information)
- EVPPI: sum of per-parameter <= total EVPI
- EVPPI: theta dominant when tau2 is small relative to SE
- EVSI: monotonically increasing with N (more patients = more info)
- EVSI: bounded by EVPI (can't exceed perfect information value)
- EVSI: approaches EVPI as N -> infinity
- Optimal N: exists when cost > 0, equals infinity when cost = 0
- Break-even N: exists when EVSI_pop(N) can exceed cost(N)
- GRADE bridge: High when P(wrong) < 0.05, Very Low when > 0.40
- Population EVPI: scales linearly with population
- Discount factor: decreases with higher discount rate
- BCG example: known values from published VoI literature
- Edge cases: tau2=0 (no heterogeneity), k=1 (single study), very large SE

### Browser tests (Selenium, 15+)
- Tab navigation, data input, built-in example loading
- Compute button produces results
- All 4 charts render
- Report export works
- Accessibility (aria-labels, keyboard nav)

## Built-in Examples

### 1. BCG Vaccine (from Colditz et al. 1994)
- 13 RCTs, logRR, theta=-0.71, tau2=0.31
- MCID=-0.2 (20% relative risk reduction)
- Population: 1M (TB-endemic countries), horizon: 20y

### 2. Statins for Primary Prevention
- 5 RCTs, logOR, theta=-0.25, tau2=0.02
- MCID=-0.1
- Population: 10M, horizon: 10y

### 3. SGLT2i in Heart Failure
- 4 RCTs, logHR, theta=-0.30, tau2=0.01
- MCID=-0.15
- Population: 500K, horizon: 5y

## File Structure

```
C:\Models\MetaVoI\
  metavoi/
    __init__.py
    models.py          # Dataclasses: VoIInput, VoIResult, EVSIPoint
    posterior.py        # Posterior distribution from MA (predictive dist)
    evpi.py             # EVPI computation (Monte Carlo)
    evppi.py            # EVPPI computation (per-parameter)
    evsi.py             # EVSI computation (preposterior)
    optimal.py          # Optimal N search, break-even
    grade_bridge.py     # P(wrong) -> GRADE mapping
    pipeline.py         # run_voi() orchestrator
    certifier.py        # TruthCert certification
  tests/
    conftest.py         # Fixtures (BCG, statin, SGLT2i)
    test_posterior.py
    test_evpi.py
    test_evppi.py
    test_evsi.py
    test_optimal.py
    test_grade_bridge.py
    test_pipeline.py
  app/
    metavoi.html        # Single-file browser app
  data/
    bcg.json
    statin.json
    sglt2i.json
  docs/
    superpowers/specs/  # This file
    superpowers/plans/  # Implementation plan
  setup.py
  README.md
  LICENSE
  CITATION.cff
```

## Out of Scope

- Full probabilistic sensitivity analysis (PSA) — that's a decision model, not MA-driven
- Multi-arm decision models (treat A vs B vs C) — binary only for v1
- Network meta-analysis VoI — future extension
- Cost-effectiveness plane / CEAC — would require full economic model
- WTP threshold uncertainty — fixed WTP for v1
