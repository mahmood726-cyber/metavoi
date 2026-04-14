# MetaVoI — Value of Information from Meta-Analysis

**World-first** browser-based tool that computes Value of Information (VoI) directly from meta-analysis output. Answers: *"Is there enough evidence, or should we fund another trial?"*

## Quick Start

### Browser App (no installation)
Open `app/metavoi.html` in any modern browser. Load a built-in example or paste your own data.

### Python Engine
```bash
pip install -e .
```

```python
from metavoi.models import VoIInput
from metavoi.pipeline import run_voi

inp = VoIInput(theta=-0.71, se=0.18, tau2=0.31, k=13, mcid=-0.2,
               population=1_000_000, horizon_years=20, cost_per_patient=500)
result = run_voi(inp)
print(f"EVPI: {result.evpi:.4f}")
print(f"Optimal next trial: N={result.optimal_n}")
print(f"GRADE certainty: {result.implied_certainty}")
```

## What It Computes

| Metric | Description |
|--------|-------------|
| **EVPI** | Maximum value of eliminating ALL parameter uncertainty |
| **EVPPI** | Per-parameter VoI — is it theta or tau2 driving uncertainty? |
| **EVSI(N)** | Value of running a new trial with N patients |
| **Optimal N** | Trial size that maximizes net benefit (EVSI - cost) |
| **GRADE bridge** | Maps P(wrong decision) to GRADE certainty levels |

## Built-in Examples

1. **BCG Vaccine** — 13 RCTs, high heterogeneity, TB-endemic population
2. **Statins** — 5 RCTs, low heterogeneity, primary prevention
3. **SGLT2i in HF** — 4 RCTs, landmark cardiology trials

## Validation

- 43 pytest tests (posterior, EVPI, EVPPI, EVSI, optimal, GRADE, pipeline)
- Monte Carlo convergence verified (10,000 simulations, seed=42)
- EVSI bounded by EVPI (mathematical property)
- GRADE thresholds calibrated to published certainty criteria

## Citation

Mahmood Ahmad. MetaVoI: Value of Information from Meta-Analysis. 2026.

## License

MIT
