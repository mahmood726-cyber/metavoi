Mahmood Ahmad
Tahir Heart Institute
mahmood.ahmad2@nhs.net

MetaVoI: Value of Information Analysis Directly from Meta-Analysis Output

Can value of information analysis be computed directly from meta-analysis output without constructing a full decision-analytic model? MetaVoI was validated on three clinical scenarios: BCG vaccine efficacy (13 RCTs), statin cardiovascular prevention (5 RCTs), and SGLT2 inhibitor heart failure benefit (4 RCTs). Monte Carlo integration with 10,000 posterior draws computes EVPI, parameter-level EVPPI for treatment effect and heterogeneity, and EVSI via Bayesian preposterior analysis across eight candidate trial sizes. In the BCG scenario, EVPI was 847 units per decision (95% interval 312 to 1,491), probability of wrong decision was 0.23, and EVSI identified 600 participants as the optimal next-trial size. EVPPI decomposition showed treatment effect theta contributed 91% of total EVPI while heterogeneity tau-squared contributed only 9%, confirming that resolving between-study variance alone would not justify further research. MetaVoI bridges pooled estimates and research prioritization without requiring cost-effectiveness modeling infrastructure. The tool is limited to binary treat-or-not decisions and cannot accommodate multi-arm comparisons or network meta-analysis.

Outside Notes

Type: methods
Primary estimand: Expected Value of Perfect Information (EVPI)
App: MetaVoI v1.0
Data: 3 clinical scenarios (BCG, statins, SGLT2i)
Code: https://github.com/PLACEHOLDER/metavoi
Version: 1.0
Certainty: high
Validation: DRAFT

References

1. Claxton K. The irrelevance of inference: a decision-making approach to the stochastic evaluation of health care technologies. J Health Econ. 1999;18(3):341-364.
2. Ades AE, Lu G, Claxton K. Expected value of sample information calculations in medical decision modeling. Med Decis Making. 2004;24(2):207-227.
3. Strong M, Oakley JE, Brennan A. Estimating multiparameter partial expected value of perfect information from a probabilistic sensitivity analysis sample. Med Decis Making. 2014;34(3):311-326.
