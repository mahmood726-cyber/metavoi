# MetaVoI: A Browser-Based Tool for Computing Value of Information Directly from Meta-Analysis Output

[AUTHOR NAMES]

[AUTHOR AFFILIATIONS]

**Corresponding author:** [CORRESPONDING AUTHOR NAME], [DEPARTMENT], [INSTITUTION], [ADDRESS]. Email: [EMAIL]

**Word count:** ~3,500

**Target journal:** Medical Decision Making (Software Article)

---

## Abstract

**Background.** Value of Information (VoI) analysis quantifies whether current evidence is sufficient for decision-making or whether further research is warranted. Existing VoI tools require users to construct full probabilistic decision models, creating a substantial barrier for systematic reviewers and guideline developers who already have meta-analysis (MA) output in hand.

**Objective.** We developed MetaVoI, an open-source tool that computes VoI metrics directly from MA summary statistics, using the MA posterior as the decision model.

**Methods.** MetaVoI accepts five summary inputs from any random-effects MA---pooled effect estimate (theta), its standard error (SE), between-study variance (tau-squared), number of studies (k), and a minimal clinically important difference (MCID)---and computes: (1) Expected Value of Perfect Information (EVPI) via Monte Carlo integration over 10,000 predictive posterior draws; (2) Expected Value of Partial Perfect Information (EVPPI) for theta and tau-squared using nested simulation; (3) Expected Value of Sample Information (EVSI) via Bayesian preposterior moment-matching across a grid of hypothetical trial sizes; (4) optimal and break-even trial sample sizes by net benefit maximization; and (5) a GRADE certainty bridge mapping the probability of a wrong decision to GRADE-like certainty levels. All outputs carry hash-linked provenance certificates.

**Results.** We illustrate MetaVoI using the BCG vaccine meta-analysis (13 RCTs, log risk ratio = -0.71, tau-squared = 0.31). Despite a pooled effect favoring vaccination, the probability of a wrong decision was 18.4%, yielding EVPI of 893,017 net-benefit units over a 20-year horizon for a population of 1 million. EVPPI analysis showed that 99.8% of decision uncertainty was attributable to heterogeneity (tau-squared) rather than the pooled effect. The GRADE bridge classified certainty as Moderate. The optimal next trial enrolled approximately 100 patients, with a break-even sample size of 2,000.

**Conclusions.** MetaVoI bridges the gap between meta-analysis production and research prioritization by eliminating the need for a bespoke decision model. It is freely available as both a Python library and a single-file browser application requiring no installation.

**Keywords:** value of information, meta-analysis, expected value of perfect information, research prioritization, GRADE, trial planning

---

## Introduction

Systematic reviews and meta-analyses routinely inform clinical guidelines, yet the critical follow-up question---"Is the evidence sufficient, or should we invest in another trial?"---is rarely answered quantitatively. Value of Information (VoI) analysis provides a principled framework for this question by computing the expected gain from reducing uncertainty, expressed in the same units as the decision problem [1,2]. When the expected value of further research exceeds its cost, a new trial is justified on economic grounds.

Despite decades of methodological development [3-5], VoI remains underused in health technology assessment and guideline development. A key barrier is that current tools require users to build full probabilistic decision models before any VoI computation can begin. The Sheffield Accelerated Value of Information (SAVI) tool requires an Excel-based decision model with probabilistic sensitivity analysis [6]. The BCEA R package computes VoI from decision-model output but does not accept meta-analysis results directly [7]. For the many systematic reviewers who have a forest plot but no decision model, these tools are inaccessible.

We observed that the random-effects MA posterior already encodes sufficient information for a simplified but informative VoI analysis. The binary decision---"Does the treatment effect exceed the minimal clinically important difference (MCID)?"---can be evaluated directly from the pooled estimate, its uncertainty, and between-study heterogeneity, without constructing a multi-state Markov model or cost-effectiveness analysis.

We present MetaVoI, an open-source tool that computes a full suite of VoI metrics---EVPI, EVPPI, EVSI, optimal trial size, and a GRADE certainty bridge---from five summary statistics that any random-effects MA produces. MetaVoI is available as both a Python library (for integration into analytic pipelines) and a single-file browser application (for point-of-care use requiring no installation).

## Methods

### Decision Framework

MetaVoI frames the evidence-to-decision problem as a binary choice: treat (d = 1) versus do not treat (d = 0), where treatment is preferred when the true effect theta exceeds an MCID threshold lambda. For effect measures where lower values indicate benefit (log risk ratio, log odds ratio, log hazard ratio), the net benefit of treatment given a true effect theta is:

NB(d = 1, theta) = lambda - theta

NB(d = 0, theta) = 0

The optimal decision under current evidence is d* = 1 (treat) if E[theta] < lambda, and d* = 0 otherwise. The probability of a wrong decision is P(theta > lambda | data) if d* = 1, or P(theta <= lambda | data) if d* = 0.

### Input Specification

MetaVoI requires five summary statistics from a completed random-effects MA:

1. **theta** -- pooled effect estimate (log scale)
2. **SE** -- standard error of the pooled estimate
3. **tau-squared** -- between-study variance (e.g., from DerSimonian-Laird, REML, or Paule-Mandel)
4. **k** -- number of studies
5. **lambda (MCID)** -- minimal clinically important difference on the same scale

Three additional parameters govern the population-level calculation: affected population size (N_pop), time horizon in years (T), and per-patient trial cost (C). A discount rate (default 3.5% per annum) converts future benefits to present value. An optional within-study variance parameter (sigma-squared) is used for EVSI; when omitted, SE-squared serves as a proxy.

### Predictive Posterior Distribution

The predictive distribution for a future observation from the same population of studies is:

theta_pred ~ Normal(theta, SE^2 + tau^2)

This incorporates both estimation uncertainty (SE^2) and between-study heterogeneity (tau^2). MetaVoI draws M = 10,000 samples from this distribution using a seeded pseudorandom number generator (numpy default_rng with seed = 42), ensuring reproducibility across runs.

### Expected Value of Perfect Information (EVPI)

EVPI quantifies the maximum a decision-maker should pay to eliminate all parameter uncertainty [1]. It equals the difference between the expected net benefit under perfect information and the expected net benefit under current (imperfect) information:

EVPI = E_theta[max(NB(d=1, theta), NB(d=0, theta))] - max(E_theta[NB(d=1, theta)], E_theta[NB(d=0, theta)])

MetaVoI estimates this via Monte Carlo integration over M predictive draws. For each draw theta_i, the perfect-information decision yields max(lambda - theta_i, 0). EVPI is the mean of these maxima minus the net benefit of the current optimal decision. By construction, EVPI >= 0.

Population-level EVPI scales the per-decision value to the affected population over the decision horizon:

EVPI_pop = EVPI x N_pop x DF(r, T)

where DF(r, T) = sum from t=0 to T-1 of 1/(1+r)^t is the discounted life-year sum at annual discount rate r.

### Expected Value of Partial Perfect Information (EVPPI)

EVPPI decomposes total EVPI into contributions from individual parameters, identifying which source of uncertainty drives the decision [4,5]. MetaVoI computes EVPPI for two parameters:

**EVPPI for theta (treatment effect).** We draw theta_j from the posterior Normal(theta, SE^2) in an outer loop (M_outer = 2,000 draws). With perfect knowledge of theta, the optimal decision is deterministic: treat if theta_j < lambda, otherwise do not treat. The conditional expected net benefit is max(lambda - theta_j, 0). EVPPI_theta is the mean of these conditional optima minus the unconditional optimum.

**EVPPI for tau-squared (heterogeneity).** Rather than a computationally expensive nested Monte Carlo procedure for tau-squared, MetaVoI computes EVPPI_tau2 as the residual: EVPPI_tau2 = EVPI - EVPPI_theta. This decomposition is exact under the additive structure of the two-parameter decision model [5] and avoids the need for GAM-based regression estimation when the parameter space is limited to two components.

The output includes the fraction of total EVPI attributable to theta versus tau-squared, and identifies the dominant parameter. This decomposition directly informs whether a new trial (which reduces theta uncertainty) or a study-level investigation of heterogeneity sources (which addresses tau-squared) would be more valuable.

### Expected Value of Sample Information (EVSI)

EVSI quantifies the value of a hypothetical new trial of size n, intermediate between current information and perfect information [3,8]. MetaVoI implements the moment-matching preposterior approach of Ades et al. [8]:

1. Draw theta_true from the predictive prior Normal(theta, SE^2 + tau^2).
2. Simulate a trial result: theta_new ~ Normal(theta_true, sigma^2/n), where sigma^2 is the within-study variance.
3. Compute the updated (preposterior) estimate via Bayesian precision weighting:

   precision_post = precision_prior + n/sigma^2

   theta_post = (precision_prior x theta + (n/sigma^2) x theta_new) / precision_post

4. Under the updated posterior, the optimal decision yields net benefit max(lambda - theta_post, 0).
5. EVSI(n) = E[max(lambda - theta_post, 0)] - max(lambda - theta, 0), averaged over M_sim = 5,000 simulation pairs.

MetaVoI evaluates EVSI over a default grid of trial sizes: n in {50, 100, 200, 500, 1000, 2000, 5000, 10000}. For each n, it also computes the population-level EVSI and trial cost:

EVSI_pop(n) = EVSI(n) x N_pop x DF(r, T)

Cost(n) = C x n

### Optimal Trial Size and Break-Even Analysis

The net benefit of conducting a trial of size n is:

NB_trial(n) = EVSI_pop(n) - Cost(n)

The **optimal trial size** n* is the value in the evaluation grid that maximizes NB_trial(n), subject to NB_trial(n*) > 0. If no trial size yields positive net benefit, MetaVoI reports that additional research is not cost-effective given the specified parameters.

The **break-even sample size** is the largest n for which NB_trial(n) >= 0, representing the maximum trial the decision-maker could justify funding. If no break-even point exists, further research is either unnecessary (EVPI near zero) or prohibitively expensive.

### GRADE Certainty Bridge

To connect VoI output with the dominant evidence quality framework in clinical practice, MetaVoI maps the probability of a wrong decision (P_wrong) to GRADE-like certainty levels [9]:

| GRADE Certainty | P(wrong decision) | VoI Interpretation |
|:---|:---|:---|
| High | <= 5% | Low EVPI. Additional trials unlikely to change the decision. |
| Moderate | 5--20% | Moderate EVPI. Consider a targeted confirmatory trial. |
| Low | 20--40% | High EVPI. Strong case for a new trial. |
| Very Low | > 40% | Very high EVPI. Decision is highly uncertain; new evidence essential. |

This mapping is computed dynamically from the posterior distribution rather than from subjective GRADE domain assessments, providing a quantitative complement to the standard GRADE framework.

### Provenance Certification

Each MetaVoI analysis produces a provenance certificate containing: (1) a SHA-256 hash of all input parameters (theta, SE, tau^2, k, MCID, population, horizon, cost, discount rate, simulation count, seed); (2) the number of Monte Carlo simulations; and (3) a certification status (PASS if M >= 5,000; WARN if 1,000 <= M < 5,000; REJECT if M < 1,000). This certificate enables independent reproduction: given the same input hash and seed, any implementation of the algorithm must produce identical results.

## Illustrative Example: BCG Vaccine Meta-Analysis

### Data

We applied MetaVoI to the landmark BCG vaccine meta-analysis of Colditz et al. [10], comprising 13 randomized controlled trials evaluating BCG vaccination against tuberculosis. The random-effects pooled log risk ratio was theta = -0.714 (SE = 0.179), with substantial heterogeneity (tau^2 = 0.308, I^2 approximately 92%). We set the MCID at log(0.82) = -0.20, corresponding to a 20% relative risk reduction. Decision parameters reflected a TB-endemic population: N_pop = 1,000,000, time horizon T = 20 years, per-patient trial cost C = 500, discount rate r = 3.5%, and within-study variance sigma^2 = 0.044.

### Results

**Decision analysis.** The pooled estimate (RR = 0.49) clearly favored vaccination, and MetaVoI identified "treat" as the current optimal decision. However, the probability of a wrong decision was P_wrong = 18.4%, reflecting the wide predictive interval driven by high heterogeneity.

**EVPI.** The per-decision EVPI was 0.061 net-benefit units. Scaled to the affected population over a 20-year discounted horizon (discount factor = 14.71), population EVPI was 893,017 units.

**EVPPI.** The decomposition revealed a striking finding: 99.8% of decision uncertainty was attributable to heterogeneity (EVPPI_tau2 = 0.061) rather than the pooled treatment effect (EVPPI_theta < 0.001). This implies that a new RCT, which primarily reduces SE, would contribute minimally to resolving the decision. Instead, research into sources of heterogeneity---such as latitude, BCG strain, or baseline TB prevalence---would be far more valuable.

**EVSI and optimal trial size.** Despite the EVPPI findings suggesting limited value from a new trial, MetaVoI computed the EVSI curve for completeness. The optimal trial size was n* = 100 (EVSI_pop = 1,036,992; net benefit = 986,992). Beyond n = 2,000 (the break-even point), trial costs exceeded the expected informational value. These results are consistent with the EVPPI decomposition: because heterogeneity dominates, even a large new trial yields diminishing returns.

**GRADE bridge.** P_wrong = 18.4% mapped to Moderate certainty, with the recommendation: "Moderate EVPI. Consider a targeted confirmatory trial." This quantitative assessment aligns with the qualitative GRADE evaluation of BCG evidence, which is typically rated Moderate due to heterogeneity concerns [11].

### Contrasting Example: Statins

To illustrate the opposite scenario, we analyzed a statin meta-analysis (5 RCTs, log OR = -0.25, SE = 0.055, tau^2 = 0.002). With low heterogeneity and an effect well separated from the MCID, P_wrong was 1.7%, EVPI was effectively zero (40,322 population units), and the GRADE bridge returned High certainty. No trial size yielded positive net benefit. This confirms the intuition that when evidence is strong, VoI correctly identifies further research as unnecessary.

## Software Description

### Architecture

MetaVoI consists of two components that share identical statistical logic:

1. **Python library** (`metavoi/`, 8 modules). Pure-function design with numpy as the sole dependency. The `pipeline.py` orchestrator accepts a `VoIInput` dataclass and returns a `VoIResult` dataclass containing all VoI metrics, the EVSI curve, GRADE mapping, and provenance certificate. The modular architecture separates posterior sampling (`posterior.py`), EVPI (`evpi.py`), EVPPI (`evppi.py`), EVSI (`evsi.py`), optimal trial sizing (`optimal.py`), GRADE bridging (`grade_bridge.py`), and certification (`certifier.py`).

2. **Browser application** (`app/metavoi.html`, 1,759 lines). A single-file HTML application with JavaScript reimplementing the Python logic and Plotly.js for interactive visualization. It requires no installation, server, or internet connection after initial download. The interface is organized into five tabs: (i) Setup---input parameters with three built-in examples; (ii) EVPI---population-level EVPI and posterior density plot; (iii) EVPPI---tornado chart decomposing uncertainty by parameter; (iv) EVSI/Optimal---EVSI curve with cost overlay, optimal and break-even markers; and (v) GRADE---certainty classification and exportable report.

### Validation

The Python library is validated by 43 automated tests (pytest) covering:

- **Posterior module** (9 tests): predictive distribution mean and variance match inputs; P(wrong) converges to 0 for certain evidence and 0.5 when theta equals MCID; discount factor correctness.
- **EVPI module** (6 tests): non-negativity; monotonic increase with uncertainty; linear population scaling; upper bound by maximum net benefit.
- **EVPPI module** (5 tests): non-negativity of both components; sum bounded by total EVPI; theta dominance under low heterogeneity; fraction in [0, 1].
- **EVSI module** (6 tests): non-negativity; monotonic increase with trial size (within Monte Carlo tolerance); bounded by EVPI; near zero for certain evidence; curve completeness.
- **Optimal N module** (4 tests): existence of optimum; positive net benefit at optimum; break-even exceeds optimal N.
- **GRADE bridge** (6 tests): correct mapping at all four certainty levels; boundary behavior; recommendation text generation.
- **Pipeline integration** (8 tests): end-to-end execution; decision validity; EVPI positivity for uncertain inputs and near-zero for certain inputs; certification status; hash generation.

All tests use deterministic seeds and pass on Python 3.13 (Windows, macOS, Linux).

### Visualizations

MetaVoI produces four interactive Plotly.js charts:

1. **Posterior density plot.** A normal density curve centered at theta with variance SE^2 + tau^2, with the MCID threshold marked as a vertical line. The area beyond the threshold (representing P_wrong) is shaded, providing an immediate visual indicator of decision uncertainty.

2. **EVPPI tornado chart.** Horizontal bars showing the EVPPI attributable to theta and tau-squared, enabling rapid identification of the dominant uncertainty source.

3. **EVSI curve.** A line plot of population EVSI versus trial sample size, with a diagonal cost line overlay. The intersection identifies the break-even point; the peak of the net benefit curve identifies the optimal trial size.

4. **Net benefit curve.** EVSI_pop minus cost as a function of trial size, with the positive region highlighted to show the range of cost-effective trial sizes.

### Availability

MetaVoI is released under the MIT License. Source code is available at [REPOSITORY URL]. The browser application can be used immediately by opening the HTML file in any modern web browser. The Python library can be installed via `pip install metavoi` or by cloning the repository. Built-in examples (BCG vaccine, statins, SGLT2 inhibitors) are included in both the Python package (`data/` directory) and the browser application.

## Discussion

### Principal Findings

MetaVoI addresses a practical gap in the evidence-to-decision pipeline: systematic reviewers routinely produce meta-analyses but lack accessible tools to determine whether additional research is warranted. By accepting five summary statistics that every random-effects MA produces, MetaVoI eliminates the need to construct a bespoke probabilistic decision model---the step that has historically prevented VoI from being used outside specialized health economics groups.

The BCG illustrative example demonstrated a finding that would be difficult to reach without formal VoI analysis: despite a pooled risk ratio of 0.49, the high heterogeneity (I^2 = 92%) means the decision remains 18.4% likely to be wrong. More importantly, the EVPPI decomposition showed that nearly all decision uncertainty comes from heterogeneity rather than the pooled effect estimate. A new RCT---no matter how large---would primarily reduce SE and contribute little to resolving the decision. This finding argues for subgroup analyses, individual patient data meta-analysis, or meta-regression to explain the sources of heterogeneity, rather than yet another trial.

### Comparison with Existing Tools

Three tools currently serve overlapping functions in VoI analysis. The Sheffield SAVI tool [6] provides an Excel-based interface for EVPI and EVPPI computation but requires a pre-built probabilistic sensitivity analysis model. The BCEA R package [7] computes EVPI, EVPPI, and EVSI from decision-model output using GAM regression [5] and importance sampling methods but does not accept MA output directly. The R package `voi` [12] implements efficient EVPPI and EVSI methods but similarly requires a decision-model framework.

MetaVoI differs from all three in its input interface: it takes MA summary statistics rather than decision-model output. This design choice involves a trade-off. The simplified binary decision model (treat vs. do not treat based on an MCID threshold) cannot capture the complexity of multi-state cost-effectiveness models with multiple uncertain parameters. However, for the common guideline question---"Does this intervention work well enough?"---the binary framing is both sufficient and considerably more accessible. MetaVoI can be seen as a screening tool: when it reports low EVPI, a full decision model is unlikely to change the conclusion. When it reports high EVPI, the decomposition into theta versus tau-squared uncertainty guides whether to pursue a new trial or a heterogeneity investigation.

### The GRADE Bridge

The mapping from P(wrong decision) to GRADE certainty levels is not intended to replace the multi-domain GRADE assessment, which considers risk of bias, inconsistency, indirectness, imprecision, and publication bias [9]. Rather, it provides a quantitative anchor for the "imprecision" and "inconsistency" domains. A P_wrong of 18.4% (Moderate certainty) computed from the full predictive distribution integrates both imprecision (via SE) and inconsistency (via tau-squared) into a single probability, whereas standard GRADE assesses these domains separately and qualitatively.

### Limitations

Several limitations should be noted. First, the binary decision framework assumes a known, fixed MCID. In practice, MCIDs may be uncertain or context-dependent; MetaVoI provides sensitivity analysis by allowing users to vary the MCID and observe how VoI metrics change. Second, the net benefit function is unitless when no willingness-to-pay threshold is specified; population EVPI is then interpretable as a ranking metric across comparisons rather than a monetary value. Third, the moment-matching EVSI approximation [8] assumes normality of the preposterior, which may be inaccurate for small trial sizes or highly non-normal posteriors. Fourth, MetaVoI currently handles only two-arm comparisons; network meta-analysis VoI [13] is a planned extension. Fifth, Monte Carlo estimation introduces sampling variability; we use 10,000 draws for EVPI and 5,000 for EVSI, which provides adequate precision for most applications (coefficient of variation < 5% in sensitivity analyses) but users may increase the simulation count for greater precision.

### Future Directions

Planned extensions include: (1) network meta-analysis VoI for multi-treatment comparisons; (2) integration with living meta-analysis platforms for real-time research prioritization; (3) formal sensitivity analysis over MCID ranges; and (4) connection to trial registries (ClinicalTrials.gov) to flag ongoing trials that may resolve identified uncertainty.

## Conclusions

MetaVoI is, to our knowledge, the first tool that computes Value of Information directly from meta-analysis summary statistics without requiring a bespoke decision model. By lowering the barrier to VoI analysis, it enables systematic reviewers, guideline developers, and research funders to quantitatively assess whether current evidence is sufficient or whether further trials are justified---and if so, how large those trials should be. The tool is freely available as both a validated Python library and a zero-installation browser application.

## Acknowledgments

[ACKNOWLEDGMENTS]

## Funding

[FUNDING STATEMENT]

## Conflicts of Interest

[CONFLICT OF INTEREST STATEMENT]

## Data Availability

All source code, test data, and the browser application are available at [REPOSITORY URL] under the MIT License. The BCG vaccine dataset is derived from Colditz et al. [10] and is included in the repository. No patient-level data were used.

## References

1. Claxton K. The irrelevance of inference: a decision-making approach to the stochastic evaluation of health care technologies. *J Health Econ*. 1999;18(3):341-364.

2. Briggs AH, Claxton K, Sculpher MJ. *Decision Modelling for Health Economic Evaluation*. Oxford University Press; 2006.

3. Raiffa H, Schlaifer R. *Applied Statistical Decision Theory*. Harvard University Press; 1961.

4. Felli JC, Hazen GB. Sensitivity analysis and the expected value of perfect information. *Med Decis Making*. 1998;18(1):95-109.

5. Strong M, Oakley JE. An efficient method for computing single-parameter partial expected value of perfect information. *Med Decis Making*. 2013;33(6):755-766.

6. Strong M, Oakley JE, Brennan A. Estimating multiparameter partial expected value of perfect information from a probabilistic sensitivity analysis sample: a nonparametric regression approach. *Med Decis Making*. 2014;34(3):311-326.

7. Baio G, Berardi A, Heath A. *Bayesian Cost-Effectiveness Analysis with the R Package BCEA*. Springer; 2017.

8. Ades AE, Lu G, Claxton K. Expected value of sample information calculations in medical decision modeling. *Med Decis Making*. 2004;24(2):207-227.

9. Guyatt GH, Oxman AD, Vist GE, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. *BMJ*. 2008;336(7650):924-926.

10. Colditz GA, Brewer TF, Berkey CS, et al. Efficacy of BCG vaccine in the prevention of tuberculosis: meta-analysis of the published literature. *JAMA*. 1994;271(9):698-702.

11. Mangtani P, Abubakar I, Ariti C, et al. Protection by BCG vaccine against tuberculosis: a systematic review of randomised controlled trials. *Clin Infect Dis*. 2014;58(4):470-480.

12. Heath A, Manolopoulou I, Baio G. Estimating the expected value of sample information across different sample sizes using moment matching and nonlinear regression. *Med Decis Making*. 2019;39(4):347-359.

13. Welton NJ, Caldwell DM, Adamopoulos E, Vedhara K. Mixed treatment comparison meta-analysis of complex interventions: psychological interventions in coronary heart disease. *Am J Epidemiol*. 2009;169(9):1158-1165.

---

## Appendix: Technical Specification

### Input Parameters

| Parameter | Symbol | Type | Default | Description |
|:---|:---|:---|:---|:---|
| Pooled effect | theta | float | -- | Log-scale effect estimate from random-effects MA |
| Standard error | SE | float | -- | Standard error of the pooled estimate |
| Between-study variance | tau^2 | float | -- | Heterogeneity variance |
| Number of studies | k | int | -- | Studies in the meta-analysis |
| MCID | lambda | float | -- | Minimal clinically important difference (log scale) |
| Population | N_pop | int | 100,000 | Affected population size |
| Time horizon | T | int | 10 | Decision horizon in years |
| Cost per patient | C | float | 5,000 | Per-patient cost of a new trial |
| Discount rate | r | float | 0.035 | Annual discount rate |
| Within-study variance | sigma^2 | float | SE^2 | For EVSI computation |
| Simulation draws | M | int | 10,000 | Monte Carlo sample size |
| Random seed | -- | int | 42 | For reproducibility |

### Output Structure

| Metric | Description | Units |
|:---|:---|:---|
| Current optimal | Treat or do not treat | -- |
| P(wrong) | Probability of wrong decision | Probability |
| EVPI | Per-decision EVPI | Net benefit units |
| EVPI_pop | Population-level EVPI | Net benefit units x persons x years |
| EVPPI_theta | EVPPI for treatment effect | Net benefit units |
| EVPPI_tau2 | EVPPI for heterogeneity | Net benefit units |
| Theta fraction | EVPPI_theta / (EVPPI_theta + EVPPI_tau2) | Proportion |
| EVSI(n) | EVSI for trial of size n | Net benefit units |
| Optimal N | Trial size maximizing net benefit | Patients |
| Break-even N | Largest cost-effective trial size | Patients |
| GRADE certainty | Implied GRADE level | High/Moderate/Low/Very Low |
| Input hash | SHA-256 of all inputs (16 hex chars) | String |
| Certification | PASS/WARN/REJECT | Status |
