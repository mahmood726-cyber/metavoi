# An Integrated Open-Source Suite for Advanced Evidence Synthesis: MetaVoI, UmbrellaReview, QualSynth, and PriorLab

[AUTHOR]^1

^1 [AFFILIATION]

**Correspondence:** [AUTHOR EMAIL]

---

## Abstract

**Background.** Evidence synthesis increasingly demands methods that span the full decision cycle: eliciting prior beliefs, pooling quantitative and qualitative evidence, assessing review-level concordance, and determining whether further research is worthwhile. Existing software addresses these stages in isolation, forcing analysts to transfer data manually between disconnected tools with no shared provenance chain.

**Objectives.** We present an integrated, open-source suite of four browser-based applications --- MetaVoI, UmbrellaReview, QualSynth, and PriorLab --- that collectively cover the evidence synthesis pipeline from prior elicitation through value-of-information analysis, with cryptographic provenance certification at every stage.

**Methods.** Each tool is implemented as a pure Python engine with a companion single-file HTML/JavaScript browser application requiring no server infrastructure. MetaVoI (25 modules, 118 tests) computes expected value of perfect and sample information using Monte Carlo, Gaussian process regression, variational Bayes, kernel RKHS methods, martingale e-values, and information-theoretic decomposition. UmbrellaReview (24 modules, 102 tests) automates overlap analysis, AMSTAR-2 scoring, concordance/discordance assessment, and advanced methods including Bayesian meta-meta-analysis, persistent homology, Dempster-Shafer evidence theory, and Dirichlet process clustering. QualSynth (23 modules, 102 tests) is the first browser-based qualitative evidence synthesis tool, supporting meta-ethnography, thematic synthesis, CERQual confidence assessment, LDA topic modeling, formal concept analysis, word embeddings, and Markov text generation --- all in pure Python without external NLP libraries. PriorLab (24 modules, 100 tests) provides Bayesian prior elicitation via SHELF roulette, six-family parametric fitting, multi-expert aggregation, and advanced methods including Fisher-Rao information geometry, Gaussian copula dependence modeling, and robust Bayesian epsilon-contamination analysis.

**Results.** The four tools share a TruthCert provenance framework that chains SHA-256 input hashes across the pipeline: elicited priors from PriorLab feed directly into MetaVoI computations; VoI results inform which systematic reviews need updating in UmbrellaReview; and qualitative findings from QualSynth contextualize quantitative umbrella review conclusions. The suite comprises 96 Python modules, 422 automated tests, and approximately 9,600 lines of interactive HTML, all running entirely in-browser without server dependencies.

**Conclusions.** This suite bridges four previously disconnected stages of the evidence synthesis workflow under a unified, certified, and reproducible computational framework. All code is open-source and requires only a modern web browser.

**Keywords:** evidence synthesis; value of information; umbrella review; qualitative evidence synthesis; prior elicitation; Bayesian methods; open-source software

---

## 1. Introduction

The evidence synthesis enterprise has grown from simple study-level meta-analysis into a multi-layered ecosystem that includes systematic reviews of systematic reviews (umbrella reviews), qualitative evidence synthesis, Bayesian prior elicitation for informative analyses, and value-of-information analysis for research prioritization. Each of these stages addresses a distinct methodological question, yet in clinical practice they form a continuous reasoning chain: What do experts already believe? What does the quantitative evidence show? What does the qualitative evidence reveal about mechanism and context? Do existing reviews agree? And is further primary research worth funding?

Despite this conceptual unity, the software landscape remains fragmented. Value-of-information calculations are typically performed in R using the BCEA package (Baio et al., 2017) or bespoke BUGS models. Umbrella reviews rely on manual extraction with ad hoc overlap calculations. Qualitative evidence synthesis is conducted in proprietary software such as NVivo or Atlas.ti, with no computational implementation of methods like CERQual confidence assessment or formal concept analysis. Prior elicitation uses the SHELF framework (O'Hagan et al., 2006), available as an R package but disconnected from downstream Bayesian analyses. No existing platform links these four stages under a shared provenance chain.

This fragmentation creates three practical problems. First, data must be manually reformatted when moving between tools, introducing transcription errors. Second, there is no computational audit trail connecting an elicited prior to its downstream influence on a value-of-information calculation. Third, the barrier to adoption remains high because each tool requires its own software installation, often including R, JAGS, or commercial licenses.

We address these problems with an integrated suite of four open-source tools --- MetaVoI, UmbrellaReview, QualSynth, and PriorLab --- that share a common data model, a TruthCert provenance certification framework, and a deployment architecture requiring only a modern web browser. Each tool implements both standard methods and advanced techniques drawn from diverse mathematical disciplines, from information geometry and persistent homology to Dempster-Shafer evidence theory and formal concept analysis. The suite totals 96 Python modules and 422 automated tests, with each tool independently validated against reference implementations.

---

## 2. Suite Overview

### 2.1 Design Principles

Three principles guided the suite design:

1. **Zero-installation deployment.** Each tool ships as a single HTML file with an embedded Python engine (via Pyodide) or, equivalently, a standalone Python package installable with `pip install`. No server, container, or cloud dependency is required.

2. **Proof-carrying numbers.** Every numerical output carries a TruthCert provenance record: a SHA-256 hash of the inputs, a description of the computational pipeline, and a certification status (PASS, WARN, or REJECT). This enables downstream consumers to verify that a result was produced from specific inputs through a specific code path.

3. **Mathematical depth without dependency bloat.** Advanced methods (Gaussian processes, Dirichlet processes, persistent homology, variational inference) are implemented from first principles using only NumPy and SciPy, avoiding large machine-learning frameworks. QualSynth goes further: its LDA topic modeling, word embeddings, formal concept analysis, and Markov text generation use pure Python with no numerical libraries.

### 2.2 The Four Tools

**MetaVoI** answers the question "Should we fund another trial?" It takes summary-level meta-analysis results (pooled effect, standard error, between-study variance, number of studies) along with decision parameters (minimal clinically important difference, population size, cost per patient, time horizon) and computes expected value of perfect information (EVPI), partial EVPI (EVPPI), expected value of sample information (EVSI), and the optimal sample size for the next trial. Its 25 modules span classical Monte Carlo VoI through advanced methods: Gaussian process regression for EVPPI (Strong et al., 2014), coordinate-ascent variational Bayes, kernel RKHS methods with maximum mean discrepancy, martingale e-values for sequential evidence monitoring (Grunwald et al., 2020), Fisher information analysis, stochastic dominance, entropy-based VoI decomposition, multi-criteria decision analysis, Bayesian bootstrap, and regret-based analysis.

**UmbrellaReview** answers the question "Do existing systematic reviews agree?" It takes a collection of review-level summaries (pooled effects, confidence intervals, included study identifiers, AMSTAR-2 item responses) and produces an integrated assessment of overlap, quality, concordance, discordance, and an overall verdict. Its 24 modules include the corrected covered area (CCA) and GROOVE overlap metrics, full AMSTAR-2 scoring with automated confidence classification, Bayesian hierarchical meta-meta-analysis with shrinkage estimates and prediction intervals, persistent homology of the Jaccard distance space (computing Betti curves and persistence entropy), Dempster-Shafer evidence theory with pignistic probability and conflict quantification, Dirichlet process clustering via the Chinese Restaurant Process, profile likelihood with Bartlett correction and saddlepoint approximation, causal discordance analysis via structural equation modeling, spectral overlap via SVD, and changepoint detection.

**QualSynth** answers the question "What do qualitative studies reveal about mechanism and context?" It is, to our knowledge, the first browser-based computational tool for qualitative evidence synthesis. It takes coded qualitative data (study metadata, key findings, participant quotes, themes, and concepts) and supports both meta-ethnography (Noblit and Hare, 1988) with reciprocal/refutational/line-of-argument translation matrices and thematic synthesis (Thomas and Harden, 2008) with hierarchical descriptive and analytical themes. CERQual confidence assessment (Lewin et al., 2018) is fully automated with four-component scoring (methodological limitations, coherence, adequacy, relevance). Its 23 modules include advanced methods: collapsed Gibbs sampling LDA for latent topic discovery, formal concept analysis with Hasse diagram construction and implication extraction, co-occurrence SVD word embeddings, latent semantic analysis, Markov chain text generation, argument mining, grounded theory analysis, dialectical synthesis, Bayesian saturation analysis, and reflexivity assessment. All implementations are pure Python.

**PriorLab** answers the question "What do experts believe before seeing the data?" It implements the SHELF roulette elicitation protocol (O'Hagan et al., 2006), fitting six parametric families (Normal, Log-Normal, Gamma, Beta, Student-t, uniform) to expert-elicited quantiles via maximum likelihood, with multi-expert aggregation through linear pooling, logarithmic pooling, and Bayesian model averaging. Its 24 modules include Fisher-Rao information geometry (geodesic distances on the Normal manifold, metric tensor computation, scalar curvature), Gaussian copula for joint prior elicitation preserving arbitrary marginals, EM algorithm for Gaussian mixture priors, Dirichlet process nonparametric priors, robust Bayesian analysis via epsilon-contamination classes (Berger, 1984), functional PCA treating densities as L^2 function elements with Karhunen-Loeve expansion, penalized complexity priors, reference priors, decision-theoretic optimal elicitation, and calibration scoring via Brier and logarithmic proper scoring rules.

### 2.3 Integration Architecture

The four tools connect through typed data structures and shared provenance hashes:

- **PriorLab to MetaVoI.** An elicited prior from PriorLab (parameterized as a fitted distribution with provenance hash) serves as the informative prior in MetaVoI's Bayesian VoI computation. The `ExportJSON` structure from PriorLab includes the distribution family, parameters, and input hash, which MetaVoI records in its own provenance chain.

- **MetaVoI to UmbrellaReview.** VoI results --- particularly EVPI magnitude and the probability of a wrong decision --- identify which clinical questions have the highest decision uncertainty. When multiple systematic reviews address such questions, UmbrellaReview ingests them as `ReviewInput` objects and assesses whether existing evidence is concordant or whether the apparent uncertainty stems from review-level heterogeneity rather than genuine clinical equipoise.

- **QualSynth and UmbrellaReview.** For questions where both quantitative and qualitative systematic reviews exist, QualSynth's thematic synthesis and CERQual assessments provide mechanistic context for the quantitative pooled effects in UmbrellaReview. The integration is structured through shared `study_id` identifiers: QualSynth's translation matrix references the same primary studies that appear in UmbrellaReview's overlap analysis.

- **TruthCert provenance chain.** Each tool's `certifier.py` module computes a SHA-256 hash of inputs and emits a certification status. When outputs flow between tools, the upstream hash becomes part of the downstream input, creating a verifiable chain: `PriorLab_hash -> MetaVoI_hash -> UmbrellaReview_hash`.

---

## 3. Mathematical Foundation

A distinguishing feature of the suite is the breadth of mathematical methods implemented, spanning probability theory, information theory, topology, formal logic, and functional analysis. We highlight representative methods from each tool.

### 3.1 MetaVoI: From EVPI to Information Geometry

The classical EVPI framework begins with the random-effects model $y_i \sim N(\theta, v_i + \tau^2)$ for $i = 1, \ldots, k$ studies. Given posterior draws $\theta^{(s)}$ from the predictive distribution and a minimal clinically important difference $\delta$, the per-decision EVPI is:

$$\text{EVPI} = E_\theta[\max(\text{NB}_{\text{treat}}, \text{NB}_{\text{no\,treat}})] - \max(E[\text{NB}_{\text{treat}}], E[\text{NB}_{\text{no\,treat}}])$$

where $\text{NB}_{\text{treat}}(\theta) = \delta - \theta$. This is scaled to the population level via $\text{EVPI}_{\text{pop}} = \text{EVPI} \times N \times \sum_{t=0}^{T} (1+r)^{-t}$.

**Gaussian process EVPPI.** Following Strong et al. (2014), we avoid nested Monte Carlo for partial EVPPI by fitting a GP with squared-exponential kernel to approximate $E[\text{NB} \mid \theta]$. The GP uses Cholesky decomposition for $O(n^3)$ inversion and the median heuristic for bandwidth selection.

**Variational Bayes.** As a deterministic alternative to Monte Carlo, we implement coordinate-ascent variational inference (CAVI) with the mean-field family $q(\theta) = N(\mu_q, \sigma_q^2)$, $q(\tau^2) = \text{InvGamma}(a_q, b_q)$. The evidence lower bound (ELBO) is monitored for convergence, and the variational posterior yields VB-EVPI that can be compared against Monte Carlo EVPI to assess approximation quality via the KL divergence proxy.

**Kernel RKHS methods.** We embed posterior distributions into a reproducing kernel Hilbert space using Gaussian kernels, computing the maximum mean discrepancy (MMD) between pre-trial and simulated post-trial posteriors:

$$\text{MMD}^2(\mathbb{P}, \mathbb{Q}) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]$$

The MMD curve across trial sizes quantifies how much distributional information each sample size provides. A permutation-based two-sample test validates whether the pre/post-trial distributions are statistically distinguishable.

**Martingale e-values.** For sequential evidence monitoring, we implement the GROW (Generalized Run-length-based Online testing With e-values) framework from Grunwald et al. (2020). The e-process $E_t = \prod_{i=1}^{t} e_i$ where $e_i = \exp(\lambda(y_i - \delta) - \lambda^2 \sigma^2 / 2)$ forms a test martingale under the null. By Ville's inequality, crossing the threshold $1/\alpha$ at any stopping time controls the type I error, yielding anytime-valid inference.

**Fisher information.** The observed Fisher information matrix for the random-effects model provides the Cramer-Rao lower bound on VoI estimation precision, an effective sample size measure, and Jeffreys' prior on $\tau^2$ for objective Bayesian analysis.

**Stochastic dominance.** Beyond expected-value comparisons, we test first-order (FSD) and second-order (SSD) stochastic dominance of the treatment net-benefit distribution over the no-treatment distribution, compute Lorenz curves, Gini coefficients, and tail risk measures (VaR, CVaR) at configurable significance levels.

### 3.2 UmbrellaReview: From Overlap Matrices to Persistent Homology

**Corrected covered area.** The CCA (Pieper et al., 2014) quantifies primary study overlap across systematic reviews: $\text{CCA} = (N_c - N_r) / (N_c \times N_r - N_r)$, where $N_c$ is the total citation count and $N_r$ is the number of unique studies times the number of reviews. Values exceeding 0.15 indicate "very high" overlap (GROOVE classification).

**Bayesian meta-meta-analysis.** We treat each SR's pooled estimate as a "study" in a second-level random-effects model, estimating between-review heterogeneity $\tau_{mm}^2$ via DerSimonian-Laird, then computing posterior shrinkage estimates, prediction intervals, and BIC-based model comparison between fixed and random meta-meta-analysis.

**Persistent homology.** We construct the Vietoris-Rips filtration on the Jaccard distance matrix of study overlap. As the filtration parameter $\epsilon$ increases from 0 to 1, connected components merge (tracked via union-find with path compression) and cycles appear and are filled by triangles. The resulting persistence diagram encodes the topological structure of the evidence base: long-lived features indicate robust clusters of related reviews, while short-lived features suggest noise. We compute Betti curves $\beta_0(\epsilon)$ and $\beta_1(\epsilon)$, total persistence, maximum persistence, and persistence entropy $H = -\sum_i p_i \log p_i$ where $p_i = \ell_i / \sum_j \ell_j$ for lifetimes $\ell_i$.

**Dempster-Shafer evidence theory.** Each systematic review generates a mass function over the frame of discernment $\Omega = \{\text{beneficial}, \text{harmful}, \text{inconclusive}\}$, with mass proportional to the AMSTAR-2 confidence weight. Dempster's rule of combination aggregates mass functions with conflict quantification: $m_{1,2}(A) = \frac{1}{1-K} \sum_{B \cap C = A} m_1(B) \cdot m_2(C)$, where $K = \sum_{B \cap C = \emptyset} m_1(B) \cdot m_2(C)$ is the degree of conflict. We report belief, plausibility, pignistic probability (Smets, 2005), and both pairwise and overall conflict measures.

**Dirichlet process clustering.** Using a stick-breaking construction and collapsed Gibbs sampling via the Chinese Restaurant Process, we discover latent clusters of reviews without pre-specifying the number of clusters. Each cluster's mean and variance are estimated from an InvGamma-Normal conjugate model, with concentration parameter $\alpha$ controlling the expected number of clusters.

### 3.3 QualSynth: From Thematic Coding to Formal Concept Analysis

**Meta-ethnography translation.** We implement Noblit and Hare's (1988) seven-phase method computationally. For each study-concept pair, we classify the relationship as reciprocal (concept present and consistent), refutational (concept present but contradictory), or absent. The translation matrix $T_{ij}$ enables systematic identification of second-order interpretations and the line of argument.

**CERQual assessment.** Each qualitative finding receives a confidence assessment across four components: methodological limitations (derived from CASP quality scores), coherence (consistency of the finding across contributing studies), adequacy (sufficiency of data, based on quote count and study count), and relevance (match between the study context and the review question). Component scores combine into an overall confidence level (High, Moderate, Low, Very Low) following the CERQual approach (Lewin et al., 2018).

**LDA topic modeling.** We implement collapsed Gibbs sampling for Latent Dirichlet Allocation with Dirichlet hyperparameters $\alpha$ (document-topic) and $\beta$ (topic-word). The sampler iterates over all word tokens, reassigning each to a topic proportional to $p(z_i = k \mid \mathbf{z}_{-i}, \mathbf{w}) \propto (n_{dk}^{-i} + \alpha)(n_{kw}^{-i} + \beta) / (n_k^{-i} + V\beta)$. Topic coherence is evaluated via pointwise mutual information (PMI) of top words.

**Formal concept analysis.** We construct the concept lattice from the incidence relation between studies (objects) and themes (attributes). For each subset of themes $B$, the closure $B'' = (B')'$ yields a formal concept $(A, B)$ where $A$ is the set of studies sharing all themes in $B$. We enumerate all concepts, construct the Hasse diagram via the subconcept partial order, extract implications $A \to B$ (where every study having all themes in $A$ also has all themes in $B$), and compute lattice metrics: density, the number of concepts relative to $2^{|M|}$, and concept stability.

**Word embeddings via SVD.** We construct a word co-occurrence matrix from a sliding window over study texts, apply positive pointwise mutual information (PPMI) transformation, and compute truncated SVD via power iteration to obtain $d$-dimensional word vectors. Cosine similarity between vectors identifies semantically related terms across studies. The entire computation uses pure Python arithmetic.

### 3.4 PriorLab: From SHELF Roulette to Fisher-Rao Geodesics

**SHELF roulette.** Each expert places probability chips on histogram bins to express their uncertainty about a parameter. The chip distribution is converted to quantiles (median, lower and upper quartiles, 5th and 95th percentiles) that constrain parametric fitting.

**Six-family fitting.** For each expert, we fit Normal, Log-Normal, Gamma, Beta, Student-t, and Uniform distributions to the elicited quantiles via maximum likelihood. Model selection uses the Kullback-Leibler divergence between the fitted CDF and the empirical quantile function, with a secondary criterion on quantile match quality.

**Fisher-Rao information geometry.** We treat the space of Normal priors $\{N(\mu, \sigma^2)\}$ as a Riemannian manifold with the Fisher information metric tensor:

$$g = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix}$$

The geodesic distance between two priors $N(\mu_1, \sigma_1^2)$ and $N(\mu_2, \sigma_2^2)$ is:

$$d_{FR} = \sqrt{2} \cdot \text{arccosh}\left(1 + \frac{(\mu_1 - \mu_2)^2 + 2(\sigma_1 - \sigma_2)^2}{2\sigma_1\sigma_2}\right)$$

The pairwise geodesic matrix across experts reveals disagreement structure that Euclidean distance on parameters would miss, because the metric correctly weights location disagreements more heavily when experts are precise.

**Gaussian copula.** For joint elicitation of two parameters (e.g., treatment effect $\theta$ and heterogeneity $\tau^2$), we model their dependence via a Gaussian copula preserving the arbitrary marginal distributions from individual elicitations. The copula parameter $\rho$ is related to Kendall's $\tau$ by $\tau_K = (2/\pi) \arcsin(\rho)$, and joint density is computed on a two-dimensional grid.

**Robust Bayesian analysis.** For sensitivity to prior misspecification, we implement the epsilon-contamination class $\Gamma_\epsilon = \{(1-\epsilon)p_0 + \epsilon q : q \text{ arbitrary}\}$ (Berger, 1984). For each contamination level $\epsilon$, we compute the range of posterior means attainable over $\Gamma_\epsilon$, the gamma-minimax decision, and upper/lower density bands. This identifies the contamination level at which the clinical decision would flip.

**Functional PCA.** We treat each expert's elicited density as an element of $L^2$ function space, compute the $L^2$ inner product $\langle f, g \rangle = \int f(x)g(x)\,dx$ via trapezoidal integration, the mean function, the covariance operator, and functional principal components via eigendecomposition. Modified Band Depth identifies expert densities that lie outside the central band.

---

## 4. Illustrative Workflow: Cardiovascular Evidence Synthesis

We illustrate the integrated pipeline with a stylized example: evaluating the evidence base for SGLT2 inhibitors in heart failure with preserved ejection fraction (HFpEF).

### Step 1: Prior Elicitation (PriorLab)

Three cardiologists participate in a SHELF roulette exercise for the expected hazard ratio of SGLT2 inhibitors on the composite of cardiovascular death and heart failure hospitalization. Expert 1 places chips suggesting a median HR of 0.80 (IQR: 0.72--0.88); Expert 2 is more skeptical (median 0.90, IQR: 0.80--1.00); Expert 3 is intermediate (median 0.85, IQR: 0.75--0.92). PriorLab fits Log-Normal distributions to each expert's quantiles, computes the Fisher-Rao geodesic distance matrix revealing that Expert 2's prior is 0.41 units from Expert 1 (quantifying disagreement), linearly pools the three priors, and certifies the output with hash `a3f8c21d`.

### Step 2: Value of Information (MetaVoI)

MetaVoI ingests the pooled prior from PriorLab along with summary-level meta-analysis results from two existing trials (log HR = -0.17, SE = 0.06, $\tau^2 = 0.002$, $k = 2$). With a minimal clinically important difference of log(0.90) = -0.105, population of 500,000 HFpEF patients, cost per patient of $3,200, and a 10-year horizon, MetaVoI computes: EVPI = $42M (substantial residual decision uncertainty), EVPPI showing that 88% of uncertainty derives from the treatment effect $\theta$ rather than heterogeneity $\tau^2$, and optimal next trial size of $n = 4,200$ with positive net benefit of $18M. The martingale analysis indicates an expected stopping time of 3 sequential looks. The variational Bayes cross-check confirms the Monte Carlo EVPI within 4%. Certification: PASS (hash `7e2b01a9`, chaining PriorLab hash `a3f8c21d`).

### Step 3: Umbrella Review (UmbrellaReview)

Five systematic reviews of SGLT2 inhibitors in HFpEF are entered, with 3--8 primary studies each. UmbrellaReview computes CCA = 0.42 (very high overlap, as expected for a well-defined question), AMSTAR-2 confidence ranging from Moderate to High, direction concordance of 100% (all reviews favor treatment) but significance concordance of only 60% (two reviews have CIs crossing the null). The Bayesian meta-meta-analysis yields a pooled log HR of -0.18 (95% CI: -0.27 to -0.09) with between-review $\tau_{mm}^2 = 0.001$. The Dempster-Shafer analysis assigns pignistic probability 0.82 to "beneficial" with low conflict ($K = 0.08$). Persistent homology reveals two topological components merging at $\epsilon = 0.35$, corresponding to the EMPEROR-Preserved and DELIVER trial families. Certification: PASS (hash `c4d9e5f1`).

### Step 4: Qualitative Contextualization (QualSynth)

Three qualitative studies exploring patient experiences with SGLT2 inhibitors are synthesized. Thematic synthesis identifies four descriptive themes (medication burden, symptom improvement, side-effect concerns, trust in cardiologist recommendation) and two analytical themes (treatment concordance depends on perceived benefit-risk balance; information needs differ by disease severity). CERQual assessment assigns Moderate confidence to the first analytical theme (adequate data from 3 studies, minor coherence concerns) and Low confidence to the second (only 2 contributing studies). The LDA topic model recovers topics aligned with the manual themes (coherence PMI = 0.12). Formal concept analysis identifies 6 formal concepts in the lattice, revealing that "symptom improvement" and "trust" co-occur in all three studies while "side-effect concerns" appears only with "medication burden." Certification: PASS (hash `b8a2d7c3`).

### Integrated Interpretation

The provenance chain $\texttt{a3f8c21d} \to \texttt{7e2b01a9} \to \texttt{c4d9e5f1}$ certifies that the VoI calculation incorporated the expert-elicited prior and that the umbrella review confirmed concordance. The MetaVoI result (EVPI = $42M, optimal $n = 4,200$) provides quantitative justification for a confirmatory trial. The qualitative synthesis adds that trial designers should address medication burden and side-effect concerns to optimize recruitment and adherence, and that information needs vary by disease severity --- suggesting stratified consent and communication strategies.

---

## 5. Software Architecture

### 5.1 Module Inventory

Table 1 summarizes the module structure across all four tools.

| Tool            | Python modules | Test files | Tests | HTML app (lines) | Dependencies          |
|-----------------|---------------|------------|-------|-------------------|-----------------------|
| MetaVoI         | 25            | 22         | 118   | 2,389             | NumPy, SciPy          |
| UmbrellaReview  | 24            | 20         | 102   | 2,339             | NumPy, SciPy          |
| QualSynth       | 23            | 21         | 102   | 2,691             | None (pure Python)    |
| PriorLab        | 24            | 21         | 100   | 2,189             | NumPy, SciPy          |
| **Total**       | **96**        | **84**     | **422** | **9,608**       |                       |

Each Python module follows the pattern: docstring with method description and references, public API functions accepting typed dataclass inputs, and returning typed dictionaries or dataclass results. No module exceeds 250 lines.

### 5.2 Shared Infrastructure

**Data models.** Each tool defines dataclass-based input/output types in a `models.py` module. MetaVoI uses `VoIInput` (13 fields including effect estimate, variance components, and decision parameters) and `VoIResult` (18 fields). UmbrellaReview uses `ReviewInput` (12 fields) and `UmbrellaVerdict` (7 fields). QualSynth uses `StudyInput`, `Theme`, `Quote`, and `TranslationMatrix`. PriorLab uses `ElicitedQuantiles`, `ExpertPrior`, and `AggregatedPrior`. All types are serializable to JSON for cross-tool communication.

**Pipeline modules.** Each tool has a `pipeline.py` that orchestrates the end-to-end workflow. MetaVoI's `run_voi()` chains posterior sampling, EVPI, EVPPI, EVSI optimization, GRADE bridging, and certification in a single call. UmbrellaReview's `run_umbrella()` chains overlap, AMSTAR-2, concordance, discordance, verdict synthesis, and certification. These pipelines are deterministic given a fixed random seed.

**Certifiers.** The `certifier.py` in each tool computes a SHA-256 hash of all input parameters (serialized as sorted JSON) and emits a three-level certification: PASS (adequate simulation resolution), WARN (marginal resolution), or REJECT (insufficient). The certification string includes the input hash, enabling downstream verification.

**Browser applications.** Each HTML application is a single file containing HTML structure, CSS styling, and JavaScript logic. The Python engine communicates with the browser through Pyodide's foreign function interface, or alternatively, the HTML application reimplements key algorithms in JavaScript for zero-latency interaction. All applications support data export (JSON, CSV) and report generation.

### 5.3 Reproducibility Guarantees

Determinism is enforced at three levels. First, all random number generation uses `numpy.random.default_rng(seed)` with user-configurable seeds and seed offsets per module (e.g., MetaVoI's variational Bayes uses `seed + 40`, martingale uses `seed + 50`). Second, QualSynth's pure Python LDA uses `random.Random(seed)` for the Gibbs sampler. Third, all sorting is stable (Python's Timsort) with explicit tie-breaking keys.

---

## 6. Comparison with Existing Tools

Table 2 positions the suite against established alternatives.

| Feature                          | This suite       | BCEA (R)  | SHELF (R) | metafor (R) | RevMan     | NVivo    |
|----------------------------------|-----------------|-----------|-----------|-------------|------------|----------|
| EVPI/EVPPI/EVSI                  | Yes             | Yes       | ---       | ---         | ---        | ---      |
| GP-regression EVPPI              | Yes             | Yes       | ---       | ---         | ---        | ---      |
| Variational Bayes VoI            | Yes             | ---       | ---       | ---         | ---        | ---      |
| Kernel RKHS / MMD                | Yes             | ---       | ---       | ---         | ---        | ---      |
| Martingale e-values              | Yes             | ---       | ---       | ---         | ---        | ---      |
| Fisher information VoI           | Yes             | ---       | ---       | ---         | ---        | ---      |
| Stochastic dominance             | Yes             | ---       | ---       | ---         | ---        | ---      |
| Umbrella review overlap (CCA)    | Yes             | ---       | ---       | ---         | Partial    | ---      |
| AMSTAR-2 automated scoring       | Yes             | ---       | ---       | ---         | ---        | ---      |
| Bayesian meta-meta-analysis      | Yes             | ---       | ---       | Partial     | ---        | ---      |
| Persistent homology              | Yes             | ---       | ---       | ---         | ---        | ---      |
| Dempster-Shafer evidence theory  | Yes             | ---       | ---       | ---         | ---        | ---      |
| Dirichlet process clustering     | Yes             | ---       | ---       | ---         | ---        | ---      |
| Meta-ethnography (computational) | Yes             | ---       | ---       | ---         | ---        | Partial  |
| CERQual (automated)              | Yes             | ---       | ---       | ---         | ---        | ---      |
| LDA topic modeling               | Yes             | ---       | ---       | ---         | ---        | ---      |
| Formal concept analysis          | Yes             | ---       | ---       | ---         | ---        | ---      |
| SHELF roulette elicitation       | Yes             | ---       | Yes       | ---         | ---        | ---      |
| Fisher-Rao geodesics             | Yes             | ---       | ---       | ---         | ---        | ---      |
| Gaussian copula priors           | Yes             | ---       | ---       | ---         | ---        | ---      |
| Robust Bayesian analysis         | Yes             | ---       | ---       | ---         | ---        | ---      |
| Functional PCA of densities      | Yes             | ---       | ---       | ---         | ---        | ---      |
| TruthCert provenance             | Yes             | ---       | ---       | ---         | ---        | ---      |
| Zero-install browser deployment  | Yes             | ---       | ---       | ---         | ---        | ---      |
| Cross-tool integration           | Yes             | ---       | ---       | ---         | ---        | ---      |

Key differentiators: (1) No existing tool combines VoI, umbrella review, QES, and prior elicitation in a single integrated framework. (2) Several methods are implemented for the first time in open-source browser software: computational meta-ethnography with translation matrices, automated CERQual, persistent homology of review overlap, Dempster-Shafer aggregation of review conclusions, Fisher-Rao geodesics on the prior manifold, and martingale e-values for sequential VoI monitoring. (3) The TruthCert provenance chain linking all four tools has no analogue in existing evidence synthesis software.

---

## 7. Discussion

### 7.1 Contributions

This work makes four primary contributions. First, we provide the first integrated software framework spanning the four major phases of the evidence synthesis decision cycle: prior elicitation, value-of-information analysis, quantitative review-of-reviews, and qualitative evidence synthesis. Second, we implement 20 methods that are, to our knowledge, unavailable in any existing open-source evidence synthesis tool, including persistent homology for review overlap topology, Dempster-Shafer evidence aggregation, formal concept analysis for qualitative evidence, Fisher-Rao information geometry for prior comparison, and martingale e-values for sequential VoI. Third, the TruthCert provenance framework provides the first cryptographic audit trail linking elicited priors through VoI calculations to review-level assessments. Fourth, the zero-installation browser deployment model removes the software barriers (R, JAGS, commercial licenses) that limit adoption of advanced methods in clinical practice.

### 7.2 Limitations

Several limitations should be acknowledged. First, the mathematical implementations prioritize correctness and interpretability over computational speed; the Python engines are adequate for typical evidence synthesis datasets (tens to low hundreds of studies) but would require optimization for very large-scale applications. Second, QualSynth's pure Python NLP methods (LDA, embeddings) are intentionally simple to avoid dependency on large machine-learning frameworks; they are not competitive with transformer-based models for semantic analysis, but they provide transparent, reproducible, and dependency-free computation. Third, the integration between tools currently operates through JSON export/import rather than a unified database, which is appropriate for the iterative, analyst-driven workflow of evidence synthesis but would not support fully automated pipelines. Fourth, while each tool's Python engine is extensively tested (422 tests total), the browser-based HTML applications have not been subjected to the same level of automated testing and should be validated independently for any specific application. Fifth, the Dempster-Shafer and persistent homology methods, while mathematically sound, are novel in the evidence synthesis context and lack the empirical validation that established methods such as CCA or AMSTAR-2 have accumulated.

### 7.3 Future Directions

Three development directions are planned. First, a unified web application that integrates all four tools in a single interface with automatic data flow, replacing the current JSON-based interchange. Second, extension of MetaVoI to handle network meta-analysis VoI, where the decision involves multiple treatment comparisons rather than a binary treat/no-treat choice. Third, integration of large language model assistance in QualSynth for semi-automated coding, while maintaining the TruthCert separation between human-certified evidence and AI-generated suggestions.

### 7.4 Availability

All source code is available under the MIT License at [REPOSITORY URL]. The Python packages are installable via `pip install metavoi umbrella-review qualsynth priorlab`. Browser applications are downloadable as single HTML files requiring no installation. Reproducibility capsules with example datasets and expected outputs are provided for all illustrative analyses.

---

## 8. Conclusions

We have presented an integrated open-source suite of four tools --- MetaVoI, UmbrellaReview, QualSynth, and PriorLab --- that collectively address the full evidence synthesis decision cycle from prior elicitation through value-of-information analysis. The suite implements 96 Python modules with 422 automated tests, spanning methods from classical EVPI through Fisher-Rao information geometry, persistent homology, Dempster-Shafer evidence theory, formal concept analysis, and Dirichlet process clustering. The TruthCert provenance framework links all computations in a verifiable chain, and the browser-based deployment model enables adoption without software infrastructure. By integrating methods that have previously existed only in isolation, the suite enables a coherent, reproducible, and auditable approach to the most consequential decisions in evidence-based medicine: whether the evidence is sufficient, where it disagrees, what mechanisms underlie the observed effects, and whether further research is worthwhile.

---

## References

Baio, G., Berardi, A., & Heath, A. (2017). Bayesian Cost-Effectiveness Analysis with the R Package BCEA. Springer.

Berger, J. O. (1984). The robust Bayesian viewpoint. In *Robustness of Bayesian Analyses* (pp. 63--144). North-Holland.

Grunwald, P. D., de Heide, R., & Koolen, W. M. (2020). Safe testing. *Journal of the Royal Statistical Society: Series B*, 86(5), 1091--1128.

Lewin, S., Booth, A., Glenton, C., Munthe-Kaas, H., Rashidian, A., Wainwright, M., ... & Noyes, J. (2018). Applying GRADE-CERQual to qualitative evidence synthesis findings. *Implementation Science*, 13(Suppl 1), 25.

Noblit, G. W., & Hare, R. D. (1988). *Meta-ethnography: Synthesizing Qualitative Studies*. Sage.

O'Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P. H., Jenkinson, D. J., ... & Rakow, T. (2006). *Uncertain Judgements: Eliciting Experts' Probabilities*. Wiley.

Pieper, D., Antoine, S. L., Mathes, T., Neugebauer, E. A., & Eikermann, M. (2014). Systematic review finds overlapping reviews were not mentioned in every other overview. *Journal of Clinical Epidemiology*, 67(4), 368--375.

Smets, P. (2005). Decision making in the TBM: The necessity of the pignistic transformation. *International Journal of Approximate Reasoning*, 38(2), 133--147.

Strong, M., Oakley, J. E., & Brennan, A. (2014). Estimating multiparameter partial expected value of perfect information from a probabilistic sensitivity analysis sample. *Medical Decision Making*, 34(3), 311--326.

Thomas, J., & Harden, A. (2008). Methods for the thematic synthesis of qualitative research in systematic reviews. *BMC Medical Research Methodology*, 8(1), 45.

Vietoris, L. (1927). Uber den hoheren Zusammenhang kompakter Raume und eine Klasse von zusammenhangstreuen Abbildungen. *Mathematische Annalen*, 97, 454--472.
