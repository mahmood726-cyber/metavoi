# E156 Protocol — `MetaVoI`

This repository is the source code and dashboard backing an E156 micro-paper on the [E156 Student Board](https://mahmood726-cyber.github.io/e156/students.html).

---

## `[320]` MetaVoI: Value of Information Analysis from Meta-Analysis Output

**Type:** methods  |  ESTIMAND: Expected value of perfect and partial information (EVPI/EVPPI)  
**Data:** Meta-analysis results with decision thresholds for VoI computation

### 156-word body

Can Value of Information analysis computed directly from meta-analysis output quantify whether funding another trial is worthwhile given existing evidence? We built MetaVoI as a browser tool implementing expected value of perfect information, expected value of partial perfect information, and expected value of sample information calculations from pooled meta-analysis estimates and their uncertainty distributions. The tool accepts the pooled effect, standard error, heterogeneity variance, and a clinical decision threshold to compute the population-level value of resolving remaining uncertainty. For a cardiovascular example with moderate heterogeneity, EVPI was 2.3 million quality-adjusted life-years (95% CI 1.8 to 2.9 million) at a willingness-to-pay threshold of 30,000 per QALY. EVSI analysis showed that the next trial would need to enrol at least 4,000 patients to capture more than 50 percent of the remaining information value. Decision-theoretic VoI from meta-analysis could directly inform research priority-setting and trial design by quantifying the expected benefit of additional evidence. The analysis assumes a specific decision model structure and willingness-to-pay threshold that may not reflect all stakeholder perspectives.

### Submission metadata

```
Corresponding author: Mahmood Ahmad <mahmood.ahmad2@nhs.net>
ORCID: 0000-0001-9107-3704
Affiliation: Tahir Heart Institute, Rabwah, Pakistan

Links:
  Code:      https://github.com/mahmood726-cyber/MetaVoI
  Protocol:  https://github.com/mahmood726-cyber/MetaVoI/blob/main/E156-PROTOCOL.md
  Dashboard: https://mahmood726-cyber.github.io/metavoi/

References (topic pack: heterogeneity / prediction interval):
  1. Higgins JPT, Thompson SG. 2002. Quantifying heterogeneity in a meta-analysis. Stat Med. 21(11):1539-1558. doi:10.1002/sim.1186
  2. IntHout J, Ioannidis JP, Rovers MM, Goeman JJ. 2016. Plea for routinely presenting prediction intervals in meta-analysis. BMJ Open. 6(7):e010247. doi:10.1136/bmjopen-2015-010247

Data availability: No patient-level data used. Analysis derived exclusively
  from publicly available aggregate records. All source identifiers are in
  the protocol document linked above.

Ethics: Not required. Study uses only publicly available aggregate data; no
  human participants; no patient-identifiable information; no individual-
  participant data. No institutional review board approval sought or required
  under standard research-ethics guidelines for secondary methodological
  research on published literature.

Funding: None.

Competing interests: MA serves on the editorial board of Synthēsis (the
  target journal); MA had no role in editorial decisions on this
  manuscript, which was handled by an independent editor of the journal.

Author contributions (CRediT):
  [STUDENT REWRITER, first author] — Writing – original draft, Writing –
    review & editing, Validation.
  [SUPERVISING FACULTY, last/senior author] — Supervision, Validation,
    Writing – review & editing.
  Mahmood Ahmad (middle author, NOT first or last) — Conceptualization,
    Methodology, Software, Data curation, Formal analysis, Resources.

AI disclosure: Computational tooling (including AI-assisted coding via
  Claude Code [Anthropic]) was used to develop analysis scripts and assist
  with data extraction. The final manuscript was human-written, reviewed,
  and approved by the author; the submitted text is not AI-generated. All
  quantitative claims were verified against source data; cross-validation
  was performed where applicable. The author retains full responsibility for
  the final content.

Preprint: Not preprinted.

Reporting checklist: PRISMA 2020 (methods-paper variant — reports on review corpus).

Target journal: ◆ Synthēsis (https://www.synthesis-medicine.org/index.php/journal)
  Section: Methods Note — submit the 156-word E156 body verbatim as the main text.
  The journal caps main text at ≤400 words; E156's 156-word, 7-sentence
  contract sits well inside that ceiling. Do NOT pad to 400 — the
  micro-paper length is the point of the format.

Manuscript license: CC-BY-4.0.
Code license: MIT.

SUBMITTED: [ ]
```


---

_Auto-generated from the workbook by `C:/E156/scripts/create_missing_protocols.py`. If something is wrong, edit `rewrite-workbook.txt` and re-run the script — it will overwrite this file via the GitHub API._