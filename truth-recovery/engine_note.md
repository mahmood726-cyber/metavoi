# Engine under test

metavoi's EVPI is pure Python with no DOM. Rather than copy a function body, the
harness/tests import the repo's OWN function VERBATIM at runtime:

    from metavoi.evpi import compute_evpi

Source: metavoi/metavoi/evpi.py:compute_evpi (16 lines). It implements
EVPI = E[max(NB_treat, NB_no_treat)] - max(E[NB_treat], E[NB_no_treat]) by Monte
Carlo, with NB_treat = mcid - draws and NB_no_treat = 0, clamped to >= 0.

The 2389-line app/metavoi.html mirrors this formula in JS (grep "evpi"); the Python
package is the authoritative, importable engine, so it is what we validate.
