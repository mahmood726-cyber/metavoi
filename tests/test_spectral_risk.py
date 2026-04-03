"""Tests for spectral risk measures."""

from metavoi.spectral_risk import compute_spectral_risk


def test_spectral_evpi_non_negative(bcg_input):
    """All spectral EVPI values must be >= 0."""
    res = compute_spectral_risk(bcg_input)
    for name, evpi in res["spectral_evpi_by_spectrum"].items():
        assert evpi >= 0, f"Spectral EVPI for {name} = {evpi} < 0"


def test_cvar05_non_negative_and_consistent(bcg_input):
    """CVaR_0.05 EVPI must be >= 0 and present in the spectrum dict."""
    res = compute_spectral_risk(bcg_input)
    assert res["cvar_05_evpi"] >= 0, (
        f"CVaR_05 EVPI {res['cvar_05_evpi']:.6f} should be non-negative"
    )
    assert "cvar_05" in res["spectral_evpi_by_spectrum"]
    assert res["cvar_05_evpi"] == res["spectral_evpi_by_spectrum"]["cvar_05"]


def test_risk_aversion_curve_populated(bcg_input):
    """Risk aversion curve should have 5 entries with increasing gamma."""
    res = compute_spectral_risk(bcg_input)
    curve = res["risk_aversion_curve"]
    assert len(curve) == 5
    gammas = [pt["gamma"] for pt in curve]
    assert gammas == [0.1, 0.5, 1.0, 2.0, 5.0]
    for pt in curve:
        assert pt["evpi"] >= 0


def test_wang_curve_populated(bcg_input):
    """Wang transform curve should have 4 entries with increasing lambda."""
    res = compute_spectral_risk(bcg_input)
    curve = res["wang_curve"]
    assert len(curve) == 4
    lambdas = [pt["lambda"] for pt in curve]
    assert lambdas == [0.0, 0.5, 1.0, 1.5]
    for pt in curve:
        assert pt["evpi"] >= 0


def test_optimal_spectrum_valid(bcg_input):
    """Optimal spectrum must be one of the named spectra."""
    res = compute_spectral_risk(bcg_input)
    valid_names = {"risk_neutral", "cvar_05", "exponential_1", "wang_05"}
    assert res["optimal_spectrum"] in valid_names, (
        f"Optimal spectrum '{res['optimal_spectrum']}' not in {valid_names}"
    )
