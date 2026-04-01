import pytest
from metavoi.pipeline import run_voi
from metavoi.models import VoIResult


def test_pipeline_returns_result(bcg_input):
    result = run_voi(bcg_input)
    assert isinstance(result, VoIResult)


def test_pipeline_decision_is_treat_or_no_treat(bcg_input):
    result = run_voi(bcg_input)
    assert result.current_optimal in ("treat", "no_treat")


def test_pipeline_evpi_positive_for_uncertain(uncertain_input):
    result = run_voi(uncertain_input)
    assert result.evpi > 0.001


def test_pipeline_evpi_near_zero_for_certain(certain_input):
    result = run_voi(certain_input)
    assert result.evpi < 0.001


def test_pipeline_evsi_curve_populated(bcg_input):
    result = run_voi(bcg_input)
    assert len(result.evsi_curve) > 0


def test_pipeline_certification_pass(bcg_input):
    result = run_voi(bcg_input)
    assert result.certification in ("PASS", "WARN")


def test_pipeline_grade_populated(bcg_input):
    result = run_voi(bcg_input)
    assert result.implied_certainty in ("High", "Moderate", "Low", "Very Low")


def test_pipeline_hash_nonempty(bcg_input):
    result = run_voi(bcg_input)
    assert len(result.input_hash) > 0
