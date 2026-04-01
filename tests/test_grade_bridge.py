import pytest
from metavoi.grade_bridge import grade_from_p_wrong


def test_high_certainty():
    result = grade_from_p_wrong(0.02)
    assert result["certainty"] == "High"


def test_moderate_certainty():
    result = grade_from_p_wrong(0.12)
    assert result["certainty"] == "Moderate"


def test_low_certainty():
    result = grade_from_p_wrong(0.30)
    assert result["certainty"] == "Low"


def test_very_low_certainty():
    result = grade_from_p_wrong(0.45)
    assert result["certainty"] == "Very Low"


def test_boundary_high_moderate():
    result = grade_from_p_wrong(0.05)
    assert result["certainty"] == "High"


def test_recommendation_present():
    result = grade_from_p_wrong(0.12)
    assert len(result["recommendation"]) > 10
