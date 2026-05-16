import json
from pathlib import Path

from metavoi.models import VoIInput


ROOT = Path(__file__).resolve().parents[1]


def test_fixture_data_has_required_schema():
    for path in (ROOT / "data").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["studies"]
        assert {"mcid", "population", "horizon_years", "cost_per_patient"}.issubset(payload)
        for study in payload["studies"]:
            assert {"yi", "vi", "label"}.issubset(study)
            assert study["vi"] > 0


def test_voi_input_contract_accepts_canonical_values():
    item = VoIInput(
        theta=-0.25,
        se=0.055,
        tau2=0.002,
        k=5,
        mcid=-0.1,
        population=10_000_000,
        horizon_years=10,
        cost_per_patient=2000,
        discount_rate=0.035,
        within_study_var=0.017,
        n_sim=1000,
        seed=42,
    )
    assert item.k == 5
    assert item.se > 0
    assert item.population > 0
