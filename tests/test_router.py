from routetune_core.contracts import ModelProfile
from routetune_core.model import CandidateEstimator, EngineBundle, create_bootstrap_bundle
from routetune_core.router import Router


def _bundle(*, economy: float, premium: float, threshold: float = 0.85) -> EngineBundle:
    return EngineBundle(
        version="test",
        created_at="2026-10-08T00:00:00+00:00",
        dataset_fingerprint="test",
        quality_threshold=threshold,
        ood_coverage_threshold=0.1,
        profiles={
            "economy": ModelProfile("economy", 1, 100, 10),
            "premium": ModelProfile("premium", 2, 500, 10),
        },
        estimators={
            "economy": CandidateEstimator("economy", None, None, economy),
            "premium": CandidateEstimator("premium", None, None, premium),
        },
        fallback_model_id="premium",
        training_record_count=10,
        calibration_record_count=10,
    )


def test_router_selects_cheapest_model_above_threshold() -> None:
    decision = Router(_bundle(economy=0.9, premium=0.99)).decide("req", "hello")
    assert decision.selected_model_id == "economy"
    assert not decision.used_fallback


def test_router_falls_back_to_strongest_when_nothing_qualifies() -> None:
    decision = Router(_bundle(economy=0.2, premium=0.8)).decide("req", "hello")
    assert decision.selected_model_id == "premium"
    assert decision.used_fallback
    assert decision.reason == "NO_CANDIDATE_MEETS_QUALITY_THRESHOLD"


def test_bootstrap_bundle_safely_routes_everything_to_highest_cost_rank() -> None:
    bundle = create_bootstrap_bundle({"economy": 1, "premium": 2})

    decision = Router(bundle).decide("req", "a request without training data")

    assert bundle.training_record_count == 0
    assert decision.selected_model_id == "premium"
    assert decision.used_fallback
