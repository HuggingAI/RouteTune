from __future__ import annotations

from routetune_core.contracts import (
    CandidatePrediction,
    RoutingDecision,
    RoutingPrediction,
)
from routetune_core.model import EngineBundle


class Router:
    def __init__(self, bundle: EngineBundle) -> None:
        self.bundle = bundle

    def predict(self, request_id: str, request_text: str) -> RoutingPrediction:
        candidates: list[CandidatePrediction] = []
        coverage_values: list[float] = []
        for model_id, profile in self.bundle.profiles.items():
            probabilities, coverages = self.bundle.estimators[model_id].predict([request_text])
            probability = float(probabilities[0])
            coverage = float(coverages[0])
            coverage_values.append(coverage)
            eligible = probability >= self.bundle.quality_threshold
            candidates.append(
                CandidatePrediction(
                    model_id=model_id,
                    success_probability=probability,
                    cost_rank=profile.cost_rank,
                    vocabulary_coverage=coverage,
                    eligible=eligible,
                    exclusion_reason=None if eligible else "BELOW_QUALITY_THRESHOLD",
                )
            )

        abstained = max(coverage_values, default=0.0) < self.bundle.ood_coverage_threshold
        return RoutingPrediction(
            request_id=request_id,
            candidates=tuple(sorted(candidates, key=lambda item: item.cost_rank)),
            abstained=abstained,
            bundle_version=self.bundle.version,
        )

    def decide(self, request_id: str, request_text: str) -> RoutingDecision:
        prediction = self.predict(request_id, request_text)
        fallback_model = self.bundle.fallback_model_id

        if prediction.abstained:
            return RoutingDecision(
                request_id=request_id,
                selected_model_id=fallback_model,
                prediction=prediction,
                used_fallback=True,
                reason="OOD_FALLBACK_TO_STRONGEST",
            )

        eligible = [candidate for candidate in prediction.candidates if candidate.eligible]
        if not eligible:
            return RoutingDecision(
                request_id=request_id,
                selected_model_id=fallback_model,
                prediction=prediction,
                used_fallback=True,
                reason="NO_CANDIDATE_MEETS_QUALITY_THRESHOLD",
            )

        selected = min(eligible, key=lambda candidate: candidate.cost_rank)
        return RoutingDecision(
            request_id=request_id,
            selected_model_id=selected.model_id,
            prediction=prediction,
            used_fallback=False,
            reason="LOWEST_COST_ABOVE_QUALITY_THRESHOLD",
        )
