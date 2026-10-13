from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


class ContractError(ValueError):
    """Raised when an input violates an Engine contract."""


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{field_name} must be a number")
    number = float(value)
    if number < 0:
        raise ContractError(f"{field_name} must be non-negative")
    return number


@dataclass(frozen=True)
class Annotation:
    passed: bool
    failure_reason: str | None = None
    score: float | None = None
    comment: str | None = None
    schema_version: str = "pass-fail/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ContractError("annotation.passed must be a boolean")
        if self.score is not None and not 1 <= self.score <= 5:
            raise ContractError("annotation.score must be between 1 and 5")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Annotation:
        if "passed" not in data:
            raise ContractError("annotation.passed is required")
        return cls(
            passed=data["passed"],
            failure_reason=data.get("failure_reason"),
            score=float(data["score"]) if data.get("score") is not None else None,
            comment=data.get("comment"),
            schema_version=str(data.get("schema_version", "pass-fail/v1")),
        )


@dataclass(frozen=True)
class CandidateResult:
    model_id: str
    annotation: Annotation
    cost_rank: int
    latency_ms: float
    output_hash: str

    def __post_init__(self) -> None:
        _required_text(self.model_id, "candidate.model_id")
        _required_text(self.output_hash, "candidate.output_hash")
        if isinstance(self.cost_rank, bool) or not isinstance(self.cost_rank, int):
            raise ContractError("candidate.cost_rank must be an integer")
        if self.cost_rank <= 0:
            raise ContractError("candidate.cost_rank must be positive")
        _non_negative(self.latency_ms, "candidate.latency_ms")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateResult:
        return cls(
            model_id=_required_text(data.get("model_id"), "candidate.model_id"),
            annotation=Annotation.from_dict(data.get("annotation", {})),
            cost_rank=int(data.get("cost_rank", 0)),
            latency_ms=_non_negative(data.get("latency_ms"), "candidate.latency_ms"),
            output_hash=_required_text(data.get("output_hash"), "candidate.output_hash"),
        )


@dataclass(frozen=True)
class EvaluationRecord:
    request_id: str
    request_text: str
    candidates: tuple[CandidateResult, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.request_text, "request_text")
        if len(self.candidates) < 2:
            raise ContractError("each record requires at least two candidates")
        model_ids = [candidate.model_id for candidate in self.candidates]
        if len(model_ids) != len(set(model_ids)):
            raise ContractError("candidate model_id values must be unique within a record")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise ContractError("metadata keys and values must be strings")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRecord:
        raw_candidates = data.get("candidates")
        if not isinstance(raw_candidates, list):
            raise ContractError("candidates must be a list")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ContractError("metadata must be an object")
        return cls(
            request_id=_required_text(data.get("request_id"), "request_id"),
            request_text=_required_text(data.get("request_text"), "request_text"),
            candidates=tuple(CandidateResult.from_dict(item) for item in raw_candidates),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    cost_rank: int
    average_latency_ms: float
    sample_count: int

    def __post_init__(self) -> None:
        _required_text(self.model_id, "model_id")
        if isinstance(self.cost_rank, bool) or not isinstance(self.cost_rank, int):
            raise ContractError("cost_rank must be an integer")
        if self.cost_rank <= 0:
            raise ContractError("cost_rank must be positive")
        _non_negative(self.average_latency_ms, "average_latency_ms")
        if self.sample_count < 0:
            raise ContractError("sample_count must be non-negative")


@dataclass(frozen=True)
class CandidatePrediction:
    model_id: str
    success_probability: float
    cost_rank: int
    vocabulary_coverage: float
    eligible: bool
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class RoutingPrediction:
    request_id: str
    candidates: tuple[CandidatePrediction, ...]
    abstained: bool
    bundle_version: str


@dataclass(frozen=True)
class RoutingDecision:
    request_id: str
    selected_model_id: str
    prediction: RoutingPrediction
    used_fallback: bool
    reason: str
