from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from routetune_core.contracts import CandidateResult, EvaluationRecord
from routetune_core.model import EngineBundle
from routetune_core.router import Router


@dataclass(frozen=True)
class PolicyMetrics:
    policy_name: str
    quality_rate: float
    p95_latency_ms: float
    selection_counts: dict[str, int]
    high_cost_model_avoidance_rate: float
    fallback_rate: float = 0.0


@dataclass(frozen=True)
class EvaluationReport:
    bundle_version: str
    test_record_count: int
    quality_threshold: float
    policies: tuple[PolicyMetrics, ...]
    high_cost_model_avoidance_rate: float
    quality_delta_vs_highest_cost: float
    calibration_brier_by_model: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (directory / "report.md").write_text(self.to_markdown(), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = [
            "# RouteTune 离线验证报告",
            "",
            f"- Bundle：`{self.bundle_version}`",
            f"- 测试请求：{self.test_record_count}",
            f"- 预测质量门槛：{self.quality_threshold:.1%}",
            f"- 高成本模型避用率：{self.high_cost_model_avoidance_rate:.1%}",
            f"- 相比固定高成本模型质量变化：{self.quality_delta_vs_highest_cost:+.1%}",
            "",
            "|策略|实际通过率|高成本模型避用率|P95 延迟|Fallback|模型选择|",
            "|---|---:|---:|---:|---:|---|",
        ]
        for policy in self.policies:
            selections = ", ".join(
                f"{model}={count}" for model, count in sorted(policy.selection_counts.items())
            )
            lines.append(
                f"|{policy.policy_name}|{policy.quality_rate:.1%}|"
                f"{policy.high_cost_model_avoidance_rate:.1%}|{policy.p95_latency_ms:.0f} ms|"
                f"{policy.fallback_rate:.1%}|{selections}|"
            )
        lines.extend(["", "## 每个候选模型的 Brier Score", ""])
        for model_id, score in sorted(self.calibration_brier_by_model.items()):
            lines.append(f"- `{model_id}`：{score:.4f}")
        lines.append("")
        return "\n".join(lines)


def evaluate(bundle: EngineBundle, records: Sequence[EvaluationRecord]) -> EvaluationReport:
    router = Router(bundle)
    highest_cost = max(bundle.profiles.values(), key=lambda item: item.cost_rank).model_id
    cheapest = min(bundle.profiles.values(), key=lambda item: item.cost_rank).model_id

    route_selections: list[str] = []
    route_fallbacks: list[bool] = []
    for record in records:
        decision = router.decide(record.request_id, record.request_text)
        route_selections.append(decision.selected_model_id)
        route_fallbacks.append(decision.used_fallback)

    policies = (
        _metrics_for_selections(
            "routetune", records, route_selections, highest_cost, route_fallbacks
        ),
        _metrics_for_model("fixed_highest_cost", records, highest_cost, highest_cost),
        _metrics_for_model("fixed_cheapest", records, cheapest, highest_cost),
        _metrics_for_selector(
            "current_rule", records, _metadata_baseline(highest_cost), highest_cost
        ),
        _metrics_for_selector("oracle", records, _oracle_selector(bundle), highest_cost),
    )
    policy_by_name = {policy.policy_name: policy for policy in policies}
    routed = policy_by_name["routetune"]
    fixed_highest_cost = policy_by_name["fixed_highest_cost"]
    return EvaluationReport(
        bundle_version=bundle.version,
        test_record_count=len(records),
        quality_threshold=bundle.quality_threshold,
        policies=policies,
        high_cost_model_avoidance_rate=routed.high_cost_model_avoidance_rate,
        quality_delta_vs_highest_cost=routed.quality_rate - fixed_highest_cost.quality_rate,
        calibration_brier_by_model=_brier_scores(bundle, records),
    )


def _metrics_for_model(
    policy_name: str,
    records: Sequence[EvaluationRecord],
    model_id: str,
    highest_cost_model_id: str,
) -> PolicyMetrics:
    return _metrics_for_selections(
        policy_name, records, [model_id] * len(records), highest_cost_model_id
    )


def _metrics_for_selector(
    policy_name: str,
    records: Sequence[EvaluationRecord],
    selector: Callable[[EvaluationRecord], str],
    highest_cost_model_id: str,
) -> PolicyMetrics:
    return _metrics_for_selections(
        policy_name,
        records,
        [selector(record) for record in records],
        highest_cost_model_id,
    )


def _metrics_for_selections(
    policy_name: str,
    records: Sequence[EvaluationRecord],
    selections: Sequence[str],
    highest_cost_model_id: str,
    fallbacks: Sequence[bool] | None = None,
) -> PolicyMetrics:
    selected = [
        _candidate(record, model_id) for record, model_id in zip(records, selections, strict=True)
    ]
    return PolicyMetrics(
        policy_name=policy_name,
        quality_rate=mean(candidate.annotation.passed for candidate in selected),
        p95_latency_ms=float(np.percentile([candidate.latency_ms for candidate in selected], 95)),
        selection_counts=dict(Counter(selections)),
        high_cost_model_avoidance_rate=mean(
            model_id != highest_cost_model_id for model_id in selections
        ),
        fallback_rate=mean(fallbacks) if fallbacks else 0.0,
    )


def _candidate(record: EvaluationRecord, model_id: str) -> CandidateResult:
    try:
        return next(candidate for candidate in record.candidates if candidate.model_id == model_id)
    except StopIteration as exc:
        raise KeyError(f"record {record.request_id} has no candidate {model_id}") from exc


def _metadata_baseline(default_model_id: str) -> Callable[[EvaluationRecord], str]:
    return lambda record: record.metadata.get("baseline_model_id", default_model_id)


def _oracle_selector(bundle: EngineBundle) -> Callable[[EvaluationRecord], str]:
    fallback = bundle.fallback_model_id

    def select(record: EvaluationRecord) -> str:
        passing = [candidate for candidate in record.candidates if candidate.annotation.passed]
        if not passing:
            return fallback
        return min(passing, key=lambda item: item.cost_rank).model_id

    return select


def _brier_scores(bundle: EngineBundle, records: Sequence[EvaluationRecord]) -> dict[str, float]:
    scores: dict[str, float] = {}
    texts = [record.request_text for record in records]
    for model_id, estimator in bundle.estimators.items():
        probabilities, _ = estimator.predict(texts)
        labels = np.asarray(
            [int(_candidate(record, model_id).annotation.passed) for record in records], dtype=float
        )
        scores[model_id] = float(np.mean((probabilities - labels) ** 2))
    return scores
