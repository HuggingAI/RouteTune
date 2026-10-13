from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean

from routetune_core.contracts import ContractError, EvaluationRecord, ModelProfile


def load_jsonl(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                record = EvaluationRecord.from_dict(payload)
            except (json.JSONDecodeError, ContractError, TypeError) as exc:
                raise ContractError(f"invalid record at {path}:{line_number}: {exc}") from exc
            if record.request_id in seen_ids:
                raise ContractError(
                    f"duplicate request_id at {path}:{line_number}: {record.request_id}"
                )
            seen_ids.add(record.request_id)
            records.append(record)
    if len(records) < 10:
        raise ContractError("at least 10 records are required")
    _validate_candidate_sets(records)
    return records


def write_jsonl(records: Iterable[EvaluationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _validate_candidate_sets(records: Sequence[EvaluationRecord]) -> None:
    expected = {candidate.model_id for candidate in records[0].candidates}
    for record in records[1:]:
        actual = {candidate.model_id for candidate in record.candidates}
        if actual != expected:
            raise ContractError(
                "all MVP records must contain the same candidate set; "
                f"request {record.request_id} has {sorted(actual)}, expected {sorted(expected)}"
            )
    profiles = derive_model_profiles(records)
    if len({profile.cost_rank for profile in profiles.values()}) != len(profiles):
        raise ContractError("candidate models must have unique cost_rank values")


def derive_model_profiles(records: Sequence[EvaluationRecord]) -> dict[str, ModelProfile]:
    cost_ranks: dict[str, set[int]] = defaultdict(set)
    latencies: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for candidate in record.candidates:
            cost_ranks[candidate.model_id].add(candidate.cost_rank)
            latencies[candidate.model_id].append(candidate.latency_ms)
    inconsistent = {model_id: ranks for model_id, ranks in cost_ranks.items() if len(ranks) != 1}
    if inconsistent:
        raise ContractError(f"cost_rank must be stable per model: {inconsistent}")
    return {
        model_id: ModelProfile(
            model_id=model_id,
            cost_rank=next(iter(model_ranks)),
            average_latency_ms=mean(latencies[model_id]),
            sample_count=len(latencies[model_id]),
        )
        for model_id, model_ranks in cost_ranks.items()
    }


def split_records(
    records: Sequence[EvaluationRecord],
    *,
    seed: int = 42,
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], list[EvaluationRecord]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < calibration_ratio < 1:
        raise ValueError("calibration_ratio must be between 0 and 1")
    if train_ratio + calibration_ratio >= 1:
        raise ValueError("train and calibration ratios must leave a test split")

    categories = [record.metadata.get("category") for record in records]
    if all(categories):
        grouped: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for record, category in zip(records, categories, strict=True):
            assert category is not None
            grouped[category].append(record)
        if all(len(group) >= 3 for group in grouped.values()):
            return _split_stratified(
                grouped,
                seed=seed,
                train_ratio=train_ratio,
                calibration_ratio=calibration_ratio,
            )

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_end = max(1, int(len(shuffled) * train_ratio))
    calibration_end = max(train_end + 1, int(len(shuffled) * (train_ratio + calibration_ratio)))
    calibration_end = min(calibration_end, len(shuffled) - 1)
    return shuffled[:train_end], shuffled[train_end:calibration_end], shuffled[calibration_end:]


def _split_stratified(
    grouped: Mapping[str, Sequence[EvaluationRecord]],
    *,
    seed: int,
    train_ratio: float,
    calibration_ratio: float,
) -> tuple[list[EvaluationRecord], list[EvaluationRecord], list[EvaluationRecord]]:
    rng = random.Random(seed)
    training: list[EvaluationRecord] = []
    calibration: list[EvaluationRecord] = []
    test: list[EvaluationRecord] = []
    for category in sorted(grouped):
        group = list(grouped[category])
        rng.shuffle(group)
        train_count = max(1, int(len(group) * train_ratio))
        calibration_count = max(1, int(len(group) * calibration_ratio))
        if train_count + calibration_count >= len(group):
            train_count = len(group) - 2
            calibration_count = 1
        training.extend(group[:train_count])
        calibration.extend(group[train_count : train_count + calibration_count])
        test.extend(group[train_count + calibration_count :])
    rng.shuffle(training)
    rng.shuffle(calibration)
    rng.shuffle(test)
    return training, calibration, test
