from pathlib import Path

import pytest

from routetune_core.contracts import Annotation, CandidateResult, ContractError, EvaluationRecord
from routetune_core.dataset import load_jsonl, split_records, write_jsonl
from routetune_core.demo_data import generate_demo_records


def test_annotation_rejects_invalid_score() -> None:
    with pytest.raises(ContractError, match="between 1 and 5"):
        Annotation(passed=True, score=6)


def test_record_requires_unique_candidate_models() -> None:
    candidate = CandidateResult(
        model_id="same",
        annotation=Annotation(passed=True),
        cost_rank=1,
        latency_ms=10,
        output_hash="abc",
    )
    with pytest.raises(ContractError, match="unique"):
        EvaluationRecord(
            request_id="req",
            request_text="hello",
            candidates=(candidate, candidate),
        )


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = generate_demo_records(count=30, seed=7)
    path = tmp_path / "records.jsonl"
    write_jsonl(records, path)
    loaded = load_jsonl(path)
    assert loaded == records


def test_split_never_leaks_request_ids() -> None:
    records = generate_demo_records(count=60, seed=7)
    training, calibration, test = split_records(records, seed=11)
    train_ids = {record.request_id for record in training}
    calibration_ids = {record.request_id for record in calibration}
    test_ids = {record.request_id for record in test}
    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)
    assert train_ids | calibration_ids | test_ids == {record.request_id for record in records}


def test_split_stratifies_public_benchmark_categories() -> None:
    records = generate_demo_records(count=42, seed=5)
    categorized = [
        EvaluationRecord(
            request_id=record.request_id,
            request_text=record.request_text,
            candidates=record.candidates,
            metadata={"category": f"category-{index % 2}"},
        )
        for index, record in enumerate(records)
    ]
    training, calibration, test = split_records(categorized, seed=5)
    for split in (training, calibration, test):
        assert {record.metadata["category"] for record in split} == {
            "category-0",
            "category-1",
        }
