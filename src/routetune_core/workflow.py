from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from routetune_core.contracts import EvaluationRecord
from routetune_core.dataset import split_records
from routetune_core.evaluation import EvaluationReport, evaluate
from routetune_core.model import EngineBundle, train_bundle


@dataclass(frozen=True)
class ExperimentResult:
    bundle: EngineBundle
    report: EvaluationReport
    train_count: int
    calibration_count: int
    test_count: int


def run_experiment(
    records: Sequence[EvaluationRecord],
    output_directory: Path,
    *,
    quality_threshold: float = 0.85,
    ood_coverage_threshold: float = 0.1,
    seed: int = 42,
) -> ExperimentResult:
    training, calibration, test = split_records(records, seed=seed)
    bundle = train_bundle(
        training,
        calibration,
        quality_threshold=quality_threshold,
        ood_coverage_threshold=ood_coverage_threshold,
        seed=seed,
    )
    report = evaluate(bundle, test)

    bundle_directory = output_directory / "bundle"
    bundle.save(bundle_directory)
    report.write(output_directory)
    split_manifest = {
        "seed": seed,
        "training_request_ids": [record.request_id for record in training],
        "calibration_request_ids": [record.request_id for record in calibration],
        "test_request_ids": [record.request_id for record in test],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "split.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ExperimentResult(
        bundle=bundle,
        report=report,
        train_count=len(training),
        calibration_count=len(calibration),
        test_count=len(test),
    )
