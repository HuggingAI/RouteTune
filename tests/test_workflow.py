from pathlib import Path

from routetune_core.contracts import Annotation, CandidateResult, EvaluationRecord
from routetune_core.demo_data import generate_demo_records
from routetune_core.model import EngineBundle, train_bundle
from routetune_core.workflow import run_experiment


def test_demo_proves_quality_constrained_downrouting(tmp_path: Path) -> None:
    records = generate_demo_records(count=600, seed=42)
    result = run_experiment(records, tmp_path, seed=42, quality_threshold=0.85)
    policies = {policy.policy_name: policy for policy in result.report.policies}

    assert result.report.high_cost_model_avoidance_rate > 0.20
    assert policies["routetune"].quality_rate >= 0.85
    assert (
        policies["oracle"].high_cost_model_avoidance_rate
        >= policies["routetune"].high_cost_model_avoidance_rate
    )
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "bundle" / "manifest.json").exists()
    assert EngineBundle.load(tmp_path / "bundle").version == result.bundle.version


def test_policy_threshold_changes_bundle_version(tmp_path: Path) -> None:
    records = generate_demo_records(count=60, seed=9)
    lower = run_experiment(records, tmp_path / "lower", seed=9, quality_threshold=0.85)
    higher = run_experiment(records, tmp_path / "higher", seed=9, quality_threshold=0.95)
    assert lower.bundle.dataset_fingerprint == higher.bundle.dataset_fingerprint
    assert lower.bundle.version != higher.bundle.version


def test_fallback_is_selected_by_observed_quality_not_cost_rank(tmp_path: Path) -> None:
    records = [
        EvaluationRecord(
            request_id=f"record-{index}",
            request_text=f"question {index}",
            candidates=(
                CandidateResult("expensive", Annotation(False), 10, 100, f"e-{index}"),
                CandidateResult("reliable", Annotation(True), 1, 100, f"r-{index}"),
            ),
        )
        for index in range(30)
    ]
    bundle = train_bundle(records[:20], records[20:])
    assert bundle.fallback_model_id == "reliable"
