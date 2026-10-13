from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from routetune_core.contracts import EvaluationRecord, ModelProfile
from routetune_core.dataset import derive_model_profiles


@dataclass
class CandidateEstimator:
    model_id: str
    pipeline: Pipeline | None
    calibrator: LogisticRegression | None
    constant_probability: float | None

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        if self.pipeline is None:
            probability = float(self.constant_probability or 0.0)
            return np.full(len(texts), probability), np.ones(len(texts))

        base_probabilities = self.pipeline.predict_proba(texts)[:, 1]
        probabilities = _apply_calibrator(base_probabilities, self.calibrator)
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        vocabulary = vectorizer.vocabulary_
        analyzer = vectorizer.build_analyzer()
        coverage = []
        for text in texts:
            features = analyzer(text)
            if not features:
                coverage.append(0.0)
            else:
                coverage.append(sum(feature in vocabulary for feature in features) / len(features))
        return probabilities, np.asarray(coverage, dtype=float)


@dataclass
class EngineBundle:
    version: str
    created_at: str
    dataset_fingerprint: str
    quality_threshold: float
    ood_coverage_threshold: float
    profiles: dict[str, ModelProfile]
    estimators: dict[str, CandidateEstimator]
    fallback_model_id: str
    training_record_count: int
    calibration_record_count: int

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "bundle.joblib")
        manifest = {
            "version": self.version,
            "created_at": self.created_at,
            "dataset_fingerprint": self.dataset_fingerprint,
            "quality_threshold": self.quality_threshold,
            "ood_coverage_threshold": self.ood_coverage_threshold,
            "profiles": {key: asdict(value) for key, value in self.profiles.items()},
            "fallback_model_id": self.fallback_model_id,
            "training_record_count": self.training_record_count,
            "calibration_record_count": self.calibration_record_count,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> EngineBundle:
        bundle = joblib.load(directory / "bundle.joblib")
        if not isinstance(bundle, cls):
            raise TypeError("artifact is not a RouteTune EngineBundle")
        return bundle


def train_bundle(
    training_records: Sequence[EvaluationRecord],
    calibration_records: Sequence[EvaluationRecord],
    *,
    quality_threshold: float = 0.85,
    ood_coverage_threshold: float = 0.1,
    seed: int = 42,
) -> EngineBundle:
    if not 0 < quality_threshold <= 1:
        raise ValueError("quality_threshold must be between 0 and 1")
    if not 0 <= ood_coverage_threshold <= 1:
        raise ValueError("ood_coverage_threshold must be between 0 and 1")

    profiles = derive_model_profiles([*training_records, *calibration_records])
    fallback_model_id = _select_fallback_model([*training_records, *calibration_records], profiles)
    estimators: dict[str, CandidateEstimator] = {}

    for model_id in sorted(profiles):
        train_texts, train_labels = _examples_for_model(training_records, model_id)
        calibration_texts, calibration_labels = _examples_for_model(calibration_records, model_id)
        unique_labels = set(train_labels)
        if len(unique_labels) == 1:
            estimators[model_id] = CandidateEstimator(
                model_id=model_id,
                pipeline=None,
                calibrator=None,
                constant_probability=float(train_labels[0]),
            )
            continue

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 5),
                        min_df=2,
                        max_features=30_000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(train_texts, train_labels)
        calibrator = _fit_calibrator(pipeline, calibration_texts, calibration_labels, seed)
        estimators[model_id] = CandidateEstimator(
            model_id=model_id,
            pipeline=pipeline,
            calibrator=calibrator,
            constant_probability=None,
        )

    fingerprint = dataset_fingerprint([*training_records, *calibration_records])
    bundle_version = _bundle_version(
        fingerprint,
        quality_threshold=quality_threshold,
        ood_coverage_threshold=ood_coverage_threshold,
        seed=seed,
    )
    return EngineBundle(
        version=bundle_version,
        created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint=fingerprint,
        quality_threshold=quality_threshold,
        ood_coverage_threshold=ood_coverage_threshold,
        profiles=profiles,
        estimators=estimators,
        fallback_model_id=fallback_model_id,
        training_record_count=len(training_records),
        calibration_record_count=len(calibration_records),
    )


def create_bootstrap_bundle(model_cost_ranks: dict[str, int]) -> EngineBundle:
    """Create a safe cold-start bundle that sends every request to the highest cost rank."""
    if len(model_cost_ranks) < 2:
        raise ValueError("bootstrap requires at least two models")
    if any(
        not model_id.strip() or cost_rank <= 0 for model_id, cost_rank in model_cost_ranks.items()
    ):
        raise ValueError("bootstrap model ids and cost ranks must be valid")
    if len(set(model_cost_ranks.values())) != len(model_cost_ranks):
        raise ValueError("bootstrap cost ranks must be unique")
    configuration = json.dumps(model_cost_ranks, sort_keys=True, separators=(",", ":"))
    version = f"bootstrap-{hashlib.sha256(configuration.encode()).hexdigest()[:12]}"
    profiles = {
        model_id: ModelProfile(
            model_id=model_id,
            cost_rank=cost_rank,
            average_latency_ms=0,
            sample_count=0,
        )
        for model_id, cost_rank in model_cost_ranks.items()
    }
    estimators = {
        model_id: CandidateEstimator(
            model_id=model_id,
            pipeline=None,
            calibrator=None,
            constant_probability=0.0,
        )
        for model_id in model_cost_ranks
    }
    fallback_model_id = max(
        model_cost_ranks, key=lambda model_id: (model_cost_ranks[model_id], model_id)
    )
    return EngineBundle(
        version=version,
        created_at=datetime.now(UTC).isoformat(),
        dataset_fingerprint="cold-start/no-observations",
        quality_threshold=0.85,
        ood_coverage_threshold=0.1,
        profiles=profiles,
        estimators=estimators,
        fallback_model_id=fallback_model_id,
        training_record_count=0,
        calibration_record_count=0,
    )


def dataset_fingerprint(records: Sequence[EvaluationRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.request_id):
        digest.update(record.request_id.encode("utf-8"))
        digest.update(record.request_text.encode("utf-8"))
        for candidate in sorted(record.candidates, key=lambda item: item.model_id):
            digest.update(candidate.model_id.encode("utf-8"))
            digest.update(str(candidate.annotation.passed).encode("ascii"))
            digest.update(str(candidate.cost_rank).encode("ascii"))
    return digest.hexdigest()


def _bundle_version(
    fingerprint: str,
    *,
    quality_threshold: float,
    ood_coverage_threshold: float,
    seed: int,
) -> str:
    configuration = {
        "dataset_fingerprint": fingerprint,
        "estimator": "char-tfidf-logistic/v1",
        "fallback_policy": "highest-observed-quality-then-cost/v1",
        "quality_threshold": quality_threshold,
        "ood_coverage_threshold": ood_coverage_threshold,
        "seed": seed,
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"bundle-{hashlib.sha256(encoded).hexdigest()[:12]}"


def _select_fallback_model(
    records: Sequence[EvaluationRecord], profiles: dict[str, ModelProfile]
) -> str:
    pass_rates = {
        model_id: float(
            np.mean(
                [
                    next(
                        candidate.annotation.passed
                        for candidate in record.candidates
                        if candidate.model_id == model_id
                    )
                    for record in records
                ]
            )
        )
        for model_id in profiles
    }
    return max(
        profiles,
        key=lambda model_id: (
            pass_rates[model_id],
            profiles[model_id].cost_rank,
            model_id,
        ),
    )


def _examples_for_model(
    records: Sequence[EvaluationRecord], model_id: str
) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    for record in records:
        candidate = next(item for item in record.candidates if item.model_id == model_id)
        texts.append(record.request_text)
        labels.append(int(candidate.annotation.passed))
    return texts, labels


def _fit_calibrator(
    pipeline: Pipeline,
    texts: Sequence[str],
    labels: Sequence[int],
    seed: int,
) -> LogisticRegression | None:
    if len(texts) < 20 or len(set(labels)) < 2:
        return None
    probabilities = pipeline.predict_proba(texts)[:, 1]
    logits = _logit(probabilities).reshape(-1, 1)
    calibrator = LogisticRegression(random_state=seed)
    calibrator.fit(logits, labels)
    return calibrator


def _apply_calibrator(
    probabilities: np.ndarray, calibrator: LogisticRegression | None
) -> np.ndarray:
    if calibrator is None:
        return probabilities
    return calibrator.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.asarray([math.log(value / (1 - value)) for value in clipped])
