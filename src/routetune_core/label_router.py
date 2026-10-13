from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from routetune_core.contracts import EvaluationRecord

DEFAULT_ROUTE_LABELS = ("tier_1", "tier_2", "tier_3", "tier_4", "tier_5")
AVAILABLE_TRAINERS = ("tfidf_logreg",)


@dataclass(frozen=True)
class LabelCandidatePrediction:
    route_label: str
    probability: float


@dataclass(frozen=True)
class LabelRoutingDecision:
    request_id: str
    selected_route_label: str
    candidates: tuple[LabelCandidatePrediction, ...]
    confidence: float
    vocabulary_coverage: float
    used_fallback: bool
    reason: str
    bundle_version: str


@dataclass
class LabelEngineBundle:
    version: str
    created_at: str
    route_labels: tuple[str, ...]
    fallback_label: str
    confidence_threshold: float
    ood_coverage_threshold: float
    pipeline: Pipeline | None
    constant_label: str | None
    training_record_count: int
    trainer_id: str = "tfidf_logreg"

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "label-bundle.joblib")
        manifest = {
            "version": self.version,
            "created_at": self.created_at,
            "route_labels": self.route_labels,
            "fallback_label": self.fallback_label,
            "confidence_threshold": self.confidence_threshold,
            "ood_coverage_threshold": self.ood_coverage_threshold,
            "training_record_count": self.training_record_count,
            "router_family": "probabilistic-classifier",
            "trainer_id": getattr(self, "trainer_id", "tfidf_logreg"),
            "classifier": "char-tfidf-logistic/v1" if self.pipeline else "constant/v1",
        }
        (directory / "label-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> LabelEngineBundle:
        bundle = joblib.load(directory / "label-bundle.joblib")
        if not isinstance(bundle, cls):
            raise TypeError("artifact is not a RouteTune LabelEngineBundle")
        return bundle


class LabelRouter:
    def __init__(self, bundle: LabelEngineBundle) -> None:
        self.bundle = bundle

    def decide(self, request_id: str, request_text: str) -> LabelRoutingDecision:
        probabilities, coverage = self._predict(request_text)
        selected = max(probabilities, key=probabilities.get)
        confidence = probabilities[selected]
        used_fallback = False
        reason = "CLASSIFIER_LABEL"
        if coverage < self.bundle.ood_coverage_threshold:
            selected = self.bundle.fallback_label
            used_fallback = True
            reason = "OOD_FALLBACK_TO_HIGHEST_TIER"
        elif confidence < self.bundle.confidence_threshold:
            selected = self.bundle.fallback_label
            used_fallback = True
            reason = "LOW_CONFIDENCE_FALLBACK_TO_HIGHEST_TIER"
        return LabelRoutingDecision(
            request_id=request_id,
            selected_route_label=selected,
            candidates=tuple(
                LabelCandidatePrediction(route_label=label, probability=probabilities[label])
                for label in self.bundle.route_labels
            ),
            confidence=confidence,
            vocabulary_coverage=coverage,
            used_fallback=used_fallback,
            reason=reason,
            bundle_version=self.bundle.version,
        )

    def _predict(self, text: str) -> tuple[dict[str, float], float]:
        if self.bundle.pipeline is None:
            probabilities = {
                label: float(label == self.bundle.constant_label)
                for label in self.bundle.route_labels
            }
            return probabilities, 1.0
        pipeline = self.bundle.pipeline
        predicted = pipeline.predict_proba([text])[0]
        classifier: LogisticRegression = pipeline.named_steps["classifier"]
        probabilities = {label: 0.0 for label in self.bundle.route_labels}
        for label, probability in zip(classifier.classes_, predicted, strict=True):
            probabilities[str(label)] = float(probability)
        vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
        features = vectorizer.build_analyzer()(text)
        coverage = (
            sum(feature in vectorizer.vocabulary_ for feature in features) / len(features)
            if features
            else 0.0
        )
        return probabilities, float(coverage)


def create_cold_start_label_bundle(
    route_labels: Sequence[str] = DEFAULT_ROUTE_LABELS,
    *,
    fallback_label: str = "tier_5",
) -> LabelEngineBundle:
    labels = _validate_labels(route_labels, fallback_label)
    encoded = json.dumps(labels, separators=(",", ":")).encode()
    return LabelEngineBundle(
        version=f"label-bootstrap-{hashlib.sha256(encoded).hexdigest()[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        route_labels=labels,
        fallback_label=fallback_label,
        confidence_threshold=0.55,
        ood_coverage_threshold=0.1,
        pipeline=None,
        constant_label=fallback_label,
        training_record_count=0,
        trainer_id="constant",
    )


def train_label_bundle(
    records: Sequence[EvaluationRecord],
    model_route_labels: Mapping[str, str],
    *,
    route_labels: Sequence[str] = DEFAULT_ROUTE_LABELS,
    fallback_label: str = "tier_5",
    confidence_threshold: float = 0.55,
    ood_coverage_threshold: float = 0.1,
    seed: int = 42,
    strategy: str = "auto",
) -> LabelEngineBundle:
    labels = _validate_labels(route_labels, fallback_label)
    if not records:
        raise ValueError("label training requires evaluation records")
    trainer_id = resolve_training_strategy(strategy, len(records))
    targets = [
        _target_label(record, model_route_labels, labels, fallback_label) for record in records
    ]
    pipeline: Pipeline | None = None
    constant_label: str | None = None
    if len(set(targets)) == 1:
        constant_label = targets[0]
    else:
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
                    LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=seed),
                ),
            ]
        )
        pipeline.fit([record.request_text for record in records], targets)
    fingerprint_payload = {
        "trainer_id": trainer_id,
        "records": [
            (record.request_id, record.request_text, target)
            for record, target in zip(records, targets, strict=True)
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return LabelEngineBundle(
        version=f"label-bundle-{fingerprint[:12]}",
        created_at=datetime.now(UTC).isoformat(),
        route_labels=labels,
        fallback_label=fallback_label,
        confidence_threshold=confidence_threshold,
        ood_coverage_threshold=ood_coverage_threshold,
        pipeline=pipeline,
        constant_label=constant_label,
        training_record_count=len(records),
        trainer_id=trainer_id,
    )


def resolve_training_strategy(strategy: str, record_count: int) -> str:
    """Resolve the user-facing strategy without coupling callers to trainer internals."""
    if strategy == "auto":
        # The first production trainer is deliberately conservative for small datasets.
        # This decision point can select FastText or an encoder once those plugins exist.
        return "tfidf_logreg"
    if strategy not in AVAILABLE_TRAINERS:
        raise ValueError(f"router training strategy is not available: {strategy}")
    return strategy


def training_strategies() -> list[dict[str, object]]:
    return [
        {
            "strategy": "auto",
            "name": "Auto",
            "available": True,
            "recommended": True,
            "description": "根据样本量和已安装训练器自动选择；当前会选择 TF-IDF。",
        },
        {
            "strategy": "tfidf_logreg",
            "name": "TF-IDF + Logistic Regression",
            "available": True,
            "recommended": False,
            "description": "适合小数据冷启动，训练快、CPU 可部署、结果可解释。",
        },
        {
            "strategy": "fasttext",
            "name": "FastText",
            "available": False,
            "recommended": False,
            "description": "面向更大规模短文本与多语言数据的后续训练器。",
        },
        {
            "strategy": "bert",
            "name": "BERT / MiniLM Encoder",
            "available": False,
            "recommended": False,
            "description": "准确率优先的后续训练器，需要额外模型与训练资源。",
        },
    ]


def _target_label(
    record: EvaluationRecord,
    model_route_labels: Mapping[str, str],
    route_labels: tuple[str, ...],
    fallback_label: str,
) -> str:
    passed_labels = {
        model_route_labels[candidate.model_id]
        for candidate in record.candidates
        if candidate.annotation.passed and candidate.model_id in model_route_labels
    }
    for label in route_labels:
        if label in passed_labels:
            return label
    return fallback_label


def _validate_labels(route_labels: Sequence[str], fallback_label: str) -> tuple[str, ...]:
    labels = tuple(route_labels)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("route labels must contain at least two unique values")
    if any(not label.strip() for label in labels):
        raise ValueError("route labels must not be blank")
    if fallback_label not in labels:
        raise ValueError("fallback label must be one of the route labels")
    return labels
