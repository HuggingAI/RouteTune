from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol


@dataclass(frozen=True)
class SensitiveFinding:
    entity_type: str
    start: int
    end: int
    confidence: float
    detector: str

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "detector": self.detector,
        }


@dataclass(frozen=True)
class RedactionResult:
    text: str
    findings: tuple[SensitiveFinding, ...]

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for finding in self.findings:
            result[finding.entity_type] = result.get(finding.entity_type, 0) + 1
        return result


class Detector(Protocol):
    def detect(self, text: str) -> Iterable[SensitiveFinding]: ...


class RegexDetector:
    """High precision detectors for the small set of secrets/PII needed by the MVP."""

    _patterns: tuple[tuple[str, re.Pattern[str], float], ...] = (
        (
            "BEARER_TOKEN",
            re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{12,}"),
            0.99,
        ),
        (
            "API_KEY",
            re.compile(r"(?i)\b(?:sk|api[_-]?key|token)[-_=: ]+[a-z0-9][a-z0-9._-]{11,}\b"),
            0.96,
        ),
        (
            "EMAIL",
            re.compile(r"(?<![\w.+-])[\w.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+"),
            0.98,
        ),
        (
            "CN_ID",
            re.compile(
                r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
                r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
            ),
            0.96,
        ),
        (
            "PHONE",
            re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"),
            0.96,
        ),
    )

    def detect(self, text: str) -> Iterable[SensitiveFinding]:
        for entity_type, pattern, confidence in self._patterns:
            for match in pattern.finditer(text):
                yield SensitiveFinding(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    detector="regex",
                )


class DictionaryDetector:
    def __init__(self, terms: Mapping[str, Sequence[str]]) -> None:
        entries: list[tuple[str, str]] = []
        for entity_type, values in terms.items():
            for value in values:
                normalized = value.strip()
                if normalized:
                    entries.append((entity_type.upper(), normalized))
        self._entries = tuple(sorted(entries, key=lambda item: len(item[1]), reverse=True))

    def detect(self, text: str) -> Iterable[SensitiveFinding]:
        folded = text.casefold()
        for entity_type, term in self._entries:
            needle = term.casefold()
            start = 0
            while (index := folded.find(needle, start)) >= 0:
                yield SensitiveFinding(
                    entity_type=entity_type,
                    start=index,
                    end=index + len(term),
                    confidence=1.0,
                    detector="dictionary",
                )
                start = index + len(term)


class ManagedDictionaryDetector:
    """A hot-reloadable dictionary detector for dictionaries managed by the product UI."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._detector = DictionaryDetector({})

    def replace(self, terms: Mapping[str, Sequence[str]]) -> None:
        detector = DictionaryDetector(terms)
        with self._lock:
            self._detector = detector

    def detect(self, text: str) -> Iterable[SensitiveFinding]:
        with self._lock:
            detector = self._detector
        return detector.detect(text)


class SpacyDetector:
    """Optional NER adapter. spaCy and its language model are not core dependencies."""

    def __init__(self, model_name: str, labels: Mapping[str, str] | None = None) -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("install RouteTune with the 'nlp' extra to use spaCy") from exc
        self._nlp = spacy.load(model_name)
        self._labels = dict(
            labels or {"PERSON": "PERSON", "ORG": "ORGANIZATION", "GPE": "LOCATION"}
        )

    def detect(self, text: str) -> Iterable[SensitiveFinding]:
        for entity in self._nlp(text).ents:
            entity_type = self._labels.get(entity.label_)
            if entity_type:
                yield SensitiveFinding(
                    entity_type=entity_type,
                    start=entity.start_char,
                    end=entity.end_char,
                    confidence=0.75,
                    detector="spacy",
                )


@dataclass(frozen=True)
class SecurityPolicy:
    blocked_types: frozenset[str] = field(default_factory=frozenset)


class SecurityPipeline:
    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        policy: SecurityPolicy | None = None,
    ) -> None:
        self._managed_dictionary = ManagedDictionaryDetector()
        self.detectors = (*tuple(detectors or (RegexDetector(),)), self._managed_dictionary)
        self.policy = policy or SecurityPolicy()

    def set_managed_dictionary_terms(self, terms: Mapping[str, Sequence[str]]) -> None:
        self._managed_dictionary.replace(terms)

    def redact(self, text: str) -> RedactionResult:
        findings = self._non_overlapping_findings(text)
        counters: dict[str, int] = {}
        pieces: list[str] = []
        cursor = 0
        for finding in findings:
            pieces.append(text[cursor : finding.start])
            counters[finding.entity_type] = counters.get(finding.entity_type, 0) + 1
            pieces.append(f"[{finding.entity_type}_{counters[finding.entity_type]}]")
            cursor = finding.end
        pieces.append(text[cursor:])
        return RedactionResult(text="".join(pieces), findings=findings)

    def blocked_types(self, result: RedactionResult) -> tuple[str, ...]:
        return tuple(
            sorted({item.entity_type for item in result.findings} & self.policy.blocked_types)
        )

    def _non_overlapping_findings(self, text: str) -> tuple[SensitiveFinding, ...]:
        candidates = [item for detector in self.detectors for item in detector.detect(text)]
        valid = [item for item in candidates if 0 <= item.start < item.end <= len(text)]
        chosen: list[SensitiveFinding] = []
        for item in sorted(
            valid,
            key=lambda value: (value.start, -(value.end - value.start), -value.confidence),
        ):
            if any(item.start < other.end and other.start < item.end for other in chosen):
                continue
            chosen.append(item)
        return tuple(sorted(chosen, key=lambda item: item.start))
