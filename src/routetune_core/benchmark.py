from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from routetune_core.contracts import ContractError

MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PRO_TEST_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
MMLU_PRO_TEST_URL = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/"
    f"{MMLU_PRO_REVISION}/data/test-00000-of-00001.parquet"
)
MMLU_PRO_LICENSE = "MIT"
PROMPT_VERSION = "multiple-choice-direct/v1"
CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    question: str
    options: tuple[str, ...]
    answer: str
    category: str
    source: str
    benchmark: str = "mmlu-pro"
    benchmark_revision: str = MMLU_PRO_REVISION

    def __post_init__(self) -> None:
        if not self.item_id.strip() or not self.question.strip():
            raise ContractError("benchmark item id and question must be non-empty")
        if not 2 <= len(self.options) <= len(CHOICES):
            raise ContractError("benchmark item must have between 2 and 26 options")
        if any(not option.strip() for option in self.options):
            raise ContractError("benchmark options must be non-empty")
        valid_answers = CHOICES[: len(self.options)]
        if self.answer not in valid_answers:
            raise ContractError(f"answer must be one of {valid_answers}")

    def messages(self) -> list[dict[str, str]]:
        options = "\n".join(
            f"{CHOICES[index]}. {option}" for index, option in enumerate(self.options)
        )
        return [
            {
                "role": "system",
                "content": (
                    "Answer the multiple-choice question. Return only the single letter of the "
                    "correct option, with no explanation or punctuation."
                ),
            },
            {"role": "user", "content": f"Question: {self.question}\n\nOptions:\n{options}"},
        ]

    def request_text(self) -> str:
        return self.messages()[1]["content"]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["options"] = list(self.options)
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkItem:
        options = data.get("options")
        if not isinstance(options, list):
            raise ContractError("benchmark options must be a list")
        return cls(
            item_id=str(data["item_id"]),
            question=str(data["question"]),
            options=tuple(str(option) for option in options),
            answer=str(data["answer"]).upper(),
            category=str(data.get("category", "unknown")),
            source=str(data.get("source", "unknown")),
            benchmark=str(data.get("benchmark", "mmlu-pro")),
            benchmark_revision=str(data.get("benchmark_revision", MMLU_PRO_REVISION)),
        )


def extract_choice(text: str, option_count: int) -> str | None:
    """Extract a deliberately terse multiple-choice response without guessing from prose."""
    without_thinking = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    stripped = without_thinking.strip().upper()
    valid = CHOICES[:option_count]
    if stripped in valid:
        return stripped
    if len(stripped) == 3 and stripped[0] == "(" and stripped[2] == ")" and stripped[1] in valid:
        return stripped[1]
    prefixes = ("ANSWER:", "ANSWER IS", "THE ANSWER IS")
    for prefix in prefixes:
        if stripped.startswith(prefix):
            remainder = stripped.removeprefix(prefix).strip().lstrip("(")
            if remainder and remainder[0] in valid:
                return remainder[0]
    return None


def item_from_mmlu_pro_row(row: Mapping[str, Any]) -> BenchmarkItem:
    raw_options = row.get("options")
    if not isinstance(raw_options, Sequence) or isinstance(raw_options, str):
        raise ContractError("MMLU-Pro row has invalid options")
    options = tuple(str(option).strip() for option in raw_options if str(option).strip() != "N/A")
    answer = str(row.get("answer", "")).strip().upper()
    question_id = row.get("question_id")
    return BenchmarkItem(
        item_id=f"mmlu-pro:{question_id}",
        question=str(row.get("question", "")).strip(),
        options=options,
        answer=answer,
        category=str(row.get("category", "unknown")),
        source=str(row.get("src", "unknown")),
    )


def download_mmlu_pro(destination: Path, *, opener: Any = urllib.request.urlopen) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(MMLU_PRO_TEST_URL, headers={"User-Agent": "RouteTune/0.1"})
    digest = hashlib.sha256()
    try:
        with opener(request, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != MMLU_PRO_TEST_SHA256:
            raise ContractError(
                "MMLU-Pro checksum mismatch; refusing to use an unverified dataset file"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_mmlu_pro(
    path: Path,
    *,
    max_items: int | None = None,
    categories: set[str] | None = None,
    seed: int = 42,
) -> list[BenchmarkItem]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != MMLU_PRO_TEST_SHA256:
        raise ContractError("MMLU-Pro file checksum does not match the pinned dataset revision")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "MMLU-Pro loading requires the benchmark extra: uv sync --extra benchmark"
        ) from exc

    rows = parquet.read_table(path).to_pylist()
    items = [item_from_mmlu_pro_row(row) for row in rows]
    if categories:
        normalized = {category.casefold() for category in categories}
        items = [item for item in items if item.category.casefold() in normalized]
    random.Random(seed).shuffle(items)
    if max_items is not None:
        if max_items <= 0:
            raise ContractError("max_items must be positive")
        items = items[:max_items]
    return items


def write_benchmark_jsonl(items: Iterable[BenchmarkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_benchmark_jsonl(path: Path) -> list[BenchmarkItem]:
    items: list[BenchmarkItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(BenchmarkItem.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ContractError) as exc:
                raise ContractError(
                    f"invalid benchmark item at {path}:{line_number}: {exc}"
                ) from exc
    return items


def copy_verified_mmlu_pro(source: Path, destination: Path) -> Path:
    """Copy an already-downloaded official file while retaining the same integrity check."""
    if hashlib.sha256(source.read_bytes()).hexdigest() != MMLU_PRO_TEST_SHA256:
        raise ContractError("source file is not the pinned MMLU-Pro test file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination
