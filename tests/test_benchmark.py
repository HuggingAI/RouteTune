from pathlib import Path

import pytest

from routetune_core.api_eval import (
    build_evaluation_records,
    load_traces,
    run_api_evaluation,
)
from routetune_core.benchmark import BenchmarkItem, extract_choice, item_from_mmlu_pro_row
from routetune_core.providers import (
    ApiModelConfig,
    Completion,
    ProviderConfig,
    ProviderError,
    _parse_completion,
    load_model_configs,
)


def test_mmlu_row_and_conservative_answer_extraction() -> None:
    item = item_from_mmlu_pro_row(
        {
            "question_id": 7,
            "question": "2 + 2 = ?",
            "options": ["3", "4", "N/A"],
            "answer": "B",
            "category": "math",
            "src": "test",
        }
    )
    assert item.item_id == "mmlu-pro:7"
    assert item.options == ("3", "4")
    assert extract_choice("B", 2) == "B"
    assert extract_choice("(B)", 2) == "B"
    assert extract_choice("The answer is B.", 2) == "B"
    assert extract_choice("<think>considering A and B</think>\nB", 2) == "B"
    assert extract_choice("I considered A but selected B", 2) is None


def test_model_config_contains_env_name_not_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_KEY", "super-secret")
    config = tmp_path / "models.toml"
    config.write_text(
        """
[providers.test]
base_url = "https://example.test/v1"
api_key_env = "TEST_KEY"

[models.a]
provider = "test"
api_model = "small"
cost_rank = 1

[models.b]
provider = "test"
api_model = "large"
cost_rank = 2
""".strip(),
        encoding="utf-8",
    )
    models = load_model_configs(config)
    assert [model.model_id for model in models] == ["a", "b"]
    assert models[0].provider.api_key_env == "TEST_KEY"
    assert "super-secret" not in repr(models)
    assert [model.cost_rank for model in models] == [1, 2]


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, model: ApiModelConfig, messages: list[dict[str, str]]) -> Completion:
        self.calls += 1
        answer = "B" if model.model_id == "strong" else "A"
        return Completion(
            content=answer,
            reasoning_content=None,
            prompt_tokens=100,
            completion_tokens=1,
            request_id=f"request-{self.calls}",
            latency_ms=20,
        )


def test_api_eval_traces_inputs_outputs_and_resumes(tmp_path: Path) -> None:
    provider = ProviderConfig("fake", "https://example.test/v1", "IGNORED", {})
    models = [
        ApiModelConfig("cheap", provider, "cheap-api", 1),
        ApiModelConfig("strong", provider, "strong-api", 2),
    ]
    items = [
        BenchmarkItem(
            item_id=f"mmlu-pro:{index}",
            question=f"Question {index}",
            options=("wrong", "right"),
            answer="B",
            category="test",
            source="unit",
        )
        for index in range(10)
    ]
    trace_path = tmp_path / "traces.jsonl"
    client = FakeClient()

    first = run_api_evaluation(
        items,
        models,
        trace_path,
        client=client,  # type: ignore[arg-type]
        max_requests=20,
    )
    second = run_api_evaluation(
        items,
        models,
        trace_path,
        client=client,  # type: ignore[arg-type]
        max_requests=20,
    )

    traces = load_traces(trace_path)
    records = build_evaluation_records(items, models, traces)
    assert first.complete_item_count == 10
    assert first.new_request_count == 20
    assert second.new_request_count == 0
    assert second.cached_result_count == 20
    assert len(traces) == 20
    assert len(records) == 10
    assert records[0].candidates[0].annotation.passed is False
    assert records[0].candidates[1].annotation.passed is True
    assert traces[0].input_messages[1]["content"].startswith("Question:")
    assert traces[0].raw_output in {"A", "B"}


def test_provider_reports_reasoning_without_final_answer() -> None:
    provider = ProviderConfig("test", "https://example.test/v1", "", {})
    model = ApiModelConfig("test", provider, "test", 1)

    with pytest.raises(ProviderError, match="reasoning but no final answer"):
        _parse_completion(
            {
                "id": "request-1",
                "choices": [{"message": {"content": None, "reasoning": "unfinished"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 8},
            },
            model,
            1,
        )


def test_provider_rejects_empty_final_answer() -> None:
    provider = ProviderConfig("test", "https://example.test/v1", "", {})
    model = ApiModelConfig("test", provider, "test", 1)

    with pytest.raises(ProviderError, match="empty final answer"):
        _parse_completion(
            {
                "id": "request-1",
                "choices": [{"message": {"content": ""}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0},
            },
            model,
            1,
        )
