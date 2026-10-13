from __future__ import annotations

from pathlib import Path

from routetune_core.adapters import ChatMessage, ChatResponse, GenerationOptions
from routetune_core.batches import BatchManager, BatchStore
from routetune_core.contracts import Annotation
from routetune_core.label_router import LabelRouter, create_cold_start_label_bundle
from routetune_core.model_pool import ModelPoolEntry, ModelPoolStore
from routetune_core.runtime import RouteTuneRuntime
from routetune_core.security import SecurityPipeline
from routetune_core.trace_store import TraceStore


class PoolFakeAdapter:
    def __init__(self, model_ids: set[str], failing_model_ids: set[str] | None = None) -> None:
        self._model_ids = frozenset(model_ids)
        self._failing_model_ids = frozenset(failing_model_ids or set())
        self.calls: list[str] = []
        self.options: list[GenerationOptions | None] = []

    @property
    def model_ids(self) -> frozenset[str]:
        return self._model_ids

    def complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> ChatResponse:
        self.calls.append(model_id)
        self.options.append(options)
        if model_id in self._failing_model_ids:
            raise RuntimeError("provider unavailable")
        return ChatResponse(
            content=f"answer from {model_id}",
            reasoning_content=None,
            prompt_tokens=5,
            completion_tokens=4,
            provider_request_id=f"provider-{model_id}",
            latency_ms=1,
        )


def _entry(model_id: str, cost_rank: int, route_label: str = "tier_5") -> ModelPoolEntry:
    return ModelPoolEntry(
        model_id=model_id,
        provider_id="test",
        base_url="https://example.test/v1",
        api_key_env="TEST_API_KEY",
        api_model=model_id,
        cost_rank=cost_rank,
        max_tokens=100,
        route_label=route_label,
        tags=("test",),
    )


def _label_runtime(
    tmp_path: Path,
    failing_model_ids: set[str] | None = None,
) -> tuple[RouteTuneRuntime, BatchManager, PoolFakeAdapter, TraceStore]:
    database = tmp_path / "product.db"
    pool = ModelPoolStore(database)
    pool.create(_entry("advanced-a", 10))
    pool.create(_entry("advanced-b", 20))
    router = LabelRouter(create_cold_start_label_bundle())
    adapter = PoolFakeAdapter(
        {"advanced-a", "advanced-b"}, failing_model_ids=failing_model_ids
    )
    traces = TraceStore(database)
    security = SecurityPipeline()
    runtime = RouteTuneRuntime(
        router=router,
        adapter=adapter,
        security=security,
        traces=traces,
        model_pool=pool,
    )
    manager = BatchManager(
        store=BatchStore(database),
        runtime=runtime,
        router=router,
        model_pool=pool,
        security=security,
        traces=traces,
    )
    return runtime, manager, adapter, traces


def test_label_router_resolves_lowest_cost_model_inside_tier(tmp_path: Path) -> None:
    runtime, _, adapter, traces = _label_runtime(tmp_path)

    response = runtime.complete([ChatMessage(role="user", content="hello")])
    trace = traces.get(response.trace_id)

    assert response.route_label == "tier_5"
    assert response.selected_model_id == "advanced-a"
    assert adapter.calls == ["advanced-a"]
    assert trace is not None
    assert trace["selected_route_label"] == "tier_5"


def test_evaluation_only_model_is_not_selected_for_online_traffic(tmp_path: Path) -> None:
    database = tmp_path / "product.db"
    pool = ModelPoolStore(database)
    pool.create(_entry("online-model", 10))
    evaluation_model = _entry("evaluation-model", 1)
    pool.create(
        ModelPoolEntry(
            **{
                **evaluation_model.__dict__,
                "tags": ("test", "evaluation-only"),
            }
        )
    )
    adapter = PoolFakeAdapter({"online-model", "evaluation-model"})
    runtime = RouteTuneRuntime(
        router=LabelRouter(create_cold_start_label_bundle()),
        adapter=adapter,
        security=SecurityPipeline(),
        traces=TraceStore(database),
        model_pool=pool,
    )

    response = runtime.complete([ChatMessage(role="user", content="hello")])

    assert response.selected_model_id == "online-model"
    assert adapter.calls == ["online-model"]


def test_route_batch_imports_messages_runs_and_preserves_retry_history(tmp_path: Path) -> None:
    _, manager, adapter, _ = _label_runtime(tmp_path)
    content = "\n".join(
        [
            '{"id":"q1","messages":[{"role":"user","content":"hello"}],'
            '"max_tokens":321,"temperature":0.4}',
            '{"id":"q2","messages":[{"role":"user","content":"email me at a@example.com"}]}',
        ]
    )
    batch = manager.import_jsonl(name="test", content=content, mode="route")

    assert batch["item_count"] == 2
    assert batch["items"][0]["assigned_model_id"] == "advanced-a"
    assert "a@example.com" not in batch["items"][1]["messages"][0]["content"]

    manager.run(batch["batch_id"])
    completed = manager.details(batch["batch_id"])
    assert completed is not None
    assert all(item["status"] == "completed" for item in completed["items"])
    first_item = completed["items"][0]
    manager.retry_item(batch["batch_id"], first_item["item_id"])
    retried = manager.details(batch["batch_id"])
    assert retried is not None
    assert [run["requested_model_id"] for run in retried["items"][0]["runs"]] == [
        "advanced-a",
        "advanced-b",
    ]
    assert adapter.calls.count("advanced-a") == 2
    assert adapter.calls.count("advanced-b") == 1
    assert adapter.options[0] == GenerationOptions(max_tokens=321, temperature=0.4)


def test_compare_batch_uses_explicit_candidate_models(tmp_path: Path) -> None:
    _, manager, adapter, traces = _label_runtime(tmp_path)
    manager.model_pool.create(_entry("advanced-c", 30))

    batch = manager.import_jsonl(
        name="comparison",
        content='{"id":"q1","messages":[{"role":"user","content":"hello"}]}',
        mode="compare",
        model_ids=["advanced-a", "advanced-b"],
        max_tokens=512,
        temperature=0,
    )

    assert set(batch["model_ids"]) == {"advanced-a", "advanced-b"}
    assert "advanced-c" not in batch["model_ids"]

    manager.run(batch["batch_id"])
    completed = manager.details(batch["batch_id"])
    assert completed is not None
    assert completed["progress"] == {
        "expected": 2,
        "finished": 2,
        "failed": 0,
        "successful": 2,
        "annotated": 0,
        "manual_expected": 2,
        "technical_failed": 0,
        "train_ready": False,
    }
    assert adapter.options == [
        GenerationOptions(max_tokens=512, temperature=0),
        GenerationOptions(max_tokens=512, temperature=0),
    ]
    for run in completed["items"][0]["runs"]:
        traces.annotate(run["trace_id"], Annotation(passed=True))

    annotated = manager.details(batch["batch_id"])
    assert annotated is not None
    assert annotated["progress"]["annotated"] == 2
    assert annotated["progress"]["train_ready"] is True
    summary = next(
        item for item in manager.store.list_batches() if item["batch_id"] == batch["batch_id"]
    )
    assert summary["finished_run_count"] == 2
    assert summary["annotated_run_count"] == 2


def test_technical_failures_do_not_deadlock_router_training(tmp_path: Path) -> None:
    _, manager, _, traces = _label_runtime(tmp_path, failing_model_ids={"advanced-b"})
    content = "\n".join(
        f'{{"id":"q{index}","messages":[{{"role":"user","content":"request {index}"}}]}}'
        for index in range(10)
    )
    batch = manager.import_jsonl(
        name="technical failures",
        content=content,
        mode="compare",
        model_ids=["advanced-a", "advanced-b"],
    )

    manager.run(batch["batch_id"])
    completed = manager.details(batch["batch_id"])
    assert completed is not None
    for item in completed["items"]:
        successful = next(run for run in item["runs"] if run["status"] == "completed")
        traces.annotate(successful["trace_id"], Annotation(passed=True))

    annotated = manager.details(batch["batch_id"])
    assert annotated is not None
    assert annotated["progress"]["technical_failed"] == 10
    assert annotated["progress"]["annotated"] == 10
    assert annotated["progress"]["manual_expected"] == 10
    assert annotated["progress"]["train_ready"] is True
    trained = manager.train_candidate_bundle(batch["batch_id"])
    assert trained["training_record_count"] == 10
    assert trained["requested_strategy"] == "auto"
    assert trained["trainer_id"] == "tfidf_logreg"
