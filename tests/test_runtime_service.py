from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from routetune_core.adapters import ChatMessage, ChatResponse, GenerationOptions
from routetune_core.contracts import ModelProfile
from routetune_core.model import CandidateEstimator, EngineBundle
from routetune_core.router import Router
from routetune_core.runtime import RouteTuneRuntime
from routetune_core.security import SecurityPipeline, SecurityPolicy
from routetune_core.service import create_app
from routetune_core.trace_store import TraceStore


def _bundle(*, economy_probability: float = 0.95) -> EngineBundle:
    return EngineBundle(
        version="test-bundle",
        created_at="2026-10-08T00:00:00+00:00",
        dataset_fingerprint="test",
        quality_threshold=0.85,
        ood_coverage_threshold=0.0,
        profiles={
            "economy": ModelProfile("economy", 1, 100, 10),
            "premium": ModelProfile("premium", 2, 300, 10),
        },
        estimators={
            "economy": CandidateEstimator("economy", None, None, economy_probability),
            "premium": CandidateEstimator("premium", None, None, 0.99),
        },
        fallback_model_id="premium",
        training_record_count=10,
        calibration_record_count=10,
    )


class FakeAdapter:
    def __init__(self, *, fail_models: set[str] | None = None) -> None:
        self.fail_models = fail_models or set()
        self.calls: list[tuple[str, list[ChatMessage]]] = []
        self.options: list[GenerationOptions | None] = []

    @property
    def model_ids(self) -> frozenset[str]:
        return frozenset({"economy", "premium"})

    def complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> ChatResponse:
        self.calls.append((model_id, list(messages)))
        self.options.append(options)
        if model_id in self.fail_models:
            raise RuntimeError("simulated upstream failure with secret detail")
        return ChatResponse(
            content="处理完成，请发给 result@example.com",
            reasoning_content=None,
            prompt_tokens=12,
            completion_tokens=6,
            provider_request_id="provider-1",
            latency_ms=5,
        )


def _runtime(tmp_path: Path, adapter: FakeAdapter) -> tuple[RouteTuneRuntime, TraceStore]:
    store = TraceStore(tmp_path / "routetune.db")
    return (
        RouteTuneRuntime(
            router=Router(_bundle()),
            adapter=adapter,
            security=SecurityPipeline(),
            traces=store,
        ),
        store,
    )


def test_http_request_is_redacted_routed_traced_and_annotated(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runtime, store = _runtime(tmp_path, adapter)
    client = TestClient(create_app(runtime, store))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "联系我 13800138000"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "economy"
    assert "result@example.com" not in body["choices"][0]["message"]["content"]
    assert adapter.calls[0][1][0].content == "联系我 [PHONE_1]"

    trace_id = body["routetune"]["trace_id"]
    trace = client.get(f"/v1/traces/{trace_id}").json()
    assert "13800138000" not in trace["input_text"]
    assert trace["status"] == "completed"
    assert trace["input_findings"][0]["entity_type"] == "PHONE"

    annotation = client.post(
        f"/v1/traces/{trace_id}/annotations",
        json={"passed": True, "score": 5, "annotator_id": "tester"},
    )
    assert annotation.status_code == 200
    assert annotation.json()["output_hash"] == trace["output_hash"]
    duplicate = client.post(
        f"/v1/traces/{trace_id}/annotations",
        json={"passed": True, "score": 5, "annotator_id": "tester"},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["annotation_id"] == annotation.json()["annotation_id"]
    runtime.complete([ChatMessage(role="user", content="batch only")], trace_source="batch")
    stats = client.get("/v1/stats?window=24h").json()
    assert stats["annotations"]["pass_rate"] == 1.0
    assert stats["annotations"]["total"] == 1
    assert stats["requests"]["success_rate"] == 1.0
    assert stats["requests"]["p95_latency_ms"] > 0
    assert stats["requests"]["p95_latency_ms"] == stats["requests"]["p50_latency_ms"]
    assert stats["tokens"] == {"prompt": 12, "completion": 6, "total": 18}
    assert stats["models"][0]["model_id"] == "economy"
    assert stats["models"][0]["completed"] == 1
    assert stats["timeline"][0]["requests"] == 1
    assert stats["request_source"] == "api"
    assert client.get("/v1/stats?window=24h&source=all").json()["requests"]["total"] == 2
    assert client.get("/v1/stats?window=invalid").status_code == 422


def test_runtime_falls_back_and_records_actual_model(tmp_path: Path) -> None:
    adapter = FakeAdapter(fail_models={"economy"})
    runtime, store = _runtime(tmp_path, adapter)

    result = runtime.complete([ChatMessage(role="user", content="hello")])
    trace = store.get(result.trace_id)

    assert [call[0] for call in adapter.calls] == ["economy", "premium"]
    assert result.provider_fallback
    assert result.selected_model_id == "economy"
    assert result.actual_model_id == "premium"
    assert trace is not None
    assert trace["fallback_reason"] == "RUNTIMEERROR"


def test_security_policy_blocks_before_provider_call(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    store = TraceStore(tmp_path / "blocked.db")
    runtime = RouteTuneRuntime(
        router=Router(_bundle()),
        adapter=adapter,
        security=SecurityPipeline(policy=SecurityPolicy(blocked_types=frozenset({"API_KEY"}))),
        traces=store,
    )
    client = TestClient(create_app(runtime, store))

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "token=abc1234567890123"}]},
    )

    assert response.status_code == 422
    assert adapter.calls == []
    trace = store.list()[0]
    assert trace["status"] == "blocked"
    assert "abc1234567890123" not in (trace["input_text"] or "")


def test_admin_console_and_health_are_available(tmp_path: Path) -> None:
    runtime, store = _runtime(tmp_path, FakeAdapter())
    client = TestClient(create_app(runtime, store))

    assert client.get("/health").json()["status"] == "ok"
    admin = client.get("/admin")
    assert admin.status_code == 200
    assert "RouteTune Console" in admin.text


def test_api_key_protects_business_endpoints(tmp_path: Path, monkeypatch: object) -> None:
    runtime, store = _runtime(tmp_path, FakeAdapter())
    monkeypatch.setenv("TEST_ROUTETUNE_API_KEY", "local-secret")  # type: ignore[attr-defined]
    client = TestClient(create_app(runtime, store, api_key_env="TEST_ROUTETUNE_API_KEY"))

    assert client.get("/health").status_code == 200
    assert client.get("/v1/stats").status_code == 401
    assert (
        client.get("/v1/stats", headers={"Authorization": "Bearer local-secret"}).status_code == 200
    )


def test_api_rejects_multi_turn_messages(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runtime, store = _runtime(tmp_path, adapter)
    client = TestClient(create_app(runtime, store))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ]
        },
    )

    assert response.status_code == 422
    assert adapter.calls == []


def test_generation_options_belong_to_each_request(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    runtime, store = _runtime(tmp_path, adapter)
    client = TestClient(create_app(runtime, store))

    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 321,
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
        },
    )

    assert response.status_code == 200
    assert adapter.options == [
        GenerationOptions(max_tokens=321, temperature=0.4, top_p=0.9, top_k=40)
    ]


def test_uploaded_security_dictionary_is_hot_reloaded_without_exposing_terms(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    runtime, store = _runtime(tmp_path, adapter)
    client = TestClient(create_app(runtime, store))

    uploaded = client.post(
        "/v1/security/dictionaries/import",
        json={"name": "customers", "terms": {"CUSTOMER": ["客户甲"]}},
    )
    dictionaries = client.get("/v1/security/dictionaries")
    completion = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "请联系客户甲"}]},
    )

    assert uploaded.status_code == 200
    assert uploaded.json()["term_count"] == 1
    assert "客户甲" not in dictionaries.text
    assert completion.status_code == 200
    assert adapter.calls[-1][1][0].content == "请联系[CUSTOMER_1]"

    dictionary_id = uploaded.json()["dictionary_id"]
    disabled = client.put(f"/v1/security/dictionaries/{dictionary_id}", json={"enabled": False})
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "请联系客户甲"}]},
    )

    assert disabled.status_code == 200
    assert adapter.calls[-1][1][0].content == "请联系客户甲"
