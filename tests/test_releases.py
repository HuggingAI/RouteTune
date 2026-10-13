from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from routetune_core.label_router import LabelRouter, create_cold_start_label_bundle
from routetune_core.model_pool import (
    ModelAvailabilityMonitor,
    ModelPoolEntry,
    ModelPoolStore,
    RegistryAdapter,
)
from routetune_core.providers import ApiModelConfig, Completion
from routetune_core.releases import DeploymentManager, ReleaseStore
from routetune_core.runtime import RouteTuneRuntime
from routetune_core.security import SecurityPipeline
from routetune_core.security_store import SecurityDictionaryStore
from routetune_core.service import create_app
from routetune_core.trace_store import TraceStore


class FakeProviderClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(
        self, model: ApiModelConfig, messages: list[dict[str, str]], **_: object
    ) -> Completion:
        model_id = model.model_id
        self.calls.append((model_id, messages[-1]["content"]))
        return Completion(
            content=f"answer from {model_id}",
            reasoning_content=None,
            prompt_tokens=4,
            completion_tokens=4,
            request_id=f"provider-{model_id}",
            latency_ms=1,
        )


def _entry(model_id: str, cost_rank: int, *, enabled: bool = True) -> ModelPoolEntry:
    return ModelPoolEntry(
        model_id=model_id,
        provider_id="test",
        base_url="https://example.test/v1",
        api_key_env="",
        api_model=model_id,
        cost_rank=cost_rank,
        max_tokens=100,
        route_label="tier_5",
        tags=("test",),
        enabled=enabled,
    )


def _versioned_app(tmp_path: Path) -> tuple[TestClient, DeploymentManager, FakeProviderClient]:
    database = tmp_path / "routetune.db"
    pool = ModelPoolStore(database)
    pool.create(_entry("cheap", 1))
    pool.create(_entry("strong", 2))
    provider = FakeProviderClient()
    router = LabelRouter(create_cold_start_label_bundle())
    traces = TraceStore(database)
    runtime = RouteTuneRuntime(
        router=router,
        adapter=RegistryAdapter(pool, client=provider),  # type: ignore[arg-type]
        security=SecurityPipeline(),
        traces=traces,
        model_pool=pool,
    )
    dictionaries = SecurityDictionaryStore(database)
    deployments = DeploymentManager(
        draft_runtime=runtime,
        model_pool=pool,
        dictionaries=dictionaries,
        traces=traces,
        releases=ReleaseStore(database, tmp_path / "releases"),
    )
    app = create_app(
        runtime,
        traces,
        model_pool=pool,
        security_dictionaries=dictionaries,
        deployments=deployments,
    )
    return TestClient(app), deployments, provider


def test_draft_model_changes_only_reach_unified_api_after_publish(tmp_path: Path) -> None:
    client, deployments, provider = _versioned_app(tmp_path)
    initial_release = deployments.active_release_id
    health = client.get("/health").json()
    assert health["provider_mode"] == "live"
    assert health["routing_targets"] == {"tier_5": "cheap"}

    first = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    ).json()
    assert first["model"] == "cheap"
    assert first["routetune"]["release_id"] == initial_release

    cheap = client.get("/v1/model-pool").json()["data"][0]
    cheap["enabled"] = False
    cheap.pop("api_key_available")
    assert client.put("/v1/model-pool/cheap", json=cheap).status_code == 200

    still_published = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "still old"}]},
    ).json()
    assert still_published["model"] == "cheap"

    candidate_response = client.post(
        "/v1/releases/candidates", json={"notes": "disable cheap in draft"}
    )
    assert candidate_response.status_code == 200
    candidate = candidate_response.json()
    published = client.post(f"/v1/releases/{candidate['release_id']}/publish")
    assert published.status_code == 200

    after_publish = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "now new"}]},
    ).json()
    assert after_publish["model"] == "strong"
    assert after_publish["routetune"]["release_id"] == candidate["release_id"]
    assert [call[0] for call in provider.calls] == ["cheap", "cheap", "strong"]


def test_provider_catalog_supports_multiple_custom_providers(tmp_path: Path) -> None:
    client, _, _ = _versioned_app(tmp_path)
    for provider_id in ("internal-east", "internal-west"):
        response = client.post(
            "/v1/model-pool",
            json={
                "model_id": f"{provider_id}-model",
                "provider_id": provider_id,
                "base_url": f"https://{provider_id}.example.test/v1",
                "api_key_env": f"{provider_id.upper().replace('-', '_')}_API_KEY",
                "api_model": "internal-model",
                "cost_rank": 10,
                "max_tokens": 256,
                "route_label": "tier_5",
                "tags": ["internal"],
                "enabled": True,
            },
        )
        assert response.status_code == 200

    providers = {
        item["provider_id"] for item in client.get("/v1/providers").json()["data"]
    }
    assert {"minimax", "internal-east", "internal-west"}.issubset(providers)


def test_model_availability_check_updates_operational_status(tmp_path: Path) -> None:
    client, _, _ = _versioned_app(tmp_path)

    checked = client.post("/v1/model-pool/cheap/check")
    model = next(
        item for item in client.get("/v1/model-pool").json()["data"]
        if item["model_id"] == "cheap"
    )

    assert checked.status_code == 200
    assert checked.json()["available"] is True
    assert model["availability_status"] == "available"
    assert model["availability_latency_ms"] == 1


def test_model_availability_monitor_checks_enabled_models_sequentially(tmp_path: Path) -> None:
    pool = ModelPoolStore(tmp_path / "routetune.db")
    pool.create(_entry("cheap", 1))
    pool.create(_entry("strong", 2))
    provider = FakeProviderClient()
    monitor = ModelAvailabilityMonitor(pool, provider, interval_seconds=60)  # type: ignore[arg-type]

    status = monitor.run_once()

    assert [call[0] for call in provider.calls] == ["cheap", "strong"]
    assert status["last_checked_count"] == 2
    assert status["last_available_count"] == 2
    assert status["checking"] is False


def test_release_artifact_download_contains_bundle_and_manifest(tmp_path: Path) -> None:
    client, deployments, _ = _versioned_app(tmp_path)

    response = client.get(
        f"/v1/releases/{deployments.active_release_id}/artifact"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "label-bundle.joblib",
            "label-manifest.json",
        }


def test_security_dictionary_is_frozen_inside_release(tmp_path: Path) -> None:
    client, _, provider = _versioned_app(tmp_path)

    uploaded = client.post(
        "/v1/security/dictionaries/import",
        json={"name": "projects", "terms": {"PROJECT": ["Moonshot"]}},
    )
    assert uploaded.status_code == 200

    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Moonshot"}]},
    )
    assert provider.calls[-1][1] == "Moonshot"

    candidate = client.post("/v1/releases/candidates", json={}).json()
    client.post(f"/v1/releases/{candidate['release_id']}/publish")
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Moonshot"}]},
    )
    assert provider.calls[-1][1] == "[PROJECT_1]"
