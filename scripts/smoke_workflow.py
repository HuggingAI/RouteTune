"""Exercise the complete RouteTune product loop against a running local demo service.

This is a mechanical release smoke test. Its annotations are synthetic and must not be
reported as model-quality evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_DATASET = Path("examples/enterprise-pilot/enterprise-pilot-smoke-12.jsonl")
DEFAULT_OUTPUT = Path("artifacts/smoke/workflow-report.json")
DEMO_MODELS = ("mock-fast", "mock-balanced", "mock-strong")


class SmokeFailure(RuntimeError):
    """Raised when a release-smoke invariant is not satisfied."""


class ApiClient:
    def __init__(self, base_url: str, *, api_key: str | None, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise SmokeFailure(f"{method} {path} returned HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise SmokeFailure(f"{method} {path} failed: {exc.reason}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _poll_batch(
    client: ApiClient, batch_id: str, *, deadline_seconds: float
) -> dict[str, Any]:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        batch = client.request("GET", f"/v1/batches/{batch_id}")
        if batch["status"] in {"completed", "failed"}:
            return batch
        time.sleep(0.2)
    raise SmokeFailure(f"batch {batch_id} did not finish within {deadline_seconds:g}s")


def _synthetic_pass(source: str, model_id: str) -> bool:
    """Create two learnable smoke labels without claiming output correctness."""
    if source == "FinQA":
        return model_id == "mock-strong"
    return True


def run_smoke(
    *, client: ApiClient, dataset_path: Path, batch_timeout: float
) -> dict[str, Any]:
    dataset_bytes = dataset_path.read_bytes()
    dataset_text = dataset_bytes.decode("utf-8")
    dataset_rows = [json.loads(line) for line in dataset_text.splitlines() if line.strip()]
    _require(len(dataset_rows) >= 10, "smoke dataset must contain at least 10 records")

    health_before = client.request("GET", "/health")
    _require(health_before["status"] == "ok", "service health is not ok")
    _require(health_before["provider_mode"] == "demo", "smoke test requires demo-only mode")
    _require(
        set(DEMO_MODELS).issubset(health_before["online_models"]),
        "all three demo models must be online",
    )
    stats_before = client.request("GET", "/v1/stats?window=all&source=all")

    cold_completion = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": [{"role": "user", "content": "请总结这份退款申请"}],
            "max_tokens": 128,
            "request_id": "release-smoke-cold-start",
        },
    )
    _require(cold_completion["routetune"]["trace_id"], "cold-start trace is missing")
    smoke_trace_ids = {str(cold_completion["routetune"]["trace_id"])}

    imported = client.request(
        "POST",
        "/v1/batches/import",
        {
            "name": "Release smoke · synthetic annotations",
            "jsonl": dataset_text,
            "mode": "compare",
            "model_ids": list(DEMO_MODELS),
            "max_tokens": 128,
            "temperature": 0,
        },
    )
    batch_id = str(imported["batch_id"])
    client.request("POST", f"/v1/batches/{batch_id}/run")
    batch = _poll_batch(client, batch_id, deadline_seconds=batch_timeout)
    expected_runs = len(dataset_rows) * len(DEMO_MODELS)
    progress = batch["progress"]
    _require(progress["expected"] == expected_runs, "unexpected batch run count")
    _require(progress["finished"] == expected_runs, "not all batch runs reached terminal state")
    _require(progress["failed"] == 0, "demo provider produced failed batch runs")

    annotation_count = 0
    for item in batch["items"]:
        source = str(item["metadata"].get("source", ""))
        for run in item["runs"]:
            trace = run.get("trace")
            _require(run["status"] == "completed" and trace, "completed run trace is missing")
            smoke_trace_ids.add(str(trace["trace_id"]))
            model_id = str(run["requested_model_id"])
            passed = _synthetic_pass(source, model_id)
            client.request(
                "POST",
                f"/v1/batches/{batch_id}/items/{item['item_id']}/feedback",
                {
                    "trace_id": trace["trace_id"],
                    "passed": passed,
                    "failure_reason": None if passed else "synthetic_smoke_quality_gate",
                    "score": 5 if passed else 2,
                    "comment": "Synthetic release smoke annotation; not quality evidence.",
                    "annotator_id": "release-smoke-script",
                    "retry_on_fail": False,
                },
            )
            annotation_count += 1

    reviewed = client.request("GET", f"/v1/batches/{batch_id}")
    _require(reviewed["progress"]["train_ready"], "fully annotated batch is not train-ready")
    _require(reviewed["progress"]["annotated"] == expected_runs, "annotation count mismatch")

    trained = client.request(
        "POST", f"/v1/batches/{batch_id}/train", {"strategy": "auto"}
    )
    _require(trained["trainer_id"] == "tfidf_logreg", "unexpected resolved trainer")
    _require(trained["training_record_count"] == len(dataset_rows), "training count mismatch")
    release_id = str(trained["release_id"])
    published = client.request("POST", f"/v1/releases/{release_id}/publish")
    _require(
        published["deployment"]["active_release_id"] == release_id,
        "published release did not become active",
    )

    health_after = client.request("GET", "/health")
    _require(health_after["release_id"] == release_id, "health reports the wrong release")
    _require(not health_after["cold_start"], "published trained bundle is still cold-start")

    learned_completion = client.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "auto",
            "messages": dataset_rows[0]["messages"],
            "max_tokens": 128,
            "temperature": 0,
            "request_id": "release-smoke-trained-router",
        },
    )
    _require(
        learned_completion["routetune"]["release_id"] == release_id,
        "post-publish trace is not bound to the active release",
    )
    _require(learned_completion["choices"][0]["message"]["content"], "empty completion")
    smoke_trace_ids.add(str(learned_completion["routetune"]["trace_id"]))

    stats = client.request("GET", "/v1/stats?window=all&source=all")
    traces = client.request("GET", "/v1/traces?limit=200")
    request_delta = stats["requests"]["total"] - stats_before["requests"]["total"]
    failed_delta = stats["requests"]["failed"] - stats_before["requests"]["failed"]
    prompt_token_delta = stats["tokens"]["prompt"] - stats_before["tokens"]["prompt"]
    completion_token_delta = (
        stats["tokens"]["completion"] - stats_before["tokens"]["completion"]
    )
    returned_trace_ids = {str(item["trace_id"]) for item in traces["data"]}
    _require(request_delta == expected_runs + 2, "unexpected smoke request delta")
    _require(failed_delta == 0, "smoke workflow contains failed traces")
    _require(smoke_trace_ids <= returned_trace_ids, "trace list is missing smoke requests")

    return {
        "schema_version": 1,
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "scope": "mechanical product-loop smoke; synthetic annotations are not quality evidence",
        "dataset": {
            "path": str(dataset_path),
            "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "records": len(dataset_rows),
        },
        "provider_mode": health_before["provider_mode"],
        "models": list(DEMO_MODELS),
        "cold_start": {
            "release_id": cold_completion["routetune"]["release_id"],
            "route_label": cold_completion["routetune"]["route_label"],
            "actual_model": cold_completion["routetune"]["actual_model"],
        },
        "batch": {
            "batch_id": batch_id,
            "records": len(dataset_rows),
            "expected_runs": expected_runs,
            "finished_runs": progress["finished"],
            "annotations": annotation_count,
        },
        "training": {
            "trainer_id": trained["trainer_id"],
            "training_record_count": trained["training_record_count"],
            "bundle_version": trained["bundle_version"],
        },
        "publication": {
            "release_id": release_id,
            "active_release_id": health_after["release_id"],
        },
        "post_publish": {
            "route_label": learned_completion["routetune"]["route_label"],
            "selected_model": learned_completion["routetune"]["selected_model"],
            "actual_model": learned_completion["routetune"]["actual_model"],
            "release_id": learned_completion["routetune"]["release_id"],
        },
        "observability": {
            "workflow_request_delta": request_delta,
            "workflow_failed_delta": failed_delta,
            "workflow_prompt_tokens": prompt_token_delta,
            "workflow_completion_tokens": completion_token_delta,
            "verified_trace_count": len(smoke_trace_ids),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--api-key")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--request-timeout", type=float, default=10)
    parser.add_argument("--batch-timeout", type=float, default=60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_smoke(
        client=ApiClient(
            args.base_url,
            api_key=args.api_key,
            timeout=args.request_timeout,
        ),
        dataset_path=args.dataset,
        batch_timeout=args.batch_timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"RouteTune release smoke: {report['status']}")
    print(f"Report: {args.output}")
    print(f"Release: {report['publication']['release_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
