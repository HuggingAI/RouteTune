from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from routetune_core.adapters import OpenAICompatibleAdapter
from routetune_core.api_eval import (
    build_evaluation_records,
    load_traces,
    run_api_evaluation,
    write_evaluation_records,
)
from routetune_core.benchmark import (
    MMLU_PRO_LICENSE,
    MMLU_PRO_REVISION,
    download_mmlu_pro,
    load_benchmark_jsonl,
    load_mmlu_pro,
)
from routetune_core.contracts import ContractError
from routetune_core.dataset import load_jsonl, write_jsonl
from routetune_core.demo_data import generate_demo_records
from routetune_core.label_router import (
    DEFAULT_ROUTE_LABELS,
    LabelEngineBundle,
    LabelRouter,
    create_cold_start_label_bundle,
    train_label_bundle,
)
from routetune_core.model import EngineBundle, create_bootstrap_bundle
from routetune_core.model_pool import ModelPoolStore, RegistryAdapter
from routetune_core.providers import load_env_file, load_model_configs
from routetune_core.router import Router
from routetune_core.runtime import RouteTuneRuntime
from routetune_core.security import (
    DictionaryDetector,
    RegexDetector,
    SecurityPipeline,
    SecurityPolicy,
    SpacyDetector,
)
from routetune_core.service import create_app
from routetune_core.trace_store import TraceStore
from routetune_core.workflow import ExperimentResult, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="routetune",
        description="Train and validate a quality-constrained model router.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="generate synthetic data and run the MVP")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--records", type=int, default=600)
    demo.add_argument("--seed", type=int, default=42)
    _add_threshold_arguments(demo)

    run = subparsers.add_parser("run", help="train and evaluate from EvaluationRecord JSONL")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=42)
    _add_threshold_arguments(run)

    predict = subparsers.add_parser("predict", help="inspect one routing decision")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--request-id", default="interactive")
    predict.add_argument("--text", required=True)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="create a safe cold-start bundle from model configuration"
    )
    bootstrap.add_argument("--models", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, default=Path("artifacts/bootstrap/bundle"))

    label_bootstrap = subparsers.add_parser(
        "bootstrap-labels", help="create a cold-start fixed-label routing bundle"
    )
    label_bootstrap.add_argument(
        "--output", type=Path, default=Path("artifacts/bootstrap/label-bundle")
    )
    label_bootstrap.add_argument("--label", action="append", default=[])
    label_bootstrap.add_argument("--fallback-label")

    label_train = subparsers.add_parser(
        "train-labels", help="train a query-to-route-label classifier"
    )
    label_train.add_argument("--input", type=Path, required=True)
    label_train.add_argument("--output", type=Path, required=True)
    label_train.add_argument(
        "--model-label",
        action="append",
        required=True,
        help="mapping in MODEL_ID=ROUTE_LABEL form",
    )
    label_train.add_argument("--label", action="append", default=[])
    label_train.add_argument("--fallback-label")

    fetch = subparsers.add_parser(
        "fetch-mmlu-pro", help="download the pinned public MMLU-Pro test split"
    )
    fetch.add_argument("--output", type=Path, default=Path("data/mmlu-pro/test.parquet"))

    benchmark = subparsers.add_parser(
        "benchmark", help="call configured model APIs and create routing evaluation records"
    )
    benchmark.add_argument("--dataset", type=Path, required=True)
    benchmark.add_argument("--models", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark"))
    benchmark.add_argument("--env-file", type=Path)
    benchmark.add_argument("--max-items", type=int, default=20)
    benchmark.add_argument("--max-requests", type=int, default=40)
    benchmark.add_argument("--min-interval-seconds", type=float, default=0.0)
    benchmark.add_argument("--category", action="append", default=[])
    benchmark.add_argument("--seed", type=int, default=42)

    serve = subparsers.add_parser("serve", help="run the RouteTune HTTP service")
    serve_bundle = serve.add_mutually_exclusive_group(required=True)
    serve_bundle.add_argument("--label-bundle", type=Path)
    serve_bundle.add_argument("--bundle", type=Path, help="legacy model-bound bundle")
    serve.add_argument("--models", type=Path, required=True)
    serve.add_argument("--trace-db", type=Path, default=Path("data/routetune.db"))
    serve.add_argument("--env-file", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--storage-mode", choices=("metadata_only", "redacted"), default="redacted")
    serve.add_argument(
        "--dictionary",
        type=Path,
        action="append",
        default=[],
        help="JSON object mapping entity types to sensitive terms",
    )
    serve.add_argument(
        "--block-type",
        action="append",
        default=[],
        help="entity type to reject instead of only redact",
    )
    serve.add_argument("--spacy-model", help="optional installed spaCy model name")

    mock_litellm = subparsers.add_parser(
        "mock-litellm", help="run a local LiteLLM-compatible mock provider"
    )
    mock_litellm.add_argument("--host", default="127.0.0.1")
    mock_litellm.add_argument("--port", type=int, default=4000)

    return parser


def _add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quality-threshold", type=float, default=0.85)
    parser.add_argument("--ood-coverage-threshold", type=float, default=0.1)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        records = generate_demo_records(count=args.records, seed=args.seed)
        dataset_path = args.output / "demo-dataset.jsonl"
        write_jsonl(records, dataset_path)
        result = run_experiment(
            records,
            args.output,
            quality_threshold=args.quality_threshold,
            ood_coverage_threshold=args.ood_coverage_threshold,
            seed=args.seed,
        )
        _print_result(result, args.output)
        return 0

    if args.command == "run":
        records = load_jsonl(args.input)
        result = run_experiment(
            records,
            args.output,
            quality_threshold=args.quality_threshold,
            ood_coverage_threshold=args.ood_coverage_threshold,
            seed=args.seed,
        )
        _print_result(result, args.output)
        return 0

    if args.command == "predict":
        router = Router(EngineBundle.load(args.bundle))
        decision = router.decide(args.request_id, args.text)
        print(json.dumps(_decision_dict(decision), ensure_ascii=False, indent=2))
        return 0

    if args.command == "bootstrap":
        models = load_model_configs(args.models)
        bundle = create_bootstrap_bundle({item.model_id: item.cost_rank for item in models})
        bundle.save(args.output)
        print(f"Bundle: {args.output}")
        print(f"Fallback model: {bundle.fallback_model_id}")
        print("Mode: cold start (all requests use the highest cost-rank fallback)")
        return 0

    if args.command == "bootstrap-labels":
        labels = tuple(args.label or DEFAULT_ROUTE_LABELS)
        fallback = args.fallback_label or labels[-1]
        bundle = create_cold_start_label_bundle(labels, fallback_label=fallback)
        bundle.save(args.output)
        print(f"Label bundle: {args.output}")
        print(f"Labels: {', '.join(bundle.route_labels)}")
        print(f"Fallback label: {bundle.fallback_label}")
        return 0

    if args.command == "train-labels":
        labels = tuple(args.label or DEFAULT_ROUTE_LABELS)
        fallback = args.fallback_label or labels[-1]
        mappings: dict[str, str] = {}
        for raw_mapping in args.model_label:
            if "=" not in raw_mapping:
                raise ContractError("--model-label must use MODEL_ID=ROUTE_LABEL")
            model_id, route_label = raw_mapping.split("=", 1)
            mappings[model_id.strip()] = route_label.strip()
        bundle = train_label_bundle(
            load_jsonl(args.input),
            mappings,
            route_labels=labels,
            fallback_label=fallback,
        )
        bundle.save(args.output)
        print(f"Label bundle: {args.output}")
        print(f"Version: {bundle.version}")
        return 0

    if args.command == "fetch-mmlu-pro":
        path = download_mmlu_pro(args.output)
        print(f"Dataset: {path}")
        print(f"Revision: {MMLU_PRO_REVISION}")
        print(f"License: {MMLU_PRO_LICENSE}")
        return 0

    if args.command == "benchmark":
        if args.env_file:
            load_env_file(args.env_file)
        models = load_model_configs(args.models)
        if args.dataset.suffix == ".jsonl":
            items = load_benchmark_jsonl(args.dataset)
            if args.category:
                categories = {value.casefold() for value in args.category}
                items = [item for item in items if item.category.casefold() in categories]
            items = items[: args.max_items]
        else:
            items = load_mmlu_pro(
                args.dataset,
                max_items=args.max_items,
                categories=set(args.category),
                seed=args.seed,
            )
        trace_path = args.output / "traces.jsonl"
        summary = run_api_evaluation(
            items,
            models,
            trace_path,
            max_requests=args.max_requests,
            min_interval_seconds=args.min_interval_seconds,
        )
        records = build_evaluation_records(items, models, load_traces(trace_path))
        evaluation_path = args.output / "evaluation-records.jsonl"
        if len(records) >= 10:
            write_evaluation_records(records, evaluation_path)
        print(f"New API requests: {summary.new_request_count}")
        print(f"Cached results: {summary.cached_result_count}")
        print(f"Failed requests: {summary.failed_request_count}")
        print(f"Complete items: {summary.complete_item_count}/{summary.item_count}")
        print(f"Trace: {trace_path}")
        if len(records) >= 10:
            print(f"Evaluation records: {evaluation_path}")
        else:
            print("Evaluation records: not written (fewer than 10 complete items)")
        return 0

    if args.command == "serve":
        if args.env_file:
            load_env_file(args.env_file)
        model_configs = load_model_configs(args.models)
        model_pool = ModelPoolStore(args.trace_db)
        model_pool.seed(model_configs)
        if args.label_bundle:
            label_bundle = LabelEngineBundle.load(args.label_bundle)
            invalid_labels = {
                item["route_label"]
                for item in model_pool.list(enabled_only=True)
                if item["route_label"] not in label_bundle.route_labels
            }
            if invalid_labels:
                raise ContractError(
                    f"model pool uses labels outside bundle: {', '.join(sorted(invalid_labels))}"
                )
            router = LabelRouter(label_bundle)
            adapter = RegistryAdapter(model_pool)
        else:
            bundle = EngineBundle.load(args.bundle)
            bundle_ranks = {
                model_id: profile.cost_rank for model_id, profile in bundle.profiles.items()
            }
            configured_ranks = {item.model_id: item.cost_rank for item in model_configs}
            if bundle_ranks != configured_ranks:
                raise ContractError(
                    "bundle candidates and cost ranks must exactly match the model configuration"
                )
            router = Router(bundle)
            adapter = OpenAICompatibleAdapter(model_configs)
        detectors = [RegexDetector()]
        for dictionary_path in args.dictionary:
            raw_terms = json.loads(dictionary_path.read_text(encoding="utf-8"))
            if not isinstance(raw_terms, dict) or any(
                not isinstance(key, str)
                or not isinstance(values, list)
                or any(not isinstance(value, str) for value in values)
                for key, values in raw_terms.items()
            ):
                raise ValueError(f"invalid security dictionary: {dictionary_path}")
            detectors.append(DictionaryDetector(raw_terms))
        if args.spacy_model:
            detectors.append(SpacyDetector(args.spacy_model))

        traces = TraceStore(args.trace_db, storage_mode=args.storage_mode)
        runtime = RouteTuneRuntime(
            router=router,
            adapter=adapter,
            security=SecurityPipeline(
                detectors=detectors,
                policy=SecurityPolicy(
                    blocked_types=frozenset(value.upper() for value in args.block_type)
                ),
            ),
            traces=traces,
            model_pool=model_pool if args.label_bundle else None,
        )
        app = create_app(runtime, traces, model_pool=model_pool)
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if args.command == "mock-litellm":
        import uvicorn

        from routetune_core.mock_litellm import create_mock_litellm_app

        uvicorn.run(create_mock_litellm_app(), host=args.host, port=args.port)
        return 0

    raise AssertionError(f"unexpected command: {args.command}")


def _print_result(result: ExperimentResult, output: Path) -> None:
    routed = next(policy for policy in result.report.policies if policy.policy_name == "routetune")
    print(f"Bundle: {result.bundle.version}")
    print(
        f"Split: train={result.train_count}, calibration={result.calibration_count}, "
        f"test={result.test_count}"
    )
    print(f"RouteTune quality: {routed.quality_rate:.1%}")
    print(f"High-cost-model avoidance: {result.report.high_cost_model_avoidance_rate:.1%}")
    print(f"Report: {output / 'report.md'}")


def _decision_dict(decision: object) -> dict[str, object]:
    from dataclasses import asdict

    return asdict(decision)


if __name__ == "__main__":
    raise SystemExit(main())
