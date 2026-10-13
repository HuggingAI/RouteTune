# Enterprise pilot data pack

This directory contains a deliberately small RouteTune workflow pack, not a leaderboard.

- `enterprise-pilot-smoke-12.jsonl`: four records from each task family; run this first.
- `enterprise-pilot-full-30.jsonl`: ten records from each task family.
- `manifest.json`: exact source-file hashes and generated counts.

Each JSONL row contains the required `messages` array, a stable `id`, generation parameters, and evaluation metadata. `metadata.ground_truth` is shown only in the review UI and is never placed in the model prompt.

Source attribution:

- Banking77, PolyAI — banking customer-support intent classification, CC BY 4.0.
- MASSIVE, Amazon — the `zh-CN` test split for multilingual assistant intent classification, CC BY 4.0.
- FinQA — financial numerical reasoning over annual-report evidence, MIT.

The records are transformed subsets. See [`../../docs/ENTERPRISE_PILOT_DATASETS.md`](../../docs/ENTERPRISE_PILOT_DATASETS.md) for source URLs, pinned revisions, preparation commands, and the recommended RouteTune workflow.
