#!/usr/bin/env python3
"""Build a small, reproducible enterprise-oriented RouteTune pilot pack.

The source datasets are intentionally supplied as local files. This keeps the
transformation deterministic and avoids hiding network downloads in the script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

MASSIVE_INTENTS = [
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
]

BANKING_CATEGORIES = [
    "card_arrival",
    "card_delivery_estimate",
    "card_not_working",
    "cash_withdrawal_not_recognised",
    "cash_withdrawal_charge",
    "beneficiary_not_allowed",
    "pending_transfer",
    "declined_transfer",
    "wrong_exchange_rate_for_cash_withdrawal",
    "card_payment_not_recognised",
]

MASSIVE_CATEGORIES = [
    "email_sendemail",
    "calendar_set",
    "transport_ticket",
    "qa_stock",
    "iot_cleaning",
    "lists_createoradd",
    "weather_query",
    "recommendation_events",
    "takeaway_order",
    "qa_currency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--banking", type=Path, required=True)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--finqa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_one_per_label(
    rows: list[dict[str, Any]], labels: list[str], key: str
) -> list[dict[str, Any]]:
    selected = []
    for label in labels:
        try:
            selected.append(next(row for row in rows if row[key] == label))
        except StopIteration as exc:
            raise ValueError(f"source does not contain expected label: {label}") from exc
    return selected


def banking_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    all_labels = sorted({row["category"] for row in rows})
    prompt = (
        "You classify banking support requests. Reply with exactly one label and no "
        f"explanation. Allowed labels: {', '.join(all_labels)}"
    )
    records = []
    for index, row in enumerate(
        select_one_per_label(rows, BANKING_CATEGORIES, "category"), start=1
    ):
        records.append(
            {
                "id": f"banking77-{index:02d}-{row['category']}",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": row["text"]},
                ],
                "metadata": {
                    "source": "Banking77",
                    "source_split": "test",
                    "task_family": "customer_support_intent",
                    "difficulty": "easy",
                    "ground_truth_type": "exact_label",
                    "ground_truth": row["category"],
                    "license": "CC-BY-4.0",
                },
                "max_tokens": 512,
                "temperature": 0,
            }
        )
    return records


def massive_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("reading MASSIVE Parquet requires pyarrow") from exc

    rows = parquet.read_table(path, columns=["id", "utt", "intent"]).to_pylist()
    for row in rows:
        row["intent_name"] = MASSIVE_INTENTS[int(row["intent"])]
    prompt = (
        "你负责识别企业智能助手请求的意图。只返回一个英文标签，不要解释。"
        f"可选标签：{', '.join(MASSIVE_INTENTS)}"
    )
    records = []
    selected = select_one_per_label(rows, MASSIVE_CATEGORIES, "intent_name")
    for index, row in enumerate(selected, start=1):
        records.append(
            {
                "id": f"massive-zh-{index:02d}-{row['id']}",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(row["utt"])},
                ],
                "metadata": {
                    "source": "MASSIVE",
                    "source_split": "test",
                    "source_locale": "zh-CN",
                    "source_id": str(row["id"]),
                    "task_family": "enterprise_assistant_intent",
                    "difficulty": "medium",
                    "ground_truth_type": "exact_label",
                    "ground_truth": row["intent_name"],
                    "license": "CC-BY-4.0",
                },
                "max_tokens": 512,
                "temperature": 0,
            }
        )
    return records


def iter_finqa_questions(
    data: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result = []
    for report in data:
        questions = report.get("qa", [])
        if isinstance(questions, dict):
            questions = [questions]
        for question in questions:
            gold = question.get("gold_inds") or {}
            if (
                question.get("question")
                and str(question.get("answer", "")).strip()
                and 1 <= len(gold) <= 5
                and len(question["question"]) < 180
            ):
                result.append((report, question))
    return result


def finqa_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    candidates = iter_finqa_questions(data)
    if len(candidates) < 10:
        raise ValueError("FinQA source has fewer than ten suitable questions")
    prompt = (
        "You are a financial analyst. Answer only from the supplied annual-report "
        "evidence. Show one brief calculation when needed and finish with "
        "'Final answer: <value>'."
    )
    records = []
    for index, (report, question) in enumerate(candidates[:10], start=1):
        evidence = list((question.get("gold_inds") or {}).values())
        user_content = "Annual-report evidence:\n- " + "\n- ".join(evidence)
        user_content += f"\n\nQuestion: {question['question']}"
        records.append(
            {
                "id": f"finqa-{index:02d}",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content},
                ],
                "metadata": {
                    "source": "FinQA",
                    "source_split": "test",
                    "source_id": str(report.get("id") or report.get("filename") or index),
                    "task_family": "financial_numerical_reasoning",
                    "difficulty": "hard",
                    "ground_truth_type": "final_value",
                    "ground_truth": str(question["answer"]),
                    "reference_program": question.get("program"),
                    "license": "MIT",
                },
                "max_tokens": 512,
                "temperature": 0,
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_source = {
        "Banking77": banking_records(args.banking),
        "MASSIVE": massive_records(args.massive),
        "FinQA": finqa_records(args.finqa),
    }
    full = [record for records in by_source.values() for record in records]
    smoke = [record for records in by_source.values() for record in records[:4]]
    write_jsonl(args.output_dir / "enterprise-pilot-smoke-12.jsonl", smoke)
    write_jsonl(args.output_dir / "enterprise-pilot-full-30.jsonl", full)
    manifest = {
        "schema_version": 1,
        "generated_records": {"smoke": len(smoke), "full": len(full)},
        "source_files": {
            "Banking77": {"filename": args.banking.name, "sha256": sha256(args.banking)},
            "MASSIVE": {"filename": args.massive.name, "sha256": sha256(args.massive)},
            "FinQA": {"filename": args.finqa.name, "sha256": sha256(args.finqa)},
        },
        "counts_by_source": {name: len(records) for name, records in by_source.items()},
        "notes": "Ground truth is metadata only and is not included in model messages.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
